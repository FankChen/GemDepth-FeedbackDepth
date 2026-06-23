"""Smoke test for GemDepth training: synthetic data, 1 GPU, 5 steps.

Verifies:
  - GemDepth(vitl) constructs and loads released ckpt strict=True
  - Backbone freeze + 2-group LR setup matches train.py
  - forward(image[B,T,3,H,W]) -> (depth, pose_enc_list, extrinsic_pred, intrinsic_pred)
  - VideoDepthLoss runs without NaN
  - autocast(bf16) + backward + AdamW step succeeds
  - GPU memory is reasonable
"""
import os, sys, time, torch, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model.gemdepth import GemDepth
from loss.videoloss import VideoDepthLoss

# CPU mode if no GPU or SMOKE_CPU=1
USE_CPU = (os.environ.get('SMOKE_CPU', '') == '1') or (not torch.cuda.is_available())
DEVICE = 'cpu' if USE_CPU else 'cuda'
USE_AUTOCAST = not USE_CPU  # bf16 autocast only on GPU
torch.manual_seed(0); np.random.seed(0)

# --- smoke config ---
if USE_CPU:
    # tiny for CPU; H must be multiple of 14 (DINOv2 patch)
    B, T, H, W = 1, 2, 14*8, 14*8   # 112x112, T=2
    N_STEPS = 2
    torch.set_num_threads(min(16, os.cpu_count() or 1))
else:
    B, T, H, W = 1, 4, 518, 518
    N_STEPS = 5
CKPT = './checkpoint/gemdepth.pth'

print(f"[smoke] config: B={B} T={T} H={H} W={W} steps={N_STEPS} device={DEVICE}")
print(f"[smoke] torch={torch.__version__} cuda={torch.cuda.is_available()} autocast_bf16={USE_AUTOCAST}")

# Match train.py vitl config
model_cfg = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
print('[smoke] building model...')
model = GemDepth(**model_cfg).to(DEVICE)
print(f'[smoke] loading ckpt {CKPT}')
sd = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=True)
model.pretrained.requires_grad_(False)

# CPU has no flash-attn kernel; flip every block to the model's built-in naive path.
# Note: pytorch_naive disallows attn_mask, but Blocks here use mask=None so this is safe.
if USE_CPU:
    n_patched = 0
    for m in model.modules():
        if hasattr(m, 'attn_implementation'):
            if getattr(m, 'attn_mask', None) is not None:
                continue  # leave masked-attn blocks alone (would assert)
            m.attn_implementation = 'pytorch_naive'
            n_patched += 1
    print(f"[smoke] patched {n_patched} attention modules to pytorch_naive (CPU)")

# Optimizer groups (same as train.py)
dec_blocks_params, other_params = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    (dec_blocks_params if n.startswith('spatial_blocks') or n.startswith('time_blocks') else other_params).append(p)
print(f"[smoke] trainable params: dec_blocks={sum(p.numel() for p in dec_blocks_params)/1e6:.1f}M "
      f"other={sum(p.numel() for p in other_params)/1e6:.1f}M "
      f"frozen={sum(p.numel() for p in model.pretrained.parameters())/1e6:.1f}M")

optimizer = torch.optim.AdamW(
    [{'params': dec_blocks_params, 'lr': 1e-5},
     {'params': other_params,      'lr': 1e-6}],
    weight_decay=0.01,
)

loss_fn = VideoDepthLoss(pose_flag=True).to(DEVICE)

# --- synthetic batch ---
def make_batch():
    image = torch.rand(B, T, 3, H, W, device=DEVICE) * 2 - 1  # in [-1,1]
    # synthetic depth: smooth + positive (1..50m range)
    depth = (torch.rand(B, T, H, W, device=DEVICE) * 49 + 1)
    mask = (depth > 0).float()
    # intrinsics K (kitti-like, scaled to 518)
    fx, fy, cx, cy = 725.0, 725.0, W/2, H/2
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=DEVICE).unsqueeze(0).repeat(B, 1, 1)
    # poses: list of T tensors[B,4,4], identity + small random translation
    poses = []
    for t in range(T):
        P = torch.eye(4, device=DEVICE).unsqueeze(0).repeat(B, 1, 1)
        P[:, :3, 3] = torch.randn(B, 3, device=DEVICE) * 0.1 * t
        poses.append(P)
    return image, depth, mask, K, poses

model.train()
if not USE_CPU:
    torch.cuda.reset_peak_memory_stats()

# On CPU the optimized SDPA kernels (flash/mem-efficient) are unavailable.
# Force the math fallback so attention works everywhere.
if USE_CPU:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        sdpa_ctx = lambda: sdpa_kernel([SDPBackend.MATH])
    except ImportError:
        from contextlib import contextmanager
        @contextmanager
        def sdpa_ctx():
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                yield
else:
    from contextlib import nullcontext
    sdpa_ctx = nullcontext

for step in range(N_STEPS):
    image, depth_gt, mask, K, poses = make_batch()
    t0 = time.time()
    with sdpa_ctx():
        if USE_AUTOCAST:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                depth_pred, pose_enc_list, extrinsic_pred, intrinsic_pred = model(image)
        else:
            depth_pred, pose_enc_list, extrinsic_pred, intrinsic_pred = model(image)
        # depth_pred shape: [B,T,1,H,W] (per dataset) — train.py squeezes dim=2
        dp = depth_pred.squeeze(2) if depth_pred.ndim == 5 else depth_pred
        loss_dict = loss_fn(dp, depth_gt, mask, K, poses, pose_enc_list, extrinsic_pred)
        loss = loss_dict['total_loss']
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    dt = time.time() - t0
    mem_str = ''
    if not USE_CPU:
        mem_str = f' | peakMem={torch.cuda.max_memory_allocated()/1024**3:.1f}GiB'
    items = ' '.join(f"{k}={v.item():.4f}" for k, v in loss_dict.items() if torch.is_tensor(v))
    print(f"[smoke] step {step} loss={loss.item():.4f} | {items} | {dt:.2f}s{mem_str}", flush=True)
    assert not torch.isnan(loss), "NaN loss!"

print("[smoke] PASSED")
