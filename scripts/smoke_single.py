"""CPU smoke test for the single-stage error-map DPT head (dpt_errormap_single).

Validates:
  1. forward/backward, aux_depths population (exactly 1: the coarse z1).
  2. Zero-init equivalence: with fuse_block zero-initialised the head output is identical
     to the baseline temporal head (the single injection is a no-op at init).
  3. Gradients reach depth_head1 (via the aux loss) and fuse_block (via the main path).
  4. The error_encoder -> fuse path is learnable: once fuse_block's last conv is perturbed
     away from zero, gradients reach error_encoder too.
  5. The 'feat' (feature-metric) warp variant also runs forward/backward.

Run:
    cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
    OMP_NUM_THREADS=4 /home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python scripts/smoke_single.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dpt_temporal import DPTHeadTemporal
from model.dpt_errormap_single import DPTHeadErrorMapSingle


def _fake_inputs(in_ch, B=1, T=4, ph=8, pw=8):
    BT, L = B * T, ph * pw
    out_features = [(torch.randn(BT, L, in_ch), torch.randn(BT, in_ch)) for _ in range(4)]
    H0, W0 = ph * 14, pw * 14
    images = torch.rand(B, T, 3, H0, W0)
    ext = torch.eye(4).repeat(B, T, 1, 1)
    for t in range(T):
        ext[:, t, 0, 3] = 0.05 * t  # small horizontal camera translation per frame
    K = torch.eye(3).repeat(B, T, 1, 1)
    K[..., 0, 0] = K[..., 1, 1] = 200.0
    K[..., 0, 2], K[..., 1, 2] = W0 / 2, H0 / 2
    return out_features, images, ext, K, B, T, ph, pw


def test_forward_and_aux(warp_signal='rgb'):
    in_ch, feats, out_ch = 64, 64, [48, 96, 192, 384]
    head = DPTHeadErrorMapSingle(in_ch, feats, False, out_ch, False, num_frames=4,
                                 warp_signal=warp_signal)
    head.train()
    out_features, images, ext, K, B, T, ph, pw = _fake_inputs(in_ch)
    BT = B * T

    depth = head(out_features, ph, pw, T, images=images, extrinsics=ext, intrinsics=K)
    assert depth.shape[0] == BT and depth.shape[1] == 1, depth.shape
    assert len(head.aux_depths) == 1, f"expected 1 aux depth (z1), got {len(head.aux_depths)}"
    print(f"[single/{warp_signal}] forward OK depth.shape={tuple(depth.shape)} "
          f"n_aux={len(head.aux_depths)} (expect 1)")

    loss = depth.float().mean() + sum(a.float().mean() for a in head.aux_depths)
    # z1 (aux) must be a differentiable function of the ORIGINAL output_conv path.
    assert head.aux_depths[0].requires_grad, "z1 aux is detached from the graph"
    loss.backward()
    g_refine = head.refine_head[-1].weight.grad
    assert g_refine is not None and g_refine.abs().sum() > 0, "refine_head last conv no grad"
    print(f"[single/{warp_signal}] gradients reach refine_head; z1 aux in-graph OK")


def test_error_encoder_learnable():
    """At zero-init the error_encoder grad is masked by refine_head's zero last conv.
    Give refine_head non-zero weights + positive biases (keeps both ReLUs active) and confirm
    the error_encoder -> refine_head path then carries gradient."""
    in_ch, feats, out_ch = 64, 64, [48, 96, 192, 384]
    head = DPTHeadErrorMapSingle(in_ch, feats, False, out_ch, False, num_frames=4)
    with torch.no_grad():
        for idx in (0, 2, 4):                       # the three Conv2d layers in refine_head
            head.refine_head[idx].weight.normal_(0, 0.05)
            head.refine_head[idx].bias.fill_(0.5)   # positive bias keeps the ReLUs active
    head.train()
    out_features, images, ext, K, B, T, ph, pw = _fake_inputs(in_ch)
    depth = head(out_features, ph, pw, T, images=images, extrinsics=ext, intrinsics=K)
    depth.float().mean().backward()
    g_enc = head.error_encoder[-1].weight.grad
    assert g_enc is not None and g_enc.abs().sum() > 0, "error_encoder no grad after activating refine_head"
    print("[single] error_encoder -> refine_head path is learnable OK")


def test_z1_detached_from_main():
    """z1 is detached in _refine, so the MAIN loss (on z'1) must NOT reach output_conv2 (which
    produces z1). z1 is trained only by its own auxiliary loss."""
    in_ch, feats, out_ch = 64, 64, [48, 96, 192, 384]
    head = DPTHeadErrorMapSingle(in_ch, feats, False, out_ch, False, num_frames=4)
    with torch.no_grad():
        for idx in (0, 2, 4):                       # activate refine so the main loss has a real path
            head.refine_head[idx].weight.normal_(0, 0.05)
            head.refine_head[idx].bias.fill_(0.5)
    head.train()
    out_features, images, ext, K, B, T, ph, pw = _fake_inputs(in_ch)
    depth = head(out_features, ph, pw, T, images=images, extrinsics=ext, intrinsics=K)  # z'1 (main)
    depth.float().mean().backward()
    g_oc2 = head.scratch.output_conv2[0].weight.grad
    assert g_oc2 is None or g_oc2.abs().sum() == 0, "z1 NOT detached: main loss reached output_conv2"
    print("[single] z1 detached: main loss does not reach output_conv2 OK")


def test_zero_init_equivalence():
    in_ch, feats, out_ch = 64, 64, [48, 96, 192, 384]
    head = DPTHeadErrorMapSingle(in_ch, feats, False, out_ch, False, num_frames=4)
    base = DPTHeadTemporal(in_ch, feats, False, out_ch, False, num_frames=4)
    # Copy the shared (temporal) weights so only the error-map modules differ.
    missing, unexpected = base.load_state_dict(head.state_dict(), strict=False)
    assert len(missing) == 0, f"baseline missing shared keys: {missing[:5]}"
    head.eval()
    base.eval()
    out_features, images, ext, K, B, T, ph, pw = _fake_inputs(in_ch)

    with torch.no_grad():
        d_single = head(out_features, ph, pw, T, images=images, extrinsics=ext, intrinsics=K)
        d_base = base(out_features, ph, pw, T)
    max_diff = (d_single - d_base).abs().max().item()
    assert torch.allclose(d_single, d_base, atol=1e-5), f"not equivalent, max_diff={max_diff}"
    print(f"[single] zero-init equivalence to baseline OK (max_diff={max_diff:.2e})")


if __name__ == '__main__':
    torch.manual_seed(0)
    test_forward_and_aux('rgb')
    test_forward_and_aux('feat')
    test_error_encoder_learnable()
    test_z1_detached_from_main()
    test_zero_init_equivalence()
    print("ALL OK")
