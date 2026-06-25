"""CPU smoke test for the co-attention error-map head (方案 C), all four modality arms.

Validates, for each of {rgb, feat, hog, rgbfeat}:
  1. signal_error_map / hog_feature_map shapes + gradient flow through depth.
  2. DPTHeadErrorMapCoAttn forward/backward, aux_depths population, and gradients reaching
     the depth heads, modality encoders, and the co-attention output projection.
  3. Zero-init identity: at init the co-attention injection is a no-op (out_proj is zero), so
     the head output matches the baseline temporal head output.

Run:
    cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
    /home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python scripts/smoke_coattn.py
"""
import os
import sys

import torch

# Cap CPU threads: the login node is heavily shared, and torch grabbing all cores causes
# oversubscription that makes this tiny test crawl. 4 threads is plenty for the smoke sizes.
torch.set_num_threads(min(4, torch.get_num_threads()))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.util.warp import signal_error_map
from model.util.hog import hog_feature_map
from model.dpt_errormap_coattn import DPTHeadErrorMapCoAttn, MODALITY_PRESETS
from model.dpt_temporal import DPTHeadTemporal


def _synthetic_batch(in_ch, B=1, T=4, ph=8, pw=8):
    BT, L = B * T, ph * pw
    out_features = [(torch.randn(BT, L, in_ch), torch.randn(BT, in_ch)) for _ in range(4)]
    H0, W0 = ph * 14, pw * 14
    images = torch.rand(B, T, 3, H0, W0)
    ext = torch.eye(4).repeat(B, T, 1, 1)
    for t in range(T):
        ext[:, t, 0, 3] = 0.05 * t
    K = torch.eye(3).repeat(B, T, 1, 1)
    K[..., 0, 0] = K[..., 1, 1] = 200.0
    K[..., 0, 2], K[..., 1, 2] = W0 / 2, H0 / 2
    return out_features, images, ext, K, B, T, ph, pw


def test_signal_and_hog():
    B, T, h, w = 1, 4, 16, 16
    feat = torch.rand(B, T, 32, h, w)
    depth = (torch.rand(B, T, 1, h, w) * 10 + 1).requires_grad_(True)
    K = torch.eye(3).repeat(B, T, 1, 1)
    K[..., 0, 0] = K[..., 1, 1] = 20.0
    K[..., 0, 2], K[..., 1, 2] = w / 2, h / 2
    ext = torch.eye(4).repeat(B, T, 1, 1)
    for t in range(T):
        ext[:, t, 0, 3] = 0.1 * t

    err, valid = signal_error_map(feat, depth, K, ext, offsets=(-1, 1))
    assert err.shape == (B, T, 1, h, w), err.shape
    err.sum().backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()
    print(f"[signal] feature-error OK err.shape={tuple(err.shape)} "
          f"grad_sum={depth.grad.abs().sum().item():.4f}")

    imgs = torch.rand(B, T, 3, h, w)
    hog = hog_feature_map(imgs, nbins=9)
    assert hog.shape == (B, T, 9, h, w), hog.shape
    assert torch.isfinite(hog).all()
    print(f"[hog] OK hog.shape={tuple(hog.shape)}")


def test_arm(modality):
    in_ch, feats, out_ch = 64, 64, [48, 96, 192, 384]
    torch.manual_seed(0)
    head = DPTHeadErrorMapCoAttn(in_ch, feats, False, out_ch, False, num_frames=4,
                                 error_modalities=modality, attn_grid=4)
    head.train()
    out_features, images, ext, K, B, T, ph, pw = _synthetic_batch(in_ch)
    BT = B * T

    # Zero-init identity vs baseline temporal head (share the inherited backbone weights).
    torch.manual_seed(0)
    base = DPTHeadTemporal(in_ch, feats, False, out_ch, False, num_frames=4)
    base.load_state_dict({k: v for k, v in head.state_dict().items()
                          if k in base.state_dict()}, strict=True)
    head.eval(); base.eval()
    with torch.no_grad():
        d_head = head(out_features, ph, pw, T, images=images, extrinsics=ext, intrinsics=K)
        d_base = base(out_features, ph, pw, T)
    max_diff = (d_head - d_base).abs().max().item()
    assert max_diff < 1e-4, f"[{modality}] not identity at init: max_diff={max_diff}"
    # Zero-init: the co-attention injection must be a no-op at init.
    for key in head.error_stages:
        assert head.coattn[key].out_proj.weight.abs().sum().item() == 0.0
    head.train()

    # Activate the residual branch (emulate a trained state) so we can verify that gradient
    # flows all the way through the co-attention into the modality encoders. With the zero-init
    # out_proj, the whole upstream branch correctly receives no gradient at step 0.
    with torch.no_grad():
        for key in head.error_stages:
            head.coattn[key].out_proj.weight.normal_(0.0, 0.02)

    depth = head(out_features, ph, pw, T, images=images, extrinsics=ext, intrinsics=K)
    assert depth.shape[0] == BT and depth.shape[1] == 1, depth.shape
    assert len(head.aux_depths) == len(head.error_stages)

    loss = depth.float().mean() + sum(a.float().mean() for a in head.aux_depths)
    loss.backward()

    g_out = head.coattn['s4'].out_proj.weight.grad
    g_attn = head.coattn['s4'].attn.in_proj_weight.grad
    g_dep = head.depth_heads['s4'][0].weight.grad
    m0 = head.modalities[0]
    g_enc = head.modality_encoders[f's4_{m0}'][-1].weight.grad
    assert g_out is not None and g_out.abs().sum().item() > 0, "co-attn out_proj got no gradient"
    assert g_attn is not None and g_attn.abs().sum().item() > 0, "co-attn attention got no gradient"
    assert g_dep is not None and g_dep.abs().sum().item() > 0, "depth head got no gradient"
    assert g_enc is not None and g_enc.abs().sum().item() > 0, "modality encoder got no gradient"
    print(f"[{modality}] OK identity(max_diff={max_diff:.2e}) modalities={head.modalities} "
          f"grad(attn)={g_attn.abs().sum().item():.3e} grad(enc)={g_enc.abs().sum().item():.3e}")


if __name__ == '__main__':
    torch.manual_seed(0)
    test_signal_and_hog()
    for arm in MODALITY_PRESETS:
        test_arm(arm)
    print("ALL CO-ATTENTION SMOKE TESTS PASSED")
