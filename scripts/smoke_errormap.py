"""CPU smoke test for the error-map DPT components (no flash-attn / DINOv2 required).

Validates:
  1. photometric_error_map shapes + gradient flow through depth.
  2. DPTHeadErrorMap forward/backward, aux_depths population, and gradients reaching
     the error-map modules.

Run:
    cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
    /home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python scripts/smoke_errormap.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.util.warp import photometric_error_map, scale_intrinsics
from model.dpt_errormap import DPTHeadErrorMap


def test_warp():
    B, T, C, h, w = 1, 4, 3, 16, 16
    imgs = torch.rand(B, T, C, h, w)
    depth = (torch.rand(B, T, 1, h, w) * 10 + 1).requires_grad_(True)
    K = torch.eye(3).repeat(B, T, 1, 1)
    K[..., 0, 0] = K[..., 1, 1] = 20.0
    K[..., 0, 2], K[..., 1, 2] = w / 2, h / 2
    ext = torch.eye(4).repeat(B, T, 1, 1)
    for t in range(T):
        ext[:, t, 0, 3] = 0.1 * t  # small horizontal camera translation per frame

    err, valid = photometric_error_map(imgs, depth, K, ext, offsets=(-1, 1))
    assert err.shape == (B, T, 1, h, w), err.shape
    assert valid.shape == (B, T, 1, h, w), valid.shape
    err.sum().backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()
    print(f"[warp] OK err.shape={tuple(err.shape)} valid_frac={valid.mean().item():.3f} "
          f"depth.grad_abs_sum={depth.grad.abs().sum().item():.4f}")

    K2 = scale_intrinsics(K, (h, w), (h // 2, w // 2))
    assert torch.allclose(K2[..., 0, 0], K[..., 0, 0] * 0.5)
    print("[warp] scale_intrinsics OK")


def test_head():
    in_ch, feats, out_ch = 64, 64, [48, 96, 192, 384]
    head = DPTHeadErrorMap(in_ch, feats, False, out_ch, False, num_frames=4)
    head.train()

    B, T, ph, pw = 1, 4, 8, 8
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

    depth = head(out_features, ph, pw, T, images=images, extrinsics=ext, intrinsics=K)
    assert depth.shape[0] == BT and depth.shape[1] == 1, depth.shape
    assert len(head.aux_depths) == len(head.error_stages)
    print(f"[head] forward OK depth.shape={tuple(depth.shape)} "
          f"aux={[tuple(a.shape) for a in head.aux_depths]}")

    loss = depth.float().mean() + sum(a.float().mean() for a in head.aux_depths)
    loss.backward()

    g_enc = head.error_encoders['s4'][-1].weight.grad
    g_dep = head.depth_heads['s4'][0].weight.grad
    assert g_enc is not None and g_enc.abs().sum().item() > 0, "error encoder got no gradient"
    assert g_dep is not None and g_dep.abs().sum().item() > 0, "depth head got no gradient"
    print(f"[head] backward OK grad(error_enc last conv)={g_enc.abs().sum().item():.4e} "
          f"grad(depth_head first conv)={g_dep.abs().sum().item():.4e}")


if __name__ == '__main__':
    torch.manual_seed(0)
    test_warp()
    test_head()
    print("ALL SMOKE TESTS PASSED")
