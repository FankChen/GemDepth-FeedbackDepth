"""CPU smoke test for the refine error-map DPT head (dpt_errormap_refine).

Validates:
  1. forward/backward, aux_depths population (2 per stage: depth1 + depth2).
  2. Zero-init equivalence: with fuse_blocks zero-initialised the head output is identical
     to the baseline temporal head (injection is a no-op at init).
  3. Gradients reach the new modules (fuse_blocks / depth_heads1 / depth_heads2).

Run:
    cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
    /home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python scripts/smoke_refine.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dpt_temporal import DPTHeadTemporal
from model.dpt_errormap_refine import DPTHeadErrorMapRefine


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


def test_forward_and_aux():
    in_ch, feats, out_ch = 64, 64, [48, 96, 192, 384]
    head = DPTHeadErrorMapRefine(in_ch, feats, False, out_ch, False, num_frames=4)
    head.train()
    out_features, images, ext, K, B, T, ph, pw = _fake_inputs(in_ch)
    BT = B * T

    depth = head(out_features, ph, pw, T, images=images, extrinsics=ext, intrinsics=K)
    assert depth.shape[0] == BT and depth.shape[1] == 1, depth.shape
    assert len(head.aux_depths) == 2 * len(head.error_stages), len(head.aux_depths)
    print(f"[refine] forward OK depth.shape={tuple(depth.shape)} "
          f"n_aux={len(head.aux_depths)} (expect {2 * len(head.error_stages)}) "
          f"stages={head.error_stages}")

    loss = depth.float().mean() + sum(a.float().mean() for a in head.aux_depths)
    loss.backward()
    for key in ('s4', 's1'):
        g_fuse = head.fuse_blocks[key][-1].weight.grad
        g_d1 = head.depth_heads1[key][0].weight.grad
        g_d2 = head.depth_heads2[key][0].weight.grad
        assert g_fuse is not None and g_fuse.abs().sum() > 0, f"{key} fuse no grad"
        assert g_d1 is not None and g_d1.abs().sum() > 0, f"{key} depth1 no grad"
        assert g_d2 is not None and g_d2.abs().sum() > 0, f"{key} depth2 no grad"
    print("[refine] gradients reach fuse_blocks / depth_heads1 / depth_heads2 OK")


def test_zero_init_equivalence():
    in_ch, feats, out_ch = 64, 64, [48, 96, 192, 384]
    head = DPTHeadErrorMapRefine(in_ch, feats, False, out_ch, False, num_frames=4)
    base = DPTHeadTemporal(in_ch, feats, False, out_ch, False, num_frames=4)
    # Copy the shared (temporal) weights so only the refine modules differ.
    missing, unexpected = base.load_state_dict(head.state_dict(), strict=False)
    assert len(missing) == 0, f"baseline missing shared keys: {missing[:5]}"
    head.eval()
    base.eval()
    out_features, images, ext, K, B, T, ph, pw = _fake_inputs(in_ch)

    with torch.no_grad():
        d_refine = head(out_features, ph, pw, T, images=images, extrinsics=ext, intrinsics=K)
        d_base = base(out_features, ph, pw, T)
    max_diff = (d_refine - d_base).abs().max().item()
    assert torch.allclose(d_refine, d_base, atol=1e-5), f"not equivalent, max_diff={max_diff}"
    print(f"[refine] zero-init equivalence to baseline OK (max_diff={max_diff:.2e})")


if __name__ == '__main__':
    torch.manual_seed(0)
    test_forward_and_aux()
    test_zero_init_equivalence()
    print("ALL OK")
