"""CPU smoke test for DPTHeadPerLayer (per-layer refine + warp, deep supervision).

Validates:
  1. forward/backward for all 4 warp signals (rgb|feat|rgbfeat|hog);
  2. exactly 4 per-layer depths (z_l') and 4 pre-warp depths (z_l) are collected;
  3. gradients reach both the per-layer refine head and the warp (delta) head;
  4. main output == baseline temporal head (main output = original output_conv, unchanged).

Run:
    cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
    OMP_NUM_THREADS=4 /home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python scripts/smoke_perlayer.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dpt_temporal import DPTHeadTemporal
from model.dpt_perlayer import DPTHeadPerLayer


def _fake(in_ch, B=1, T=4, ph=8, pw=8):
    BT, L = B * T, ph * pw
    of = [(torch.randn(BT, L, in_ch), torch.randn(BT, in_ch)) for _ in range(4)]
    H0, W0 = ph * 14, pw * 14
    images = torch.rand(B, T, 3, H0, W0)
    ext = torch.eye(4).repeat(B, T, 1, 1)
    for t in range(T):
        ext[:, t, 0, 3] = 0.05 * t
    K = torch.eye(3).repeat(B, T, 1, 1)          # predicted intrinsics are (B,T,3,3)
    K[..., 0, 0] = K[..., 1, 1] = 200.0
    K[..., 0, 2], K[..., 1, 2] = W0 / 2, H0 / 2
    return of, images, ext, K, B, T, ph, pw


def test_forward(sig):
    ic, ft, oc = 64, 64, [48, 96, 192, 384]
    head = DPTHeadPerLayer(ic, ft, False, oc, False, num_frames=4, warp_signal=sig)
    head.train()
    of, im, ext, K, B, T, ph, pw = _fake(ic)
    d = head(of, ph, pw, T, images=im, extrinsics=ext, intrinsics=K)
    assert d.shape[0] == B * T and d.shape[1] == 1, d.shape
    assert len(head.layer_depths) == 4, f"expect 4 layer depths, got {len(head.layer_depths)}"
    assert len(head.layer_depths_pre) == 4, f"expect 4 pre-warp depths, got {len(head.layer_depths_pre)}"
    ds = sum(z.float().mean() for z in head.layer_depths)   # deep supervision proxy
    loss = d.float().mean() + ds
    loss.backward()
    gr = head.layer_depth_heads['p2'][0].weight.grad
    gd = head.layer_delta_heads['p2'][-1].weight.grad       # last conv is zero-init but drives dz
    assert gr is not None and gr.abs().sum() > 0, "refine head got no grad"
    assert gd is not None and gd.abs().sum() > 0, "warp/delta head got no grad"
    print(f"[perlayer/{sig}] fwd/bwd OK shape={tuple(d.shape)} layers={len(head.layer_depths)} (refine+warp in-graph)")


def test_main_equals_baseline():
    ic, ft, oc = 64, 64, [48, 96, 192, 384]
    head = DPTHeadPerLayer(ic, ft, False, oc, False, num_frames=4, warp_signal='feat')
    base = DPTHeadTemporal(ic, ft, False, oc, False, num_frames=4)
    missing, unexpected = base.load_state_dict(head.state_dict(), strict=False)
    assert len(missing) == 0, f"baseline missing shared keys: {missing[:5]}"
    head.eval()
    base.eval()
    of, im, ext, K, B, T, ph, pw = _fake(ic)
    with torch.no_grad():
        d_head = head(of, ph, pw, T, images=im, extrinsics=ext, intrinsics=K)
        d_base = base(of, ph, pw, T)
    md = (d_head - d_base).abs().max().item()
    assert torch.allclose(d_head, d_base, atol=1e-5), f"main output != baseline, max_diff={md}"
    print(f"[perlayer] main output == baseline OK (max_diff={md:.2e})")


if __name__ == '__main__':
    for sig in ['rgb', 'feat', 'rgbfeat', 'hog']:
        test_forward(sig)
    test_main_equals_baseline()
    print("ALL OK")
