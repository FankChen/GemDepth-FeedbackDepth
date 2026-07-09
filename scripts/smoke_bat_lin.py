"""CPU smoke test for DPTHeadBATLin (warp error-map + BA-T/LinStereo multi-scale iterative delta).

Validates:
  1. forward/backward for all 4 warp signals (rgb|feat|rgbfeat|hog);
  2. configurable scales ("any #layers"): 2-scale (p2,p1) and 4-scale (p4..p1);
  3. gradients reach the shared BA update block;
  4. zero-init equivalence: at init the head output == baseline temporal head exactly
     (warm-start kept, every gated delta is a no-op).

Run:
    cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
    OMP_NUM_THREADS=4 /home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python scripts/smoke_bat_lin.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dpt_temporal import DPTHeadTemporal
from model.dpt_bat_lin import DPTHeadBATLin


def _fake(in_ch, B=1, T=4, ph=8, pw=8):
    BT, L = B * T, ph * pw
    of = [(torch.randn(BT, L, in_ch), torch.randn(BT, in_ch)) for _ in range(4)]
    H0, W0 = ph * 14, pw * 14
    images = torch.rand(B, T, 3, H0, W0)
    ext = torch.eye(4).repeat(B, T, 1, 1)
    for t in range(T):
        ext[:, t, 0, 3] = 0.05 * t
    K = torch.eye(3).repeat(B, T, 1, 1)
    K[..., 0, 0] = K[..., 1, 1] = 200.0
    K[..., 0, 2], K[..., 1, 2] = W0 / 2, H0 / 2
    return of, images, ext, K, B, T, ph, pw


def test_forward(sig, scales):
    ic, ft, oc = 64, 64, [48, 96, 192, 384]
    head = DPTHeadBATLin(ic, ft, False, oc, False, num_frames=4, warp_signal=sig, scales=scales)
    head.train()
    of, im, ext, K, B, T, ph, pw = _fake(ic)
    d = head(of, ph, pw, T, images=im, extrinsics=ext, intrinsics=K)
    assert d.shape[0] == B * T and d.shape[1] == 1, d.shape
    assert len(head.aux_depths) == 1, f"expect 1 aux (z1), got {len(head.aux_depths)}"
    loss = d.float().mean() + head.aux_depths[0].float().mean()
    loss.backward()
    # delta conv is zero-init but still receives grad (d z / d delta = gate > 0).
    gd = head.ba_update.delta.weight.grad
    assert gd is not None and gd.abs().sum() > 0, "ba_update.delta got no grad"
    print(f"[batlin/{sig} scales={scales}] fwd/bwd OK shape={tuple(d.shape)} (BA-update in-graph)")


def test_zeroinit_equiv(sig, scales):
    ic, ft, oc = 64, 64, [48, 96, 192, 384]
    head = DPTHeadBATLin(ic, ft, False, oc, False, num_frames=4, warp_signal=sig, scales=scales)
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
    assert torch.allclose(d_head, d_base, atol=1e-5), f"not equivalent, max_diff={md}"
    print(f"[batlin/{sig} scales={scales}] zero-init == baseline OK (max_diff={md:.2e})")


def test_learnable(scales=('p2', 'p1')):
    """At zero-init the proj/encoder grad is masked by delta's zero weights (like errormap_single).
    Perturb the delta conv away from zero and confirm scale_proj + error_encoder then carry grad."""
    ic, ft, oc = 64, 64, [48, 96, 192, 384]
    head = DPTHeadBATLin(ic, ft, False, oc, False, num_frames=4, warp_signal='rgbfeat', scales=scales)
    with torch.no_grad():
        head.ba_update.delta.weight.normal_(0, 0.05)
        head.ba_update.delta.bias.fill_(0.01)
    head.train()
    of, im, ext, K, B, T, ph, pw = _fake(ic)
    d = head(of, ph, pw, T, images=im, extrinsics=ext, intrinsics=K)
    d.float().mean().backward()
    ge = head.error_encoder[-1].weight.grad
    gp = head.scale_proj[scales[-1]].weight.grad
    assert ge is not None and ge.abs().sum() > 0, "error_encoder no grad after activating delta"
    assert gp is not None and gp.abs().sum() > 0, "scale_proj no grad after activating delta"
    print("[batlin] scale_proj + error_encoder learnable (after activating delta) OK")


if __name__ == '__main__':
    torch.manual_seed(0)
    for sig in ('rgb', 'feat', 'rgbfeat', 'hog'):
        test_forward(sig, ('p2', 'p1'))
    test_forward('feat', ('p4', 'p3', 'p2', 'p1'))          # scalable: 4 layers
    test_learnable(('p2', 'p1'))
    for sig in ('rgb', 'feat', 'rgbfeat', 'hog'):
        test_zeroinit_equiv(sig, ('p2', 'p1'))
    test_zeroinit_equiv('feat', ('p4', 'p3', 'p2', 'p1'))
    print("ALL OK")
