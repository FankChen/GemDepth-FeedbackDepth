"""CPU smoke test for DPTHeadPerLayerRefine (per-layer refine + FEATURE delta, no warp)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dpt_temporal import DPTHeadTemporal
from model.dpt_perlayer_refine import DPTHeadPerLayerRefine


def _fake(in_ch, B=1, T=4, ph=8, pw=8):
    BT, L = B * T, ph * pw
    of = [(torch.randn(BT, L, in_ch), torch.randn(BT, in_ch)) for _ in range(4)]
    return of, B, T, ph, pw


def test_forward():
    ic, ft, oc = 64, 64, [48, 96, 192, 384]
    head = DPTHeadPerLayerRefine(ic, ft, False, oc, False, num_frames=4)
    head.train()
    of, B, T, ph, pw = _fake(ic)
    d = head(of, ph, pw, T)                                   # no images/ext/K needed (no warp)
    assert d.shape[0] == B * T and d.shape[1] == 1, d.shape
    assert d.shape[-2:] == (ph * 14, pw * 14), f"final should be full-res, got {tuple(d.shape)}"
    assert len(head.layer_depths) == 4, f"expect 4 cascaded layer depths, got {len(head.layer_depths)}"
    hs = [z.shape[-1] for z in head.layer_depths]
    assert hs == sorted(hs) and len(set(hs)) == 4, f"cascade resolutions not increasing: {hs}"
    # 4 per-layer losses + final; every refine head must get gradient
    loss = d.float().mean() + sum(z.float().mean() for z in head.layer_depths)
    loss.backward()
    g4 = head.refine_p4[0].weight.grad
    gf = head.refine_p1_fuse[-1].weight.grad                  # p1 fuse (zero-init, uses output_conv base)
    assert g4 is not None and g4.abs().sum() > 0, "refine_p4 got no grad"
    assert gf is not None and gf.abs().sum() > 0, "refine_p1_fuse got no grad"
    print(f"[perlayer_refine cascade] fwd/bwd OK final={tuple(d.shape)} layers={len(head.layer_depths)} res={hs}")


if __name__ == '__main__':
    test_forward()
    print("ALL OK")
