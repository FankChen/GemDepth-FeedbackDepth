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
    assert len(head.layer_depths) == 4, f"expect 4 layer depths, got {len(head.layer_depths)}"
    assert len(head.layer_depths_pre) == 4
    loss = d.float().mean() + sum(z.float().mean() for z in head.layer_depths)
    loss.backward()
    gr = head.layer_depth_heads['p2'][0].weight.grad
    gd = head.layer_delta_heads['p2'][-1].weight.grad         # last conv zero-init but drives dz
    assert gr is not None and gr.abs().sum() > 0, "refine head got no grad"
    assert gd is not None and gd.abs().sum() > 0, "feature delta head got no grad"
    print(f"[perlayer_refine] fwd/bwd OK shape={tuple(d.shape)} layers={len(head.layer_depths)} (refine+feat-delta in-graph)")


def test_main_equals_baseline():
    ic, ft, oc = 64, 64, [48, 96, 192, 384]
    head = DPTHeadPerLayerRefine(ic, ft, False, oc, False, num_frames=4)
    base = DPTHeadTemporal(ic, ft, False, oc, False, num_frames=4)
    missing, unexpected = base.load_state_dict(head.state_dict(), strict=False)
    assert len(missing) == 0, f"baseline missing shared keys: {missing[:5]}"
    head.eval()
    base.eval()
    of, B, T, ph, pw = _fake(ic)
    with torch.no_grad():
        d_head = head(of, ph, pw, T)
        d_base = base(of, ph, pw, T)
    md = (d_head - d_base).abs().max().item()
    assert torch.allclose(d_head, d_base, atol=1e-5), f"main output != baseline, max_diff={md}"
    print(f"[perlayer_refine] main output == baseline OK (max_diff={md:.2e})")


if __name__ == '__main__':
    test_forward()
    test_main_equals_baseline()
    print("ALL OK")
