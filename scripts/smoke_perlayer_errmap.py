"""CPU smoke test for DPTHeadPerLayerErrmap (cascaded refine + per-layer error-map correction)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dpt_perlayer_errmap import DPTHeadPerLayerErrmap


def _fake(in_ch, B=1, T=4, ph=8, pw=8):
    BT, L = B * T, ph * pw
    of = [(torch.randn(BT, L, in_ch), torch.randn(BT, in_ch)) for _ in range(4)]
    H0, W0 = ph * 14, pw * 14
    images = torch.rand(B, T, 3, H0, W0)
    ext = torch.eye(4).repeat(B, T, 1, 1)
    for t in range(T):
        ext[:, t, 0, 3] = 0.05 * t
    K = torch.eye(3).repeat(B, T, 1, 1)                          # predicted intrinsics are (B,T,3,3)
    K[..., 0, 0] = K[..., 1, 1] = 200.0
    K[..., 0, 2], K[..., 1, 2] = W0 / 2, H0 / 2
    return of, images, ext, K, B, T, ph, pw


def test_forward(sig):
    ic, ft, oc = 64, 64, [48, 96, 192, 384]
    head = DPTHeadPerLayerErrmap(ic, ft, False, oc, False, num_frames=4, warp_signal=sig)
    head.train()
    of, im, ext, K, B, T, ph, pw = _fake(ic)
    d = head(of, ph, pw, T, images=im, extrinsics=ext, intrinsics=K)
    assert d.shape[0] == B * T and d.shape[1] == 1, d.shape
    assert d.shape[-2:] == (ph * 14, pw * 14), f"final should be full-res, got {tuple(d.shape)}"
    assert len(head.layer_depths) == 4, f"expect 4 cascaded layer depths, got {len(head.layer_depths)}"
    hs = [z.shape[-1] for z in head.layer_depths]
    assert hs == sorted(hs) and len(set(hs)) == 4, f"cascade resolutions not increasing: {hs}"
    loss = d.float().mean() + sum(z.float().mean() for z in head.layer_depths)
    loss.backward()
    g4 = head.refine_p4[0].weight.grad
    ge = head.error_encoders['p2'][-1].weight.grad             # errmap last conv (zero-init, drives delta)
    assert g4 is not None and g4.abs().sum() > 0, "refine_p4 got no grad"
    assert ge is not None and ge.abs().sum() > 0, "errmap encoder got no grad"
    print(f"[perlayer_errmap/{sig}] fwd/bwd OK final={tuple(d.shape)} layers={len(head.layer_depths)} res={hs} (refine+errmap in-graph)")


if __name__ == '__main__':
    for sig in ['rgb', 'feat', 'rgbfeat', 'hog']:
        test_forward(sig)
    print("ALL OK")
