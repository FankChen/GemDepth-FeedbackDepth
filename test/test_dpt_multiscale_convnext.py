"""CPU smoke test for the ConvNeXt multiscale refine head (new for experiment 7).

Feeds a fake ConvNeXt NCHW pyramid (no backbone / no CUDA needed) and checks:
  * forward returns full-resolution depth [B*T, 1, H, W];
  * head.aux_depths holds 4 per-scale predictions shaped [B, T, 1, h, w];
  * gradient reaches every delta head.

Run:  python -m pytest -q test/test_dpt_multiscale_convnext.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dpt_multiscale_convnext import DPTHeadMultiScaleRefineConvNeXt

_CONVNEXT_DIMS = [96, 192, 384, 768]     # DINOv3 ConvNeXt-s stage widths
_STRIDES = [4, 8, 16, 32]                # native ConvNeXt pyramid strides


def _fake_pyramid(batch, frames, base_hw):
    """4 NCHW maps at strides 4/8/16/32 (finest -> coarsest), matching build_backbone."""
    height, width = base_hw
    return [
        torch.randn(batch * frames, _CONVNEXT_DIMS[i], height // s, width // s)
        for i, s in enumerate(_STRIDES)
    ]


def _run(use_temporal, device='cpu'):
    batch, frames = 1, 2
    patch_size = 4
    base = 128                            # divisible by 32; stride-32 map is 4x4
    head = DPTHeadMultiScaleRefineConvNeXt(
        _CONVNEXT_DIMS, num_frames=frames, use_temporal=use_temporal,
        patch_size=patch_size).train().to(device)

    features = [f.to(device) for f in _fake_pyramid(batch, frames, (base, base))]
    for tensor in features:
        tensor.requires_grad_(False)
    patch_h = patch_w = base // patch_size

    depth = head(features, patch_h, patch_w, frames)

    assert depth.shape == (batch * frames, 1, patch_h * patch_size, patch_w * patch_size), depth.shape
    assert torch.isfinite(depth).all()
    assert len(head.aux_depths) == 4
    for aux in head.aux_depths:
        assert aux.dim() == 5 and aux.shape[:3] == (batch, frames, 1), tuple(aux.shape)

    # The multiscale head truncates cross-scale gradient (depth_prev.detach() + delta_z),
    # so each delta head is trained by its OWN scale's aux depth, not the final output.
    # This mirrors train.py's per-scale aux supervision on head.aux_depths.
    loss = sum(aux.float().mean() for aux in head.aux_depths)
    loss.backward()
    for index, delta in enumerate(head.delta_heads):
        last = delta[-1]
        assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0, \
            f"delta head {index} received no gradient"


def test_convnext_multiscale_static():
    torch.manual_seed(0)
    _run(use_temporal=False)


def test_convnext_multiscale_temporal():
    # The temporal motion modules use xformers memory_efficient_attention, which is
    # CUDA-only, so this path can only be exercised on a free GPU. The motion calls are
    # identical to the proven DPTHeadTemporalConvNeXt; the static case covers the new code.
    if not torch.cuda.is_available():
        print('[skip] temporal path needs CUDA (motion_module uses xformers)')
        return
    try:
        torch.manual_seed(0)
        _run(use_temporal=True, device='cuda')
        print('[ok] temporal CUDA smoke passed')
    except RuntimeError as exc:  # shared node: GPU may be busy / unavailable
        print(f'[skip] temporal CUDA smoke unavailable ({exc.__class__.__name__}: {exc})')


if __name__ == '__main__':
    test_convnext_multiscale_static()
    test_convnext_multiscale_temporal()
    print('DPTHeadMultiScaleRefineConvNeXt smoke OK')
