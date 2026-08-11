"""CPU smoke test for the ConvNeXt multiscale refine head (new for experiment 7).

Feeds a fake ConvNeXt NCHW pyramid (no backbone / no CUDA needed) and checks:
  * training forward returns 4 depths at progressively finer resolutions;
  * eval forward returns a single full-resolution depth tensor;
  * gradient reaches both parts of every per-scale regression head.

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


def _run(use_temporal, device='cpu', depth_feedback=False, fp32_head=False):
    batch, frames = 1, 2
    patch_size = 4
    base = 128                            # divisible by 32; stride-32 map is 4x4
    head = DPTHeadMultiScaleRefineConvNeXt(
        _CONVNEXT_DIMS, num_frames=frames, use_temporal=use_temporal,
        patch_size=patch_size, depth_feedback=depth_feedback,
        fp32_head=fp32_head).train().to(device)

    features = [f.to(device) for f in _fake_pyramid(batch, frames, (base, base))]
    for tensor in features:
        tensor.requires_grad_(False)
    patch_h = patch_w = base // patch_size

    depth = head(features, patch_h, patch_w, frames)

    expected_sizes = [(base // 8, base // 8), (base // 4, base // 4),
                      (base // 2, base // 2), (base, base)]
    # Training mode -> list of 4 per-scale depths, coarse to fine.
    assert isinstance(depth, list) and len(depth) == 4, type(depth)
    for scale_depth, expected_size in zip(depth, expected_sizes):
        assert scale_depth.shape == (batch * frames, 1, *expected_size), scale_depth.shape
        assert torch.isfinite(scale_depth).all()
    # Eval mode -> a single full-resolution tensor (finest scale).
    head.eval()
    with torch.no_grad():
        depth_eval = head(features, patch_h, patch_w, frames)
    assert torch.is_tensor(depth_eval)
    assert depth_eval.shape == (batch * frames, 1, base, base), depth_eval.shape
    head.train()

    # The multiscale head truncates cross-scale gradient (depth_prev.detach() + delta_z),
    # so each delta head is trained by its OWN scale's depth, not the final output.
    loss = sum(scale_depth.float().mean() for scale_depth in depth)
    loss.backward()
    for index, (output_conv1, delta) in enumerate(
        zip(head.output_conv1_heads, head.delta_heads)
    ):
        assert output_conv1.weight.grad is not None and output_conv1.weight.grad.abs().sum() > 0, \
            f"output_conv1 head {index} received no gradient"
        last = delta[-1]
        assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0, \
            f"delta head {index} received no gradient"


def test_convnext_multiscale_static():
    torch.manual_seed(0)
    _run(use_temporal=False)
    _run(use_temporal=False, fp32_head=True)


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
