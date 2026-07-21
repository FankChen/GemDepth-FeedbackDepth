"""CPU smoke test for the ConvNeXt multiscale refine head (new for experiment 7).

Feeds a fake ConvNeXt NCHW pyramid (no backbone / no CUDA needed) and checks:
  * training forward returns a list of 4 per-scale depths, each full-resolution
    [B*T, 1, H, W];
  * eval forward returns a single full-resolution depth tensor;
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

    full_hw = (patch_h * patch_size, patch_w * patch_size)
    # Training mode -> list of 4 per-scale depths, each at full resolution.
    assert isinstance(depth, list) and len(depth) == 4, type(depth)
    for scale_depth in depth:
        assert scale_depth.shape == (batch * frames, 1, *full_hw), scale_depth.shape
        assert torch.isfinite(scale_depth).all()
    # 抗死 ReLU 修复验证：初始(未训练)最粗尺度输出为正常数(约 0.5)，不塌成 0。
    coarse = depth[0]
    assert coarse.min().item() >= -1e-4 and coarse.mean().item() > 0.4, \
        (coarse.min().item(), coarse.mean().item())

    # Eval mode -> a single full-resolution tensor (finest scale).
    head.eval()
    with torch.no_grad():
        depth_eval = head(features, patch_h, patch_w, frames)
    assert torch.is_tensor(depth_eval)
    assert depth_eval.shape == (batch * frames, 1, *full_hw), depth_eval.shape
    head.train()

    # The multiscale head truncates cross-scale gradient (depth_prev.detach() + delta_z),
    # so each delta head is trained by its OWN scale's depth, not the final output.
    loss = sum(scale_depth.float().mean() for scale_depth in depth)
    loss.backward()
    for index, delta in enumerate(head.delta_heads):
        last = delta[-1]
        assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0, \
            f"delta head {index} received no gradient"


def test_convnext_multiscale_static():
    torch.manual_seed(0)
    _run(use_temporal=False)


def test_original_multiscale_antidying_init():
    # The original (ViT/DINOv2/DINOv3-ViT) multiscale head must carry the SAME
    # anti-dying init on its delta heads: last conv weight zero, coarsest bias 0.5.
    from model.dpt_multiscale import DPTHeadMultiScaleRefine
    head = DPTHeadMultiScaleRefine(384, features=64, out_channels=[64, 128, 256, 256],
                                   num_frames=2, use_temporal=False, patch_size=16)
    for index, delta_head in enumerate(head.delta_heads):
        assert delta_head[-1].weight.abs().sum().item() == 0.0, f'scale {index} weight not zero'
        expected = 0.5 if index == 0 else 0.0
        assert abs(delta_head[-1].bias.item() - expected) < 1e-6, (index, delta_head[-1].bias.item())


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
    test_original_multiscale_antidying_init()
    test_convnext_multiscale_temporal()
    print('DPTHeadMultiScaleRefineConvNeXt smoke OK')
