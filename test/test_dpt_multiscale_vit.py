"""CPU smoke test for the restored ViT multiscale refine head.

The head predates the decoder refactor and was deleted by it, so the thing worth
checking is not the head's own arithmetic -- that code is unchanged and was
trained with -- but that it still fits the interface around it: the registry
resolves it, build_decoder can fill its constructor, and its forward accepts
exactly what GemDepth.forward passes.

Feeds fake ViT token features (no backbone, no CUDA) and checks it stays
interchangeable with the DPTHeadTemporal it subclasses, which is what makes the
stage1_lite pair a single-variable comparison.

Run:  python -m pytest -q test/test_dpt_multiscale_vit.py
"""
import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.decoder_registry import build_decoder, get_decoder_class
from model.dpt_multiscale import DPTHeadMultiScaleRefine
from model.dpt_temporal import DPTHeadTemporal

_EMBED_DIM = 1024                 # ViT-L
_OUT_CHANNELS = [256, 512, 1024, 1024]
_PATCH_H = _PATCH_W = 8           # a small patch grid; 518/14 = 37 in practice


def _fake_tokens(batch, frames):
    """What DPTHeadTemporal reads: four (tokens, cls) pairs, tokens (B*T, N, C)."""
    tokens = torch.randn(batch * frames, _PATCH_H * _PATCH_W, _EMBED_DIM)
    return [(tokens.clone(), None) for _ in range(4)]


def _build(cls, frames):
    return cls(_EMBED_DIM, out_channels=_OUT_CHANNELS,
               num_frames=frames, use_temporal=True, patch_size=14)


def test_registry_resolves_the_restored_head():
    assert get_decoder_class("DPTHeadMultiScaleRefine") is DPTHeadMultiScaleRefine


def test_build_decoder_can_fill_the_constructor():
    # GemDepth passes in_channels and in_channels_list side by side; this head
    # declares only the former, and the injection has to drop the rest rather
    # than raise.
    head = build_decoder(
        get_decoder_class("DPTHeadMultiScaleRefine"),
        in_channels=_EMBED_DIM,
        in_channels_list=[96, 192, 384, 768],
        features=256, use_bn=False, out_channels=_OUT_CHANNELS,
        use_clstoken=False, num_frames=2, pe='ape',
        use_temporal=True, patch_size=14,
        fullres_mode='all', depth_feedback=True, fp32_head=True,
    )
    assert isinstance(head, DPTHeadMultiScaleRefine)


def test_forward_accepts_what_gemdepth_passes():
    # gemdepth.py calls head(head_features, patch_h, patch_w, T) for heads that
    # do not declare the geometry inputs, and this head must not declare them.
    parameters = inspect.signature(DPTHeadMultiScaleRefine.forward).parameters
    assert not any(k in parameters for k in ("images", "extrinsics", "intrinsics"))
    for name in ("out_features", "patch_h", "patch_w", "frame_length"):
        assert name in parameters


def test_training_returns_one_depth_per_scale_at_native_resolution():
    frames = 2
    head = _build(DPTHeadMultiScaleRefine, frames).train()
    depths = head(_fake_tokens(1, frames), _PATCH_H, _PATCH_W, frames)

    assert isinstance(depths, list) and len(depths) == 4
    sizes = [d.shape[-1] for d in depths]
    # Coarse -> fine, and native: the loss downsamples GT to each scale, which
    # only works if the head has not already upsampled them to a common size.
    assert sizes == sorted(sizes) and len(set(sizes)) == 4


def test_eval_matches_the_temporal_head_it_replaces():
    frames = 2
    features = _fake_tokens(1, frames)
    multiscale = _build(DPTHeadMultiScaleRefine, frames).eval()
    temporal = _build(DPTHeadTemporal, frames).eval()

    with torch.no_grad():
        ms_depth = multiscale(features, _PATCH_H, _PATCH_W, frames)
        tp_depth = temporal(features, _PATCH_H, _PATCH_W, frames)

    assert torch.is_tensor(ms_depth)
    assert ms_depth.shape == tp_depth.shape


def test_every_scale_head_receives_gradient():
    frames = 2
    head = _build(DPTHeadMultiScaleRefine, frames).train()
    depths = head(_fake_tokens(1, frames), _PATCH_H, _PATCH_W, frames)
    sum(d.mean() for d in depths).backward()

    for i, delta_head in enumerate(head.delta_heads):
        grad = delta_head[0].weight.grad
        assert grad is not None and torch.isfinite(grad).all(), f"scale {i}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
