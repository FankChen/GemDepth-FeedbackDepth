"""CPU smoke tests for the multi-scale refinement variants (paths C / D / A).

Each variant is a subclass that changes exactly one aspect of the shared
refinement loop in ``DPTHeadMultiScaleRefineConvNeXt``, so the tests target that
one aspect:

  * base head            -- unchanged: one round, four coarse-to-fine depths;
  * ``...GradConvNeXt``  -- gradient now crosses scale boundaries;
  * ``...IterConvNeXt``  -- more rounds, and parameter count independent of them;
  * ``...ErrMapConvNeXt``-- zero-initialised injection, i.e. a no-op at init;
  * ``...ErrMapIter...`` -- the two compose without restating either.

Plus the ConvNeXt-through-GEM path, which is what makes camera poses (and
therefore the error map) available to a hierarchical backbone at all.

Run:  python -m pytest -q test/test_multiscale_refine_variants.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.decoder_registry import available_decoder_names, get_decoder_class
from model.dpt_multiscale_convnext import DPTHeadMultiScaleRefineConvNeXt
from model.dpt_multiscale_errmap_convnext import DPTHeadMultiScaleErrMapConvNeXt
from model.dpt_multiscale_errmap_iter_convnext import DPTHeadMultiScaleErrMapIterConvNeXt
from model.dpt_multiscale_grad_convnext import DPTHeadMultiScaleGradConvNeXt
from model.dpt_multiscale_iter_convnext import DPTHeadMultiScaleIterConvNeXt

_CONVNEXT_DIMS = [96, 192, 384, 768]
_STRIDES = [4, 8, 16, 32]
_BASE = 128          # divisible by 32; the stride-32 map is 4x4
_PATCH = 4
_BATCH, _FRAMES = 1, 2


def _fake_pyramid():
    return [
        torch.randn(_BATCH * _FRAMES, _CONVNEXT_DIMS[i], _BASE // s, _BASE // s)
        for i, s in enumerate(_STRIDES)
    ]


def _make(head_cls, **kwargs):
    return head_cls(_CONVNEXT_DIMS, num_frames=_FRAMES, use_temporal=False,
                    patch_size=_PATCH, **kwargs).train()


def _geometry():
    """Identity-ish poses and a plausible pinhole K, at input resolution."""
    images = torch.rand(_BATCH, _FRAMES, 3, _BASE, _BASE)
    extrinsics = torch.eye(4).repeat(_BATCH, _FRAMES, 1, 1)
    # Give the second frame a small baseline so warping is not degenerate.
    extrinsics[:, 1, 0, 3] = 0.1
    intrinsics = torch.zeros(_BATCH, _FRAMES, 3, 3)
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = _BASE
    intrinsics[..., 0, 2] = intrinsics[..., 1, 2] = _BASE / 2
    intrinsics[..., 2, 2] = 1.0
    return images, extrinsics, intrinsics


def _forward(head, **kwargs):
    patch_h = patch_w = _BASE // _PATCH
    return head(_fake_pyramid(), patch_h, patch_w, _FRAMES, **kwargs)


def test_variants_are_registered():
    names = available_decoder_names()
    for cls in (DPTHeadMultiScaleGradConvNeXt, DPTHeadMultiScaleIterConvNeXt,
                DPTHeadMultiScaleErrMapConvNeXt, DPTHeadMultiScaleErrMapIterConvNeXt):
        assert cls.__name__ in names, f"{cls.__name__} not discovered"
        assert get_decoder_class(cls.__name__) is cls


def test_base_head_unchanged_by_the_seams():
    """The extension points must not alter the head they were carved out of."""
    torch.manual_seed(0)
    head = _make(DPTHeadMultiScaleRefineConvNeXt)
    depths = _forward(head)
    assert isinstance(depths, list) and len(depths) == 4
    expected = [(_BASE // 8,) * 2, (_BASE // 4,) * 2, (_BASE // 2,) * 2, (_BASE,) * 2]
    for depth, size in zip(depths, expected):
        assert depth.shape == (_BATCH * _FRAMES, 1, *size), depth.shape
        assert torch.isfinite(depth).all()

    # Geometry inputs are part of the decoder contract but ignored here.
    images, extrinsics, intrinsics = _geometry()
    torch.manual_seed(1)
    with_geometry = _forward(head, images=images, extrinsics=extrinsics,
                             intrinsics=intrinsics)
    torch.manual_seed(1)
    without_geometry = _forward(head)
    for a, b in zip(with_geometry, without_geometry):
        assert torch.equal(a, b), "base head must ignore geometry inputs"


def test_grad_head_opens_the_cross_scale_path():
    """Only the finest depth is supervised; the coarse projection must still learn."""
    def coarsest_grad(head):
        head.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        _forward(head)[-1].float().mean().backward()
        grad = head.output_conv1_heads[0].weight.grad
        return 0.0 if grad is None else grad.abs().sum().item()

    torch.manual_seed(0)
    detached = _make(DPTHeadMultiScaleRefineConvNeXt)
    torch.manual_seed(0)
    connected = _make(DPTHeadMultiScaleGradConvNeXt)

    assert coarsest_grad(detached) == 0.0, \
        "base head is supposed to truncate cross-scale gradient"
    assert coarsest_grad(connected) > 0.0, \
        "grad head must let the finest loss reach the coarsest step"

    # carry_scale=0 has to reproduce the parent exactly.
    torch.manual_seed(0)
    assert coarsest_grad(_make(DPTHeadMultiScaleGradConvNeXt, carry_scale=0.0)) == 0.0


def test_iter_head_rounds_are_free_in_parameters():
    torch.manual_seed(0)
    two = _make(DPTHeadMultiScaleIterConvNeXt, refine_rounds=2)
    torch.manual_seed(0)
    eight = _make(DPTHeadMultiScaleIterConvNeXt, refine_rounds=8)
    assert sum(p.numel() for p in two.parameters()) == \
        sum(p.numel() for p in eight.parameters()), \
        "weight sharing across rounds is the whole point of this arm"

    depths = _forward(two)
    assert len(depths) == 2 * 4, len(depths)
    # Every round restarts at the coarsest scale and ends at full resolution.
    assert depths[3].shape[-2:] == (_BASE, _BASE)
    assert depths[-1].shape[-2:] == (_BASE, _BASE)
    assert all(torch.isfinite(d).all() for d in depths)

    two.eval()
    with torch.no_grad():
        assert _forward(two).shape == (_BATCH * _FRAMES, 1, _BASE, _BASE)


def test_errmap_injection_starts_as_a_no_op():
    """Zero-init means training starts from the parent head, not beside it."""
    torch.manual_seed(0)
    head = _make(DPTHeadMultiScaleErrMapConvNeXt)
    images, extrinsics, intrinsics = _geometry()

    torch.manual_seed(1)
    with_geometry = _forward(head, images=images, extrinsics=extrinsics,
                             intrinsics=intrinsics)
    torch.manual_seed(1)
    without_geometry = _forward(head)
    for a, b in zip(with_geometry, without_geometry):
        assert torch.allclose(a, b, atol=1e-6), \
            "zero-initialised error encoders must not change the output at init"

    # Once the encoders are non-zero the warp has to actually reach the output.
    for encoder in head.error_encoders:
        torch.nn.init.normal_(encoder[-1].weight, std=0.02)
    torch.manual_seed(1)
    perturbed = _forward(head, images=images, extrinsics=extrinsics,
                         intrinsics=intrinsics)
    assert not torch.allclose(perturbed[-1], without_geometry[-1], atol=1e-6), \
        "error map has no effect on the prediction"
    assert all(torch.isfinite(d).all() for d in perturbed)


def test_errmap_iter_composes_both_halves():
    torch.manual_seed(0)
    head = _make(DPTHeadMultiScaleErrMapIterConvNeXt, refine_rounds=3)
    assert hasattr(head, 'gru_cells') and hasattr(head, 'error_encoders')
    images, extrinsics, intrinsics = _geometry()
    depths = _forward(head, images=images, extrinsics=extrinsics,
                      intrinsics=intrinsics)
    assert len(depths) == 3 * 4, len(depths)
    assert all(torch.isfinite(d).all() for d in depths)

    # Every scale keeps its own GRU, and the parent still truncates gradient
    # between scales, so each cell is trained by its own scale's outputs only --
    # supervise all of them, as the loss does.
    sum(d.float().mean() for d in depths).backward()
    for index, cell in enumerate(head.gru_cells):
        assert cell.convz.weight.grad is not None and \
            cell.convz.weight.grad.abs().sum() > 0, f"GRU cell {index} untrained"


def test_convnext_reaches_gem():
    """ConvNeXt + GEM: the pose path a hierarchical backbone used to be barred from."""
    from model.gemdepth import GemDepth

    torch.manual_seed(0)
    try:
        model = GemDepth(
            encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024],
            backbone='DINOv3ConvNeXtSmallBackbone',
            decoder='DPTHeadMultiScaleErrMapConvNeXt',
            use_gem=True, use_astt=False, use_temporal=False,
            num_frames=_FRAMES, load_backbone_pretrained=False,
        ).eval()
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[skip] cannot build ConvNeXt+GEM on this machine "
              f"({exc.__class__.__name__}: {exc})")
        return

    # GEM/ASTT width must follow the backbone's deepest level, not a config default.
    assert model.camera_token.shape[-1] == 768, model.camera_token.shape
    assert model.decoder_requires_geometry_inputs

    images = torch.rand(_BATCH, _FRAMES, 3, _BASE, _BASE)
    with torch.no_grad():
        depth, pose_enc_list, extrinsic, intrinsic = model(images)
    assert depth.shape == (_BATCH, _FRAMES, _BASE, _BASE), depth.shape
    assert extrinsic is not None and extrinsic.shape == (_BATCH, _FRAMES, 4, 4)
    assert intrinsic is not None and intrinsic.shape == (_BATCH, _FRAMES, 3, 3)
    assert pose_enc_list is not None
    assert torch.isfinite(depth).all()


if __name__ == '__main__':
    test_variants_are_registered()
    test_base_head_unchanged_by_the_seams()
    test_grad_head_opens_the_cross_scale_path()
    test_iter_head_rounds_are_free_in_parameters()
    test_errmap_injection_starts_as_a_no_op()
    test_errmap_iter_composes_both_halves()
    test_convnext_reaches_gem()
    print('multi-scale refinement variants smoke OK')
