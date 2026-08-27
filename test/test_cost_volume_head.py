"""CPU smoke tests for the IGEV-MVS port (cost-volume decoder).

Covers the three things that are easy to get silently wrong in a plane-sweep head:
  * the sweep geometry -- a hypothesis that matches the true depth should win;
  * non-finite intrinsics -- GEM's focal length can still be inf early in training,
    and a NaN there used to propagate all the way to the output;
  * the output contract -- a list of refinements while training, one tensor at eval.

Run:  python -m pytest -q test/test_cost_volume_head.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.decoder_registry import available_decoder_names, get_decoder_class
from model.dpt_cost_volume_convnext import DPTHeadCostVolumeConvNeXt
from model.util.warp import plane_sweep_warp

_DIMS = [96, 192, 384, 768]
_BASE, _PATCH = 128, 4
_BATCH, _FRAMES = 1, 3


def _pyramid():
    return [torch.randn(_BATCH * _FRAMES, _DIMS[i],
                        _BASE // (_PATCH * 2 ** i), _BASE // (_PATCH * 2 ** i))
            for i in range(4)]


def _geometry(baseline=0.3):
    images = torch.rand(_BATCH, _FRAMES, 3, _BASE, _BASE)
    extrinsics = torch.eye(4).repeat(_BATCH, _FRAMES, 1, 1)
    for t in range(_FRAMES):
        extrinsics[:, t, 0, 3] = baseline * t
    intrinsics = torch.zeros(_BATCH, _FRAMES, 3, 3)
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = _BASE
    intrinsics[..., 0, 2] = intrinsics[..., 1, 2] = _BASE / 2
    intrinsics[..., 2, 2] = 1.0
    return images, extrinsics, intrinsics


def _make(**kwargs):
    kwargs.setdefault('volume_level', 1)
    kwargs.setdefault('num_sample', 16)
    kwargs.setdefault('iters', 3)
    return DPTHeadCostVolumeConvNeXt(
        _DIMS, num_frames=_FRAMES, patch_size=_PATCH, **kwargs).train()


def test_registered():
    assert 'DPTHeadCostVolumeConvNeXt' in available_decoder_names()
    assert get_decoder_class('DPTHeadCostVolumeConvNeXt') is DPTHeadCostVolumeConvNeXt


def test_plane_sweep_finds_the_true_depth():
    """The sweep must be geometrically right, not just runnable.

    Two cameras separated by a known baseline observe a fronto-parallel plane. Warping
    the source view under the correct depth hypothesis has to reproduce the reference
    view better than any wrong hypothesis does.
    """
    torch.manual_seed(0)
    height = width = 32
    true_depth = 5.0
    focal, centre = 40.0, height / 2.0

    K = torch.tensor([[focal, 0.0, centre], [0.0, focal, centre], [0.0, 0.0, 1.0]])
    K = K.unsqueeze(0)
    ext_ref = torch.eye(4).unsqueeze(0)
    ext_src = torch.eye(4).unsqueeze(0)
    ext_src[0, 0, 3] = 0.5                       # pure sideways translation

    texture = torch.randn(1, 8, height, width)
    # Reference view of the plane = source view shifted by the induced disparity.
    shift = int(round(focal * 0.5 / true_depth))
    reference = torch.roll(texture, shifts=-shift, dims=3)

    hypotheses = torch.tensor([2.0, true_depth, 20.0]).view(1, 3, 1, 1)
    hypotheses = hypotheses.expand(1, 3, height, width)
    warped, valid = plane_sweep_warp(texture, hypotheses, K, K, ext_ref, ext_src)

    # Score each hypothesis by agreement with the reference, inside the valid region.
    scores = []
    for d in range(3):
        mask = valid[:, :, d]
        agreement = (warped[:, :, d] * reference).sum(dim=1, keepdim=True)
        scores.append((agreement * mask).sum() / mask.sum().clamp_min(1.0))
    assert torch.argmax(torch.stack(scores)).item() == 1, \
        f"the true depth should score highest, got {[round(s.item(), 3) for s in scores]}"


def test_output_contract_and_gradients():
    torch.manual_seed(0)
    head = _make(iters=3)
    images, extrinsics, intrinsics = _geometry()
    outputs = head(_pyramid(), _BASE // _PATCH, _BASE // _PATCH, _FRAMES,
                   images=images, extrinsics=extrinsics, intrinsics=intrinsics)

    # One softargmin initialisation plus one prediction per iteration.
    assert isinstance(outputs, list) and len(outputs) == 1 + 3, len(outputs)
    for prediction in outputs:
        assert prediction.shape == (_BATCH * _FRAMES, 1, _BASE, _BASE), prediction.shape
        assert torch.isfinite(prediction).all()
        # Normalised inverse depth: the loss is affine-invariant, but leaving the range
        # would mean the hypothesis index escaped the volume.
        assert 0.0 <= prediction.min() and prediction.max() <= 1.0

    sum(o.float().mean() for o in outputs).backward()
    starved = [name for name, p in head.named_parameters() if p.grad is None]
    assert not starved, f"parameters never used: {starved[:5]}"
    unstable = [name for name, p in head.named_parameters()
                if not torch.isfinite(p.grad).all()]
    assert not unstable, f"non-finite gradients: {unstable[:5]}"

    head.eval()
    with torch.no_grad():
        single = head(_pyramid(), _BASE // _PATCH, _BASE // _PATCH, _FRAMES,
                      images=images, extrinsics=extrinsics, intrinsics=intrinsics)
    assert torch.is_tensor(single)
    assert single.shape == (_BATCH * _FRAMES, 1, _BASE, _BASE)


def test_survives_non_finite_intrinsics():
    """An unconverged GEM can emit inf focal lengths; that must not reach the output.

    Nor the gradients. GEM's intrinsics carry gradient, so without an explicit detach the
    sweep backpropagates into the camera head, where 0 * inf = NaN -- training checks the
    global gradient norm and aborts on a non-finite one, so this fails the run at step 0.
    The pose is evidence here, supervised by the camera loss, not something to optimise
    through the warp; asserting that no gradient reaches it pins that decision down.
    """
    torch.manual_seed(0)
    head = _make()
    images, extrinsics, intrinsics = _geometry()
    intrinsics[:, :, 0, 0] = float('inf')
    intrinsics[:, :, 1, 1] = float('inf')
    intrinsics.requires_grad_(True)
    extrinsics.requires_grad_(True)

    features = [f.requires_grad_(True) for f in _pyramid()]
    outputs = head(features, _BASE // _PATCH, _BASE // _PATCH, _FRAMES,
                   images=images, extrinsics=extrinsics, intrinsics=intrinsics)
    assert all(torch.isfinite(o).all() for o in outputs), \
        'non-finite intrinsics leaked into the prediction'

    sum(o.float().mean() for o in outputs).backward()
    assert intrinsics.grad is None and extrinsics.grad is None, \
        'the sweep must not backpropagate into the pose'
    bad = [name for name, p in head.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f'non-finite gradients in the head: {bad[:5]}'
    leaked = [i for i, f in enumerate(features)
              if f.grad is not None and not torch.isfinite(f.grad).all()]
    assert not leaked, f'non-finite gradient reached backbone levels {leaked}'


def test_zero_baseline_is_survivable():
    """A static camera carries no matching information -- it must still not blow up.

    This is a real regime in driving data (stopped at a light), and it is the known
    weakness of the method rather than a bug: with no baseline every hypothesis warps
    identically, so the volume is uninformative. The requirement here is only that
    training does not break when it happens.
    """
    torch.manual_seed(0)
    head = _make()
    images, extrinsics, intrinsics = _geometry(baseline=0.0)
    outputs = head(_pyramid(), _BASE // _PATCH, _BASE // _PATCH, _FRAMES,
                   images=images, extrinsics=extrinsics, intrinsics=intrinsics)
    assert all(torch.isfinite(o).all() for o in outputs)
    sum(o.float().mean() for o in outputs).backward()


def test_volume_at_stride_four():
    """volume_level=0 is the paper's default resolution; it must build too."""
    torch.manual_seed(0)
    head = _make(volume_level=0)
    assert head.volume_stride == _PATCH
    images, extrinsics, intrinsics = _geometry()
    outputs = head(_pyramid(), _BASE // _PATCH, _BASE // _PATCH, _FRAMES,
                   images=images, extrinsics=extrinsics, intrinsics=intrinsics)
    assert outputs[-1].shape == (_BATCH * _FRAMES, 1, _BASE, _BASE)
    assert torch.isfinite(outputs[-1]).all()


if __name__ == '__main__':
    test_registered()
    test_plane_sweep_finds_the_true_depth()
    test_output_contract_and_gradients()
    test_survives_non_finite_intrinsics()
    test_zero_baseline_is_survivable()
    test_volume_at_stride_four()
    print('cost-volume head smoke OK')
