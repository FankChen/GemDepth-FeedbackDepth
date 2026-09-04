"""Tests for config-driven training objectives."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loss.multiscale_video_l1_loss import MultiScaleVideoL1Loss
from loss.multiscale_videoloss import MultiScaleVideoDepthLoss
from loss.objective_registry import (
    available_objective_names,
    build_objective,
)
from loss.videoloss import VideoDepthLoss, compute_camera_loss


def _batch():
    torch.manual_seed(4)
    batch, frames, height, width = 2, 3, 16, 20
    prediction = torch.rand(batch, frames, 1, height, width) + 0.1
    target = torch.rand(batch, frames, 1, height, width) * 40.0 + 1.0
    mask = torch.ones_like(target)
    intrinsic = torch.eye(3).repeat(batch, 1, 1)
    extrinsic = [torch.eye(4).repeat(batch, 1, 1) for _ in range(frames)]
    return prediction, target, mask, intrinsic, extrinsic


def _assert_same_dict(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if torch.is_tensor(actual[key]):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key


def test_builtin_objectives_are_discovered():
    names = available_objective_names()
    assert {"l2", "video", "videoloss", "video_loss",
            "multiscale_video", "joint_align"}.issubset(names)


def test_unknown_objective_and_kwargs_fail_closed():
    with pytest.raises(ValueError, match="Unknown loss.objective"):
        build_objective("not_a_loss", pose_flag=False)
    with pytest.raises(ValueError, match="Unsupported loss.kwargs"):
        build_objective(
            "video", pose_flag=False,
            objective_kwargs={"alpha_typo": 0.0})


def test_legacy_l2_is_bit_exact_for_tensor_and_list():
    prediction, target, mask, intrinsic, extrinsic = _batch()
    objective = build_objective(
        "l2", pose_flag=False,
        scale_weights=[1.0, 2.0], normalize_scale_weights=False)

    actual_single = objective(
        prediction, target, mask, intrinsic, extrinsic, None, None)
    expected_single = VideoDepthLoss(pose_flag=False)(
        prediction.squeeze(2), target.squeeze(2), mask.squeeze(2),
        intrinsic, extrinsic, None, None)
    _assert_same_dict(actual_single, expected_single)

    coarse = F.interpolate(
        prediction.flatten(0, 1), size=(8, 10), mode="bilinear",
        align_corners=True).unflatten(0, prediction.shape[:2])
    predictions = [coarse, prediction]
    actual_multi = objective(
        predictions, target, mask, intrinsic, extrinsic, None, None)
    expected_multi = MultiScaleVideoL1Loss(
        scale_weights=[1.0, 2.0])(
            [item.squeeze(2) for item in predictions],
            target.squeeze(2), mask.squeeze(2),
            intrinsic, extrinsic, None, None)
    _assert_same_dict(actual_multi, expected_multi)


def test_video_objective_is_bit_exact_with_legacy_implementation():
    prediction, target, mask, intrinsic, extrinsic = _batch()
    objective = build_objective(
        "video", pose_flag=False,
        scale_weights=[1.0, 1.0], normalize_scale_weights=False,
        objective_kwargs={"alpha": 0.5, "stable_scale": 10.0})

    coarse = F.interpolate(
        prediction.flatten(0, 1), size=(8, 10), mode="bilinear",
        align_corners=True).unflatten(0, prediction.shape[:2])
    predictions = [coarse, prediction]
    actual = objective(
        predictions, target, mask, intrinsic, extrinsic, None, None)
    expected = MultiScaleVideoDepthLoss(
        pose_flag=False, scale_weights=[1.0, 1.0],
        normalize_scale_weights=False)(
            [item.squeeze(2) for item in predictions],
            target.squeeze(2), mask.squeeze(2),
            intrinsic, extrinsic, None, None)
    _assert_same_dict(actual, expected)


@pytest.mark.parametrize(
    "alpha,stable_scale,zero_key",
    [(0.0, 10.0, "gm"), (0.5, 0.0, "stable_loss"),
     (0.0, 0.0, "both")])
def test_carve_term_switches(alpha, stable_scale, zero_key):
    prediction, target, mask, intrinsic, extrinsic = _batch()
    objective = build_objective(
        "video", pose_flag=False,
        objective_kwargs={"alpha": alpha, "stable_scale": stable_scale})
    losses = objective(
        prediction, target, mask, intrinsic, extrinsic, None, None)
    if zero_key in ("gm", "both"):
        assert losses["gm"] == 0
    if zero_key in ("stable_loss", "both"):
        assert torch.equal(losses["stable_loss"], torch.zeros_like(losses["stable_loss"]))
    assert torch.isfinite(losses["total_loss"])


def test_joint_alignment_adds_a_clip_level_regression_term():
    height, width = 8, 10
    q = torch.linspace(0.1, 1.0, height * width).view(1, 1, 1, height, width)
    target = (1.0 / q).repeat(1, 2, 1, 1, 1)
    prediction = torch.cat([2.0 * q + 0.1, 0.5 * q + 0.4], dim=1)
    mask = torch.ones_like(target)
    intrinsic = torch.eye(3).unsqueeze(0)
    extrinsic = [torch.eye(4).unsqueeze(0) for _ in range(2)]

    base = build_objective(
        "video", pose_flag=False,
        objective_kwargs={"alpha": 0.0, "stable_scale": 0.0})
    joint = build_objective(
        "joint_align", pose_flag=False,
        objective_kwargs={
            "alpha": 0.0, "stable_scale": 0.0,
            "sequence_weight": 1.0,
        })
    base_losses = base(
        prediction, target, mask, intrinsic, extrinsic, None, None)
    joint_losses = joint(
        prediction, target, mask, intrinsic, extrinsic, None, None)

    # Each frame has a perfect but different affine transform.  Frame-wise SSI
    # can remove both, whereas one transform shared by the clip cannot.
    assert base_losses["ssi"] < 1e-5
    assert joint_losses["sequence_loss"] > 0.01
    assert torch.allclose(
        joint_losses["total_loss"],
        base_losses["total_loss"] + joint_losses["sequence_loss"],
        atol=1e-6)


def test_focal_supervision_is_opt_in_and_affects_camera_loss():
    batch, frames, height, width = 1, 2, 12, 16
    extrinsic = torch.eye(4).repeat(batch, frames, 1, 1)
    intrinsic = torch.tensor([
        [10.0, 0.0, width / 2],
        [0.0, 10.0, height / 2],
        [0.0, 0.0, 1.0],
    ]).unsqueeze(0)
    depth = torch.ones(batch, frames, height, width)
    mask = torch.ones_like(depth)
    prediction = torch.zeros(batch, frames, 9)
    prediction[..., 6] = 1.0  # identity quaternion, but wrong FoV

    without_focal = compute_camera_loss(weight_focal=0.0)(
        [prediction], intrinsic, extrinsic, depth, mask)
    with_focal = compute_camera_loss(weight_focal=0.5)(
        [prediction], intrinsic, extrinsic, depth, mask)

    assert without_focal["loss_FL"] > 0
    assert torch.equal(
        without_focal["loss_camera"], torch.zeros_like(without_focal["loss_camera"]))
    assert torch.allclose(
        with_focal["loss_camera"], 0.5 * with_focal["loss_FL"])
