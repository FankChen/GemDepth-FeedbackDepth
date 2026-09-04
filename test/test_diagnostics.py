"""Synthetic checks for the zero-training diagnostic tools."""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diagnose_alignment import compare_alignment
from diagnose_gem_camera import (
    _warp_stats,
    normalize_ground_truth_camera,
    rotation_error_degrees,
)
from diagnose_cost_volume_oracle import volume_quality
from model.util.warp import plane_sweep_warp


def test_alignment_diagnostic_separates_frame_and_sequence_gauges():
    height, width = 12, 16
    target_disparity = np.linspace(
        0.1, 1.0, height * width).reshape(height, width)
    depth = np.stack([1.0 / target_disparity] * 3)
    prediction = np.stack([
        2.0 * target_disparity + 0.1,
        0.5 * target_disparity + 0.4,
        1.3 * target_disparity - 0.2,
    ])
    valid = np.ones_like(prediction, dtype=bool)

    result = compare_alignment(prediction, depth, valid, max_depth=80.0)
    assert result["frame"]["absrel"] < 1e-6
    assert result["sequence"]["absrel"] > 0.01
    assert result["frame_scale_cv"] > 0.1


def test_camera_gauge_normalization_preserves_projection():
    batch, frames, height, width = 1, 3, 18, 24
    torch.manual_seed(0)
    images = torch.rand(batch, frames, 3, height, width)
    depth = torch.full((batch, frames, 1, height, width), 8.0)
    mask = torch.ones_like(depth)
    intrinsic = torch.tensor([
        [30.0, 0.0, width / 2],
        [0.0, 30.0, height / 2],
        [0.0, 0.0, 1.0],
    ]).unsqueeze(0)
    extrinsic = torch.eye(4).repeat(batch, frames, 1, 1)
    extrinsic[:, 1, 0, 3] = 0.3
    extrinsic[:, 2, 0, 3] = 0.6
    # Move on both axes: pure x motion leaves the top/bottom rows exactly at
    # grid_sample's +/-1 boundary, where float32 roundoff can flip validity even
    # though the projected coordinates are geometrically equivalent.
    extrinsic[:, 1, 1, 3] = 0.1
    extrinsic[:, 2, 1, 3] = 0.2

    relative, depth_normalized, scale = normalize_ground_truth_camera(
        depth, mask, intrinsic, extrinsic)
    intrinsic_frames = intrinsic[:, None].expand(-1, frames, -1, -1)
    metric_stats = _warp_stats(
        images, depth, intrinsic_frames, extrinsic)
    normalized_stats = _warp_stats(
        images, depth_normalized, intrinsic_frames, relative)

    assert scale.item() > 0
    assert torch.allclose(relative[:, 0], torch.eye(4).unsqueeze(0), atol=1e-6)
    assert abs(metric_stats[0] - normalized_stats[0]) < 1e-6
    assert abs(metric_stats[1] - normalized_stats[1]) < 1e-6


def test_rotation_error_degrees():
    target = torch.eye(3).view(1, 1, 3, 3)
    prediction = target.clone()
    angle = torch.tensor(np.pi / 2)
    prediction[0, 0] = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0.0],
        [torch.sin(angle), torch.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert torch.allclose(
        rotation_error_degrees(prediction, target),
        torch.tensor([[90.0]]), atol=1e-4)


def test_cost_volume_quality_ranks_known_true_bins():
    samples = torch.tensor([2.0, 4.0, 8.0, 16.0]).view(1, 4, 1, 1).expand(
        1, 4, 2, 2)
    target_depth = torch.tensor([[[[2.0, 4.0], [8.0, 16.0]]]])
    target_mask = torch.ones_like(target_depth)
    true_index = torch.tensor([[[0, 1], [2, 3]]])
    scores = torch.zeros(1, 4, 2, 2)
    scores.scatter_(1, true_index[:, None], 2.0)
    valid = torch.ones(1, 1, 4, 2, 2)

    result = volume_quality(
        scores, valid, target_depth, target_mask, samples)

    assert result["eligible_pixels"] == 4
    assert result["top1"] == 1.0
    assert result["top3"] == 1.0
    assert result["mean_rank"] == 1.0
    assert result["mean_bin_error"] == 0.0
    assert result["true_bin_margin"] == 2.0


def test_plane_sweep_is_invariant_to_joint_depth_translation_scale():
    height, width = 9, 13
    torch.manual_seed(1)
    source = torch.rand(1, 3, height, width)
    intrinsic = torch.tensor([[15.0, 0.0, width / 2],
                              [0.0, 15.0, height / 2],
                              [0.0, 0.0, 1.0]]).unsqueeze(0)
    reference_pose = torch.eye(4).unsqueeze(0)
    source_pose = reference_pose.clone()
    source_pose[:, 0, 3] = 0.3
    source_pose[:, 1, 3] = 0.1
    samples = torch.tensor([4.0, 8.0, 12.0]).view(1, 3, 1, 1).expand(
        1, 3, height, width)

    metric_warp, metric_valid = plane_sweep_warp(
        source, samples, intrinsic, intrinsic, reference_pose, source_pose)
    scale = 3.0
    normalized_pose = source_pose.clone()
    normalized_pose[..., :3, 3] /= scale
    normalized_warp, normalized_valid = plane_sweep_warp(
        source, samples / scale, intrinsic, intrinsic,
        reference_pose, normalized_pose)

    assert torch.equal(metric_valid, normalized_valid)
    assert torch.allclose(metric_warp, normalized_warp, atol=1e-5)
