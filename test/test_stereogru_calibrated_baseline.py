"""Calibrated baseline contracts; all fixtures are synthetic, not paper results."""

import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import dataset.stereogru_calibration as data
import stereogru_calibrated_baseline as pilot
from loss.objective_registry import build_objective
from model.decoder_registry import build_decoder, get_decoder_class
from model.dpt_calibrated_volume_only_convnext import (
    native_convnext_camera_input, validate_metric_cameras,
)
from model.util.warp import scale_intrinsics


def camera_fixture(frames=3, size=64):
    images = torch.rand(1, frames, 3, size, size)
    intrinsic = torch.tensor([[64., 0., 26.], [0., 64., 31.], [0., 0., 1.]]).repeat(1, frames, 1, 1)
    extrinsic = torch.eye(4).repeat(1, frames, 1, 1)
    extrinsic[0, :, 0, 3] = torch.arange(frames) * .3
    return images, intrinsic, extrinsic


def small_head():
    return build_decoder(get_decoder_class("DPTHeadCalibratedVolumeOnlyConvNeXt"),
                         {"num_sample": 8, "num_groups": 2, "match_dim": 8,
                          "hidden_dim": 16, "depth_min": 3., "depth_max": 80.},
                         in_channels_list=[8, 16, 32, 64], patch_size=4)


def test_registered_volume_only_is_calibrated_bounded_and_has_no_gru():
    torch.manual_seed(0)
    head = small_head().eval()
    images, K, E = camera_fixture()
    dims = [8, 16, 32, 64]
    features = [torch.randn(3, dim, 64 // (4 * 2 ** i), 64 // (4 * 2 ** i))
                for i, dim in enumerate(dims)]
    with pytest.raises(ValueError, match="geometry_gauge"):
        head(features, 16, 16, 3, images=images, intrinsics=K, extrinsics=E)
    out = head(features, 16, 16, 3, images=images, intrinsics=K,
               extrinsics=E, geometry_gauge="metric")
    assert out.shape == (3, 1, 64, 64)
    assert torch.isfinite(out).all() and (out >= 0).all() and (out <= 1).all()
    assert not any(name.startswith(("gru_", "encoder.", "index_head.")) for name in head.state_dict())
    assert head.last_diagnostics["raw_zero_fraction"] < .9
    out.square().mean().backward()
    assert sum(p.grad.square().sum() for p in head.matcher.parameters()) > 0


def test_metric_camera_guard_accepts_off_axis_but_rejects_failures():
    images, K, E = camera_fixture()
    K[..., 0, 2] = -25.  # Valid cropped camera, not forcibly centred.
    validate_metric_cameras(images, E, K, 3)
    bad = K.clone()
    bad[..., 1, 1] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_metric_cameras(images, E, bad, 3)
    bad = K.clone()
    bad[..., 1, 1] = 0.
    with pytest.raises(ValueError, match="positive focal"):
        validate_metric_cameras(images, E, bad, 3)
    bad_E = E.clone()
    bad_E[..., 0, 0] = -1.
    with pytest.raises(ValueError, match="proper rotations"):
        validate_metric_cameras(images, bad_E, K, 3)


def test_native_convnext_feature_rays_match_stride_convolution_centres():
    _, K, _ = camera_fixture()
    before = K.clone()
    feature_K = scale_intrinsics(native_convnext_camera_input(K, 8), (64, 64), (8, 8))
    feature_pixel = torch.tensor([3., 2., 1.])
    image_pixel = torch.tensor([3. * 8 + 3.5, 2. * 8 + 3.5, 1.])
    assert torch.allclose(torch.linalg.solve(feature_K[0, 0], feature_pixel),
                          torch.linalg.solve(K[0, 0], image_pixel))
    assert torch.equal(K, before)


def test_index_objective_anchors_affine_gauge_and_keeps_far_plane_trainable():
    criterion = build_objective("calibrated_index_l1", {"depth_min": 3., "depth_max": 80.})
    depth = torch.tensor([80., 10., 3.]).reshape(1, 1, 1, 1, 3)
    q = (depth.reciprocal() - 1. / 80) / (1. / 3 - 1. / 80)
    mask = torch.ones_like(depth)
    assert criterion(q, depth, mask)["total_loss"] == 0.
    assert criterion(.5 * q + .25, depth, mask)["total_loss"] > .1
    shifted = q.clone().requires_grad_()
    with torch.no_grad():
        shifted[..., 0] = .001  # Below the old .005 floor, still supervised.
    criterion(shifted, depth, mask)["total_loss"].backward()
    assert shifted.grad[..., 0] > 0
    with pytest.raises(ValueError, match="No valid metric"):
        criterion(q, depth, torch.zeros_like(mask))
    with pytest.raises(FloatingPointError, match="Nonfinite prediction"):
        criterion(q * float("nan"), depth, mask)


def test_clip_transform_preserves_per_frame_intrinsics_and_projection():
    native_h, native_w = 70, 149
    rng = np.random.default_rng(0)
    images = [rng.random((native_h, native_w, 3), dtype=np.float32) for _ in range(2)]
    depth = [np.full((native_h, native_w), 8., dtype=np.float32) for _ in images]
    K = np.array([[[120., 0., 50.], [0., 121., 31.], [0., 0., 1.]],
                  [[124., 0., 58.], [0., 118., 36.], [0., 0., 1.]]])
    a = data.resize_crop_clip(images, depth, K, 64, seed=5)
    b = data.resize_crop_clip(images, depth, K, 64, seed=5)
    rgb, resized_depth, transformed_K, info = a
    assert torch.equal(rgb, b[0]) and torch.equal(transformed_K, b[2])
    assert rgb.shape == (2, 3, 64, 64) and (resized_depth == 8).all()
    A = np.asarray(info["pixel_transform"])
    assert A[0, 0] != A[1, 1]  # Integer rounding requires independent sx/sy.
    point = np.array([.8, -.3, 5.])
    # Compare both projection routes at every frame, with explicit column vectors.
    assert np.allclose(transformed_K.numpy() @ point[..., None], A[None] @ (K @ point[..., None]), atol=1e-5)
    assert not torch.equal(transformed_K[0], transformed_K[1])


def test_intrinsic_resize_transform_matches_actual_opencv_sampling():
    # A coordinate ramp lets us compare the camera's inverse pixel transform
    # against the actual resized RGB, not just against the same matrix twice.
    yy, xx = np.mgrid[:32, :64].astype(np.float32)
    rgb = np.stack([xx / 64, yy / 32, np.ones_like(xx) * .5], axis=-1)
    K = np.eye(3, dtype=np.float64)[None]
    image, _, _, info = data.resize_crop_clip([rgb], [np.ones((32, 64), np.float32)],
                                            K, 64, seed=0)
    recovered = image[0].permute(1, 2, 0).numpy() * np.array([.229, .224, .225]) + np.array([.485, .456, .406])
    A = np.asarray(info["pixel_transform"])
    target_y, target_x = np.mgrid[:64, :64].astype(np.float32)
    source_x = (target_x - A[0, 2]) / A[0, 0]
    source_y = (target_y - A[1, 2]) / A[1, 1]
    interior = (source_x > 2) & (source_x < 61) & (source_y > 2) & (source_y < 29)
    # OpenCV's cubic kernel has a <=.047px ramp deviation at half-scale samples;
    # a missing half-pixel offset would shift this result by .25 native pixels.
    assert np.max(np.abs(recovered[..., 0][interior] * 64 - source_x[interior])) < .06
    assert np.max(np.abs(recovered[..., 1][interior] * 32 - source_y[interior])) < .06


def write_vkitti_scene(root, scene, frames=8):
    base = root / scene / "15-deg-left"
    rgb_dir = base / "frames/rgb/Camera_0"
    depth_dir = base / "frames/depth/Camera_0"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    extrinsic_rows, intrinsic_rows = [], []
    for frame in range(frames):
        E = np.eye(4)
        E[0, 3] = 1_000_000. + frame * .25  # Rebase in float64 before float32.
        extrinsic_rows.append([frame, 0, *E[:3].flatten()])
        intrinsic_rows.append([frame, 0, 100., 100., 47.5, 31.5])
        image = np.full((64, 96, 3), 50 + frame, np.uint8)
        depth = np.full((64, 96), 800, np.uint16)
        cv2.imwrite(str(rgb_dir / f"rgb_{frame:05}.jpg"), image)
        cv2.imwrite(str(depth_dir / f"depth_{frame:05}.png"), depth)
    np.savetxt(base / "extrinsic.txt", extrinsic_rows, header="frame cameraID matrix", comments="", fmt="%.10f")
    np.savetxt(base / "intrinsic.txt", intrinsic_rows, header="frame cameraID fx fy cx cy", comments="", fmt="%.10f")
    return base


def test_real_data_loader_path_uses_training_scenes_and_fixed_metadata(tmp_path):
    root = tmp_path / "vkitti"
    for scene in ("Scene01", "Scene02", "Scene20"):
        write_vkitti_scene(root, scene)
    clips = data.load_training_clips(root, crop_size=64, seq_len=3, num_clips=2,
                                     min_baseline_m=.4, max_baseline_m=.6)
    assert {clip["manifest"]["scene"] for clip in clips} == {"Scene01", "Scene02"}
    for clip in clips:
        assert torch.equal(clip["extrinsics"][0, 0], torch.eye(4))
        assert clip["manifest"]["max_relative_baseline_m"] == .5
        assert clip["intrinsics"].shape == (1, 3, 3, 3)
        assert (clip["depth"] == 8).all()
    with pytest.raises(ValueError, match="no threshold fallback"):
        data.load_training_clips(root, crop_size=64, seq_len=3, num_clips=1,
                                 min_baseline_m=1., max_baseline_m=2.)


def test_camera_tables_fail_on_duplicates_and_keep_float64(tmp_path):
    base = write_vkitti_scene(tmp_path, "Scene01", frames=3)
    E, K = data.read_camera_tables(base / "extrinsic.txt", base / "intrinsic.txt")
    assert E[0].dtype == K[0].dtype == np.float64
    assert (E[1] @ np.linalg.inv(E[0]))[0, 3] == .25
    with open(base / "intrinsic.txt", "a") as handle:
        handle.write("0 0 100 100 47.5 31.5\n")
    with pytest.raises(ValueError, match="Duplicate Camera_0 intrinsic"):
        data.read_camera_tables(base / "extrinsic.txt", base / "intrinsic.txt")


def test_pilot_runs_without_weights_gpu_or_method_and_refuses_overwrite(tmp_path, monkeypatch):
    images, K, E = camera_fixture()
    weight_path = tmp_path / "fixture.pth"
    weight_path.write_bytes(b"synthetic fixture, not real weights")
    clip = {"images": images, "intrinsics": K, "extrinsics": E,
            "depth": torch.full((1, 3, 1, 64, 64), 8.),
            "mask": torch.ones(1, 3, 1, 64, 64),
            "manifest": {"rgb": [str(weight_path)], "depth": [], "calibration": []}}

    class FakeBackbone(torch.nn.Module):
        embed_dims = [8, 16, 32, 64]
        patch_size = 4

        def forward(self, value):
            return [torch.randn(value.shape[0], dim, 64 // (4 * 2 ** i), 64 // (4 * 2 ** i), device=value.device)
                    for i, dim in enumerate(self.embed_dims)]

    monkeypatch.setattr(pilot, "load_clean_backbone", lambda *args: (FakeBackbone().eval(), {"synthetic": True}))
    monkeypatch.setattr(pilot, "load_training_clips", lambda *args, **kwargs: [copy.deepcopy(clip)])
    cfg = OmegaConf.load(ROOT / "config/stereogru/calibrated_volume_only.yaml")
    cfg.steps = 3
    cfg.decoder.kwargs.num_sample = 8
    cfg.decoder.kwargs.num_groups = 2
    cfg.decoder.kwargs.match_dim = 8
    cfg.decoder.kwargs.hidden_dim = 16
    result = pilot.run(cfg, weight_path, tmp_path, tmp_path / "run", torch.device("cpu"))
    assert result["status"] == "completed_implementation_pilot"
    assert result["steps"] == 3 and result["method_launched"] is False
    assert result["first_matcher_grad_after_clip"] > 0
    assert json.loads((tmp_path / "run/summary.json").read_text())["final"]["valid_pixels"] > 0
    assert (tmp_path / "run/initial_head.pth").is_file()
    assert (tmp_path / "run/baseline_head.pth").is_file()
    with pytest.raises(FileExistsError, match="no resume/overwrite"):
        pilot.run(cfg, weight_path, tmp_path, tmp_path / "run", torch.device("cpu"))