"""Fixed training-only VKITTI clips with explicit, correctly transformed cameras.

Reuse existing scene/window discovery, NOT its random crop or K approximation.
Read original extrinsics in float64 before first-camera rebasing. Read each
frame's intrinsic.txt entry; missing/ambiguous metadata is an error, not a guessed
camera. This is a small implementation-overfit dataset, never a benchmark split.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from dataset.dataset_mix import DepthVideoDataset
from dataset.vkitti_split import scene_is_selected


def read_camera_tables(extrinsic_path, intrinsic_path):
    extrinsics, intrinsics = {}, {}
    for row in np.loadtxt(extrinsic_path, skiprows=1, ndmin=2):
        if int(row[1]) != 0:
            continue
        matrix = np.eye(4, dtype=np.float64)
        if len(row) == 18:
            matrix[:] = row[2:].reshape(4, 4)
        elif len(row) == 14:
            matrix[:3] = row[2:].reshape(3, 4)
        else:
            raise ValueError(f"Unexpected VKITTI extrinsic row width {len(row)}")
        frame = int(row[0])
        if frame in extrinsics:
            raise ValueError(f"Duplicate Camera_0 extrinsic frame {frame}")
        extrinsics[frame] = matrix
    for row in np.loadtxt(intrinsic_path, skiprows=1, ndmin=2):
        if int(row[1]) != 0:
            continue
        if len(row) != 6:
            raise ValueError(f"Unexpected VKITTI intrinsic row width {len(row)}")
        frame = int(row[0])
        if frame in intrinsics:
            raise ValueError(f"Duplicate Camera_0 intrinsic frame {frame}")
        fx, fy, cx, cy = row[2:]
        intrinsics[frame] = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    return extrinsics, intrinsics


def resize_crop_clip(images, depths, intrinsics, crop_size, seed):
    """One deterministic crop for all frames; separate x/y resize ratios for K."""
    if crop_size < 64 or crop_size % 32:
        raise ValueError("crop_size must be >=64 and divisible by 32")
    height, width = images[0].shape[:2]
    if any(image.shape != images[0].shape for image in images):
        raise ValueError("A calibration clip must have a common native image size")
    scale = max(crop_size / height, crop_size / width)
    new_height, new_width = int(np.ceil(height * scale)), int(np.ceil(width * scale))
    rng = np.random.default_rng(seed)
    left = int(rng.integers(0, new_width - crop_size + 1))
    top = int(rng.integers(0, new_height - crop_size + 1))
    sx, sy = new_width / width, new_height / height
    # OpenCV resize maps zero-based pixel centres as u' = s * (u + .5) - .5.
    # Keep this half-pixel convention consistent with the warp's integer grid.
    matrix = np.array([[sx, 0., (sx - 1) / 2 - left],
                       [0., sy, (sy - 1) / 2 - top], [0., 0., 1.]])
    K = matrix[None] @ np.asarray(intrinsics, dtype=np.float64)
    transformed_rgb, transformed_depth = [], []
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    for rgb, depth in zip(images, depths):
        if depth.shape != (height, width):
            raise ValueError("RGB/depth native shapes differ")
        rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT)
        rgb = rgb[top:top + crop_size, left:left + crop_size]
        depth = depth[top:top + crop_size, left:left + crop_size]
        transformed_rgb.append(((rgb - mean) / std).transpose(2, 0, 1))
        transformed_depth.append(depth[None])
    info = {"native_hw": [height, width], "resized_hw": [new_height, new_width],
            "crop_xy": [left, top], "pixel_transform": matrix.tolist()}
    return (torch.from_numpy(np.stack(transformed_rgb)).float(),
            torch.from_numpy(np.stack(transformed_depth)).float(),
            torch.from_numpy(K).float(), info)


def load_training_clips(root, crop_size=256, seq_len=4, num_clips=2,
                        min_baseline_m=.5, max_baseline_m=5., seed=0):
    """Select motion-eligible training windows before learning; never retry bad data."""
    if not (0 < min_baseline_m <= max_baseline_m) or num_clips < 1 or seq_len < 2:
        raise ValueError("Invalid calibration subset settings")
    root = Path(root).resolve()
    discovery = DepthVideoDataset(mode="train", data_dirs=[str(root)], crop_size=crop_size,
                                  seq_len=seq_len, vkitti_scene_split=True)
    tables, candidates = {}, {}
    for label, window in discovery.data_paths:
        if label != "vkitti":
            raise ValueError("Calibrated control only accepts VKITTI Camera_0")
        rgb_path = Path(window[0][0])
        scene, variation = rgb_path.parents[4].name, rgb_path.parents[3].name
        if not scene_is_selected(scene, variation, "train"):
            raise ValueError(f"Refusing non-training scene {scene}/{variation}")
        if (scene, variation) not in tables:
            paths = [list(root.glob(f"**/{scene}/{variation}/{name}"))
                     for name in ("extrinsic.txt", "intrinsic.txt")]
            if any(len(found) != 1 for found in paths):
                raise FileNotFoundError(f"Need one unambiguous extrinsic.txt and intrinsic.txt for {scene}/{variation}: {paths}")
            E, K = read_camera_tables(paths[0][0], paths[1][0])
            tables[(scene, variation)] = E, K, [str(found[0]) for found in paths]
        E, K, calibration_paths = tables[(scene, variation)]
        frames = [int(Path(item[0]).stem.split("_")[-1]) for item in window]
        if any(b - a != 1 for a, b in zip(frames, frames[1:])):
            continue  # Gaps are not consecutive-frame clips; selection is recorded.
        if any(frame not in K or frame not in E for frame in frames):
            raise ValueError(f"Missing camera metadata for {scene}/{variation}/{frames}")
        relative = np.stack([E[frame] for frame in frames]) @ np.linalg.inv(E[frames[0]])
        baseline = float(np.linalg.norm(relative[:, :3, 3], axis=-1).max())
        if min_baseline_m <= baseline <= max_baseline_m:
            candidates.setdefault(scene, []).append((window, frames, relative, K, baseline, calibration_paths))
    # Spread across scenes first, then nonoverlapping windows; do not select by loss.
    ordered = []
    for scene in sorted(candidates):
        pool = candidates[scene]
        offset = len(pool) // 2
        ordered.append(pool[offset:] + pool[:offset])
    selected, used = [], set()
    for position in range(max(map(len, ordered), default=0)):
        for pool in ordered:
            if position >= len(pool):
                continue
            item = pool[position]
            paths = {row[0] for row in item[0]}
            if paths & used:
                continue
            selected.append(item)
            used |= paths
            if len(selected) == num_clips:
                break
        if len(selected) == num_clips:
            break
    if len(selected) != num_clips:
        raise ValueError(f"Only {len(selected)} nonoverlapping training clips satisfy baseline [{min_baseline_m},{max_baseline_m}]m; no threshold fallback")

    clips = []
    for index, (window, frames, relative, K, baseline, calibration_paths) in enumerate(selected):
        images, depths = [], []
        for image_path, depth_path, _pose in window:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if image is None or depth is None:
                raise OSError(f"Unreadable RGB/depth: {image_path}, {depth_path}")
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.)
            valid = (depth > 0) & (depth < 65500)
            depths.append(np.where(valid, depth.astype(np.float32) / 100., 0.))
        rgb, depth, intrinsic, transform = resize_crop_clip(
            images, depths, [K[frame] for frame in frames], crop_size, seed + index)
        clips.append({
            "images": rgb[None], "depth": depth[None], "mask": (depth > 0)[None],
            "intrinsics": intrinsic[None],
            "extrinsics": torch.from_numpy(relative).float()[None],
            "manifest": {"scene": Path(window[0][0]).parents[4].name, "frames": frames,
                         "rgb": [row[0] for row in window], "depth": [row[1] for row in window],
                         "calibration": calibration_paths, "max_relative_baseline_m": baseline,
                         "intrinsics": intrinsic.tolist(), "extrinsics": relative.tolist(),
                         "transform": transform, "selection": "training scenes, motion only; no loss-based selection"},
        })
    return clips