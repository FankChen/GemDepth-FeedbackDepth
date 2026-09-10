"""Fixed, hashable train/dev manifest for the calibrated C0/C1 controls.

This new protocol does not change the old dataset/pilot or use test scenes.
Scene quotas and minimum start gaps are strict: no replacement data, relaxed
motion thresholds, random retries or error-based selection are permitted.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset.stereogru_calibration import read_camera_tables, resize_crop_clip
from dataset.vkitti_split import TRAIN_SCENES
from model.dpt_calibrated_volume_only_convnext import validate_metric_cameras


class ManifestQuotaError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _unique(root, suffix):
    found = sorted(root.glob(f"**/{suffix}"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one unambiguous {suffix}, found {found}")
    return found[0]


def _frame_files(directory, prefix, extensions):
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if not path.stem.startswith(prefix + "_"):
            raise ValueError(f"Unexpected frame filename: {path}")
        frame = int(path.stem.split("_")[-1])
        if frame in result:
            raise ValueError(f"Duplicate frame {frame} in {directory}")
        result[frame] = str(path.resolve())
    if not result:
        raise ValueError(f"No frames in {directory}")
    return result


def select_spaced_candidates(candidates, count, min_start_gap):
    """Maximum-cardinality greedy spacing, then uniform thinning in time.

    Earliest-finish selection is optimal for equal-length forbidden start gaps;
    random selection could incorrectly claim an attainable quota is impossible.
    """
    spaced = []
    for row in sorted(candidates, key=lambda row: row["frames"][0]):
        if not spaced or row["frames"][0] - spaced[-1]["frames"][0] >= min_start_gap:
            spaced.append(row)
    if len(spaced) < count:
        return [], len(spaced)
    indices = np.linspace(0, len(spaced) - 1, count, dtype=int)
    return [spaced[int(index)] for index in indices], len(spaced)


def _scene_candidates(root, scene, config):
    suffix = f"{scene}/{config['variation']}"
    rgb = _frame_files(_unique(root, suffix + "/frames/rgb/Camera_0"), "rgb", {".jpg", ".png"})
    depth = _frame_files(_unique(root, suffix + "/frames/depth/Camera_0"), "depth", {".png"})
    extrinsic_path = _unique(root, suffix + "/extrinsic.txt")
    intrinsic_path = _unique(root, suffix + "/intrinsic.txt")
    E, K = read_camera_tables(extrinsic_path, intrinsic_path)
    if set(rgb) != set(depth):
        raise ValueError(f"{scene} RGB/depth frame sets differ; no silent intersection")
    frames = sorted(rgb)
    if any(frame not in E or frame not in K for frame in frames):
        raise ValueError(f"{scene} missing per-frame camera metadata")
    for frame in frames:
        if not np.isfinite(E[frame]).all() or not np.isfinite(K[frame]).all():
            raise ValueError(f"Nonfinite camera metadata at {scene}/{frame}")
    candidates = []
    consecutive = 0
    for index in range(len(frames) - config["seq_len"] + 1):
        window = frames[index:index + config["seq_len"]]
        if any(b - a != 1 for a, b in zip(window, window[1:])):
            continue
        consecutive += 1
        relative = np.stack([E[frame] for frame in window]) @ np.linalg.inv(E[window[0]])
        baseline = float(np.linalg.norm(relative[:, :3, 3], axis=-1).max())
        if config["min_baseline_m"] <= baseline <= config["max_baseline_m"]:
            candidates.append({
                "id": f"{scene}/{config['variation']}/Camera_0/{window[0]:05d}",
                "scene": scene, "variation": config["variation"], "frames": window,
                "rgb": [rgb[frame] for frame in window], "depth": [depth[frame] for frame in window],
                "calibration": [str(extrinsic_path.resolve()), str(intrinsic_path.resolve())],
                "native_intrinsics": [K[frame].tolist() for frame in window],
                "extrinsics": relative.tolist(), "max_relative_baseline_m": baseline,
            })
    return candidates, {"native_frames": len(frames), "consecutive_windows": consecutive,
                        "motion_eligible_windows": len(candidates)}


def load_record(record, crop_size):
    images, depths = [], []
    for rgb_path, depth_path in zip(record["rgb"], record["depth"]):
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise OSError(f"Unreadable selected RGB/depth: {rgb_path}, {depth_path}")
        if depth.ndim != 2 or depth.dtype != np.uint16:
            raise ValueError(f"Expected uint16 VKITTI depth in centimetres: {depth_path}")
        images.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.)
        valid = (depth > 0) & (depth < 65500)
        depths.append(np.where(valid, depth.astype(np.float32) / 100., 0.))
    images, depth, K, transform = resize_crop_clip(images, depths, record["native_intrinsics"],
                                                  crop_size, record["crop_seed"])
    clip = {"images": images[None], "depth": depth[None], "mask": (depth > 0)[None],
            "intrinsics": K[None], "extrinsics": torch.tensor(record["extrinsics"], dtype=torch.float32)[None]}
    validate_metric_cameras(clip["images"], clip["extrinsics"], clip["intrinsics"], len(record["frames"]))
    if "transform" in record and (transform != record["transform"] or K.tolist() != record["intrinsics"]):
        raise ValueError(f"Recorded crop/intrinsics did not replay: {record['id']}")
    return clip, transform, K.tolist()


def target_support(clip, depth_min, depth_max):
    depth = clip["depth"]
    return clip["mask"].bool() & torch.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)


def support_sha256(support):
    shape = json.dumps(list(support.shape)).encode()
    return hashlib.sha256(shape + b"\0" + support.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def validate_manifest(manifest):
    """Re-check split, quota, gap and support invariants on a saved manifest."""
    config = manifest["dataset"]
    observed_ids, observed_paths, observed_scenes = set(), set(), set()
    if set(manifest["splits"]) != set(config["splits"]):
        raise ValueError("Manifest split names differ from configuration")
    for split, specification in config["splits"].items():
        scenes = list(specification["scenes"])
        if len(set(scenes)) != len(scenes) or set(scenes) - TRAIN_SCENES or set(scenes) & observed_scenes:
            raise ValueError("Scene leakage, duplicate scenes or non-training-pool scene in manifest")
        observed_scenes.update(scenes)
        rows = manifest["splits"][split]
        if {row["scene"] for row in rows} != set(scenes):
            raise ValueError("Manifest contains an unexpected/missing scene")
        for scene in scenes:
            selected = sorted([row for row in rows if row["scene"] == scene], key=lambda row: row["frames"][0])
            if len(selected) != specification["clips_per_scene"]:
                raise ValueError(f"Scene quota changed: {scene}")
            starts = [row["frames"][0] for row in selected]
            if any(b - a < config["min_start_gap"] for a, b in zip(starts, starts[1:])):
                raise ValueError(f"Minimum start gap violated in {scene}")
            for row in selected:
                frames = row["frames"]
                if (len(frames) != config["seq_len"] or any(b - a != 1 for a, b in zip(frames, frames[1:]))
                        or row["variation"] != config["variation"]):
                    raise ValueError("Nonconsecutive frames or changed variation")
                if row["id"] in observed_ids or set(row["rgb"]) & observed_paths:
                    raise ValueError("Duplicate clip/overlapping RGB frames")
                observed_ids.add(row["id"])
                observed_paths.update(row["rgb"])
                if (not config["min_baseline_m"] <= row["max_relative_baseline_m"] <= config["max_baseline_m"]
                        or row["valid_pixels"] <= 0):
                    raise ValueError("Changed motion quota or empty supervision")
    return True


def build_manifest(root, config, depth_min, depth_max):
    config = json.loads(json.dumps(config))
    root = Path(root).resolve()
    if (config["seq_len"] < 2 or config["min_start_gap"] < config["seq_len"]
            or not 0 < config["min_baseline_m"] <= config["max_baseline_m"]
            or not 0 < depth_min < depth_max):
        raise ValueError("Invalid matched data geometry/quota settings")
    if config["crop_size"] < 64 or config["crop_size"] % 32:
        raise ValueError("crop_size must be >=64 and divisible by 32")
    all_scenes = [scene for value in config["splits"].values() for scene in value["scenes"]]
    if len(set(all_scenes)) != len(all_scenes) or set(all_scenes) - TRAIN_SCENES:
        raise ValueError("Train/dev scenes must be disjoint members of the training pool")
    splits, report, shortages = {}, {}, []
    for split, specification in config["splits"].items():
        quota = int(specification["clips_per_scene"])
        if quota < 1 or not specification["scenes"]:
            raise ValueError("Each split needs scenes and a positive per-scene quota")
        splits[split] = []
        for scene in specification["scenes"]:
            candidates, counts = _scene_candidates(root, scene, config)
            selected, available = select_spaced_candidates(candidates, quota, config["min_start_gap"])
            report[scene] = {**counts, "split": split, "spacing_eligible_windows": available,
                             "required": quota, "selected": len(selected)}
            if not selected:
                shortages.append(f"{scene}: {available}/{quota} spaced clips")
            splits[split].extend(selected)
    if shortages:
        raise ManifestQuotaError("Insufficient fixed quotas; no threshold/data fallback: " + "; ".join(shortages), report)
    inputs = {}
    for rows in splits.values():
        for row in rows:
            for path in row["rgb"] + row["depth"] + row["calibration"]:
                if path not in inputs:
                    inputs[path] = file_sha256(path)
    counter = 0
    for rows in splits.values():
        for row in rows:
            row["crop_seed"] = int(config["crop_seed"]) + counter
            counter += 1
            clip, transform, intrinsics = load_record(row, config["crop_size"])
            support = target_support(clip, depth_min, depth_max)
            if not support.any():
                raise ValueError(f"Selected clip has no valid depth target: {row['id']}; no replacement")
            row.update(transform=transform, intrinsics=intrinsics,
                       valid_pixels=int(support.sum()), support_sha256=support_sha256(support))
    manifest = {"version": 1, "root": str(root), "dataset": config,
                "depth_bounds": [float(depth_min), float(depth_max)], "splits": splits,
                "selection_report": report, "input_sha256": inputs,
                "selection": "motion-filter, earliest maximum spaced subset, uniformly thin; never select by error"}
    validate_manifest(manifest)
    return manifest


def load_manifest_clips(manifest, progress_check=None):
    check = progress_check if progress_check is not None else lambda: None
    validate_manifest(manifest)
    for path, digest in manifest["input_sha256"].items():
        check()
        if file_sha256(path) != digest:
            raise ValueError(f"Input file changed since manifest: {path}")
    clips = {}
    for rows in manifest["splits"].values():
        for row in rows:
            check()
            clip, _, _ = load_record(row, manifest["dataset"]["crop_size"])
            check()
            support = target_support(clip, *manifest["depth_bounds"])
            if int(support.sum()) != row["valid_pixels"] or support_sha256(support) != row["support_sha256"]:
                raise ValueError(f"Supervision mask changed: {row['id']}")
            clips[row["id"]] = clip
    return clips