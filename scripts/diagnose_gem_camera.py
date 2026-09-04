"""Audit the camera gauge produced by GEM on held-out VKITTI clips.

The camera loss supervises translation in a scene-normalised coordinate system,
not metres, and currently omits the focal/FoV term from its total. This script
measures the resulting K/R/T quality and compares adjacent-frame warps under a
consistent gauge versus the metric-depth/normalised-translation gauge used by
the first errmap and cost-volume experiments.

Example:
    python scripts/diagnose_gem_camera.py \
      --config config/vkitti/vkitti_ms_gem.yaml \
      --ckpt checkpoint/vkitti_ms_gem/final_model.pth \
      --vkitti-root /mnt/data/PROJECT_CHEN/data/train/vkitti2/vkitti
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_ROOT = os.path.join(ROOT, "evaluation", "inference")
for path in (ROOT, INFERENCE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset.dataset_mix import DepthVideoDataset, safe_collate  # noqa: E402
from model.factory import build_gemdepth_from_config  # noqa: E402
from model.util.warp import _inverse_warp  # noqa: E402
from protocol import load_experiment_config  # noqa: E402


def normalize_ground_truth_camera(depth, mask, intrinsic, extrinsic):
    """Reproduce CameraLoss's relative-pose / average-scene-depth gauge."""
    batch, frames, _, height, width = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij")
    pixels = torch.stack([xx, yy, torch.ones_like(xx)]).reshape(1, 3, -1)
    pixels = pixels.expand(batch, -1, -1)
    rays = torch.bmm(torch.inverse(intrinsic), pixels)
    camera = rays[:, None] * depth.reshape(batch, frames, 1, -1)

    rotation = extrinsic[:, :, :3, :3]
    translation = extrinsic[:, :, :3, 3:4]
    world = torch.matmul(rotation.transpose(-1, -2), camera - translation)
    camera_zero = (torch.matmul(rotation[:, :1], world)
                   + translation[:, :1])
    radial = camera_zero.norm(dim=2).reshape(batch, frames, height, width)
    valid = mask.squeeze(2)
    average_scale = ((radial * valid).sum(dim=(1, 2, 3))
                     / valid.sum(dim=(1, 2, 3)).clamp_min(1.0))
    average_scale = average_scale.clamp(min=1e-6, max=1e6)

    first_inverse = torch.inverse(extrinsic[:, 0])
    relative = torch.matmul(extrinsic, first_inverse[:, None])
    relative = relative.clone()
    relative[:, :, :3, 3] /= average_scale[:, None, None]
    return relative, depth / average_scale[:, None, None, None, None], average_scale


def rotation_error_degrees(prediction, target):
    relative = torch.matmul(
        prediction, target.transpose(-1, -2))
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def _warp_stats(images, depth, intrinsic, extrinsic):
    """Adjacent forward/backward warp validity and residual."""
    batch, frames, channels, height, width = images.shape
    if not (torch.isfinite(depth).all() and torch.isfinite(intrinsic).all()
            and torch.isfinite(extrinsic).all()):
        return 0.0, float("nan")
    inv_k = torch.inverse(intrinsic)
    valid_values, residual_values = [], []
    for offset in (-1, 1):
        first, last = max(0, -offset), min(frames, frames - offset)
        if last <= first:
            continue
        target_index = torch.arange(first, last, device=images.device)
        source_index = target_index + offset
        count = len(target_index)
        flat = batch * count
        warped, valid = _inverse_warp(
            depth[:, target_index].reshape(flat, 1, height, width),
            images[:, source_index].reshape(flat, channels, height, width),
            inv_k[:, target_index].reshape(flat, 3, 3),
            intrinsic[:, source_index].reshape(flat, 3, 3),
            extrinsic[:, target_index].reshape(flat, 4, 4),
            extrinsic[:, source_index].reshape(flat, 4, 4))
        target_image = images[:, target_index].reshape(
            flat, channels, height, width)
        residual = (target_image - warped).abs().mean(dim=1, keepdim=True)
        valid_values.append(valid.mean())
        residual_values.append(
            (residual * valid).sum() / valid.sum().clamp_min(1.0))
    return (float(torch.stack(valid_values).mean()),
            float(torch.stack(residual_values).mean()))


def _mean(values):
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _load_state(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"] if (
        isinstance(blob, dict) and "model_state_dict" in blob) else blob
    return {(key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vkitti-root", required=True)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--output", default="runlogs/gem_camera_diagnostic.json")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_experiment_config(args.config)
    if not bool(cfg.model.use_gem):
        raise ValueError("Camera diagnostics require model.use_gem=true")

    model = build_gemdepth_from_config(
        cfg, load_backbone_pretrained=False)
    model.load_state_dict(_load_state(args.ckpt), strict=True)
    model = model.to(device).eval()

    dataset_kwargs = dict(cfg.dataset.val)
    dataset_kwargs["data_dirs"] = [args.vkitti_root]
    dataset_kwargs["mode"] = "val"
    dataset = DepthVideoDataset(**dataset_kwargs)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=safe_collate)

    values = {key: [] for key in (
        "focal_relative_error", "principal_point_error_pixels",
        "rotation_error_degrees", "translation_direction_cosine",
        "translation_magnitude_ratio", "scene_scale_metres",
        "gt_warp_valid", "gt_warp_residual",
        "pred_consistent_warp_valid", "pred_consistent_warp_residual",
        "pred_mismatched_warp_valid", "pred_mismatched_warp_residual")}
    nonfinite_intrinsics = nonfinite_extrinsics = frames_seen = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            images = batch["image"].to(device)
            depth = batch["depth"].to(device).float()
            mask = ((depth > 1e-3) & (depth <= args.max_depth)).float()
            intrinsic_gt = batch["IntM"].to(device).float()
            if isinstance(batch["poses"], (list, tuple)):
                extrinsic_gt = torch.stack(batch["poses"], dim=1)
            else:
                extrinsic_gt = batch["poses"]
            extrinsic_gt = extrinsic_gt.to(device).float()

            _, _, extrinsic_pred, intrinsic_pred = model(images)
            if extrinsic_pred is None or intrinsic_pred is None:
                raise RuntimeError("GEM returned no camera prediction")
            intrinsic_pred = intrinsic_pred.float()
            extrinsic_pred = extrinsic_pred.float()
            batch_size, frame_count = extrinsic_pred.shape[:2]
            frames_seen += batch_size * frame_count
            nonfinite_intrinsics += int((~torch.isfinite(
                intrinsic_pred).flatten(2).all(2)).sum())
            nonfinite_extrinsics += int((~torch.isfinite(
                extrinsic_pred).flatten(2).all(2)).sum())

            gt_relative, depth_normalized, scene_scale = (
                normalize_ground_truth_camera(
                    depth, mask, intrinsic_gt, extrinsic_gt))
            intrinsic_gt_frames = intrinsic_gt[:, None].expand(
                -1, frame_count, -1, -1)

            focal_error = (
                (intrinsic_pred[..., (0, 1), (0, 1)]
                 - intrinsic_gt_frames[..., (0, 1), (0, 1)]).abs()
                / intrinsic_gt_frames[..., (0, 1), (0, 1)].abs().clamp_min(1e-6))
            centre_error = torch.stack([
                intrinsic_pred[..., 0, 2] - intrinsic_gt_frames[..., 0, 2],
                intrinsic_pred[..., 1, 2] - intrinsic_gt_frames[..., 1, 2],
            ], dim=-1).norm(dim=-1)
            rotation_error = rotation_error_degrees(
                extrinsic_pred[..., :3, :3], gt_relative[..., :3, :3])

            pred_t = extrinsic_pred[..., :3, 3]
            gt_t = gt_relative[..., :3, 3]
            valid_t = ((pred_t.norm(dim=-1) > 1e-6)
                       & (gt_t.norm(dim=-1) > 1e-6)
                       & torch.isfinite(pred_t).all(-1))
            if valid_t.any():
                direction = torch.nn.functional.cosine_similarity(
                    pred_t[valid_t], gt_t[valid_t], dim=-1)
                ratio = (pred_t[valid_t].norm(dim=-1)
                         / gt_t[valid_t].norm(dim=-1).clamp_min(1e-6))
                values["translation_direction_cosine"].extend(
                    direction.cpu().tolist())
                values["translation_magnitude_ratio"].extend(
                    ratio.cpu().tolist())

            values["focal_relative_error"].extend(
                focal_error.flatten().cpu().tolist())
            values["principal_point_error_pixels"].extend(
                centre_error.flatten().cpu().tolist())
            values["rotation_error_degrees"].extend(
                rotation_error.flatten().cpu().tolist())
            values["scene_scale_metres"].extend(
                scene_scale.flatten().cpu().tolist())

            gt_stats = _warp_stats(
                images.float(), depth, intrinsic_gt_frames, extrinsic_gt)
            pred_consistent = _warp_stats(
                images.float(), depth_normalized,
                intrinsic_pred, extrinsic_pred)
            pred_mismatched = _warp_stats(
                images.float(), depth, intrinsic_pred, extrinsic_pred)
            for prefix, stats in (
                    ("gt", gt_stats),
                    ("pred_consistent", pred_consistent),
                    ("pred_mismatched", pred_mismatched)):
                values[f"{prefix}_warp_valid"].append(stats[0])
                values[f"{prefix}_warp_residual"].append(stats[1])

    summary = {key: _mean(items) for key, items in values.items()}
    summary.update({
        "frames": frames_seen,
        "batches": min(args.max_batches, len(loader)),
        "nonfinite_intrinsics_fraction": (
            nonfinite_intrinsics / max(frames_seen, 1)),
        "nonfinite_extrinsics_fraction": (
            nonfinite_extrinsics / max(frames_seen, 1)),
        "camera_translation_gauge": (
            "scene-normalised; multiply predicted T by scene_scale_metres "
            "before combining with metric depth"),
    })
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
