"""Test whether backbone matching features contain usable metric-depth evidence.

This is a zero-training gate for further cost-volume work.  It captures the exact
features and GEM camera tensors immediately before the configured decoder, then
ranks the ground-truth depth hypothesis under controlled camera substitutions.
If ``gt_metric`` cannot retrieve the true bin, changing the GRU or training the
same volume longer is not justified.

Example:
    python scripts/diagnose_cost_volume_oracle.py \
      --config config/vkitti/vkitti_costvol.yaml \
      --ckpt checkpoint/vkitti_costvol/final_model.pth \
      --vkitti-root /path/to/vkitti
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_ROOT = os.path.join(ROOT, "evaluation", "inference")
for path in (ROOT, INFERENCE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset.dataset_mix import DepthVideoDataset, safe_collate  # noqa: E402
from model.factory import build_gemdepth_from_config  # noqa: E402
from model.util.cost_volume import depth_hypotheses, groupwise_correlation  # noqa: E402
from model.util.warp import plane_sweep_warp, scale_intrinsics  # noqa: E402
from protocol import load_experiment_config  # noqa: E402
from diagnose_gem_camera import normalize_ground_truth_camera  # noqa: E402


class _HeadCaptured(RuntimeError):
    pass


def capture_decoder_inputs(model, images):
    """Run the encoder/GEM path and stop immediately before the decoder."""
    captured = {}

    def capture(_module, args, kwargs):
        captured["features"] = args[0]
        captured["extrinsics"] = kwargs.get("extrinsics")
        captured["intrinsics"] = kwargs.get("intrinsics")
        raise _HeadCaptured

    handle = model.head.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model(images)
    except _HeadCaptured:
        pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Decoder pre-hook was not reached")
    return captured


def volume_quality(scores, valid, target_depth, target_mask, samples):
    """Rank each pixel's nearest ground-truth depth bin; larger score is better.

    Args:
        scores: ``(N,D,H,W)`` scalar matching scores.
        valid: ``(N,1,D,H,W)`` projection-validity mask.
        target_depth: ``(N,1,H,W)`` depth in the same gauge as ``samples``.
        target_mask: ``(N,1,H,W)`` ground-truth validity mask.
        samples: ``(N,D,H,W)`` depth hypotheses.
    """
    valid = valid[:, 0].bool() & torch.isfinite(scores)
    target_depth = target_depth[:, 0]
    target_mask = (target_mask[:, 0].bool() & torch.isfinite(target_depth)
                   & (target_depth > 0))
    true_index = (samples - target_depth[:, None]).abs().argmin(dim=1)
    true_valid = valid.gather(1, true_index[:, None]).squeeze(1)
    eligible = target_mask & true_valid
    eligible_count = int(eligible.sum())
    target_count = int(target_mask.sum())
    if eligible_count == 0:
        return {
            "eligible_pixels": 0,
            "target_pixels": target_count,
            "valid_fraction": 0.0,
            "top1": float("nan"),
            "top3": float("nan"),
            "mean_rank": float("nan"),
            "mean_bin_error": float("nan"),
            "true_bin_margin": float("nan"),
            "score_std": float("nan"),
        }

    masked_scores = scores.masked_fill(~valid, -torch.inf)
    true_score = masked_scores.gather(1, true_index[:, None]).squeeze(1)
    rank = 1 + (masked_scores > true_score[:, None]).sum(dim=1)
    winner = masked_scores.argmax(dim=1)

    other_scores = masked_scores.clone()
    other_scores.scatter_(1, true_index[:, None], -torch.inf)
    margin = true_score - other_scores.max(dim=1).values
    finite_scores = scores[valid]

    return {
        "eligible_pixels": eligible_count,
        "target_pixels": target_count,
        "valid_fraction": eligible_count / max(target_count, 1),
        "top1": float((rank[eligible] <= 1).float().mean()),
        "top3": float((rank[eligible] <= 3).float().mean()),
        "mean_rank": float(rank[eligible].float().mean()),
        "mean_bin_error": float(
            (winner[eligible] - true_index[eligible]).abs().float().mean()),
        "true_bin_margin": float(margin[eligible].mean()),
        "score_std": float(finite_scores.std(unbiased=False)) if finite_scores.numel() else float("nan"),
    }


def _load_state(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"] if (
        isinstance(blob, dict) and "model_state_dict" in blob) else blob
    return {(key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()}


def _scale_translation(extrinsics, scale):
    result = extrinsics.clone()
    result[..., :3, 3] *= scale[:, None, None]
    return result


def _aggregate(records, failures):
    eligible = sum(item["eligible_pixels"] for item in records)
    targets = sum(item["target_pixels"] for item in records)
    summary = {
        "pairs": len(records),
        "failures": failures,
        "eligible_pixels": eligible,
        "target_pixels": targets,
        "valid_fraction": eligible / max(targets, 1),
    }
    for key in ("top1", "top3", "mean_rank", "mean_bin_error",
                "true_bin_margin", "score_std"):
        weighted = [(item[key], item["eligible_pixels"]) for item in records
                    if np.isfinite(item[key]) and item["eligible_pixels"] > 0]
        denominator = sum(weight for _, weight in weighted)
        summary[key] = (sum(value * weight for value, weight in weighted)
                        / denominator if denominator else float("nan"))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vkitti-root", required=True)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--output", default="runlogs/cost_volume_oracle.json")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_experiment_config(args.config)
    if not bool(cfg.model.use_gem):
        raise ValueError("Cost-volume oracle requires model.use_gem=true")

    model = build_gemdepth_from_config(cfg, load_backbone_pretrained=False)
    model.load_state_dict(_load_state(args.ckpt), strict=True)
    model = model.to(device).eval()
    head = model.head
    required = ("matcher", "volume_level", "num_groups", "num_sample",
                "depth_min", "depth_max", "warp_offsets")
    missing = [name for name in required if not hasattr(head, name)]
    if missing:
        raise TypeError(
            f"Configured decoder is not a cost-volume head; missing={missing}")

    dataset_kwargs = dict(cfg.dataset.val)
    dataset_kwargs["data_dirs"] = [args.vkitti_root]
    dataset_kwargs["mode"] = "val"
    dataset = DepthVideoDataset(**dataset_kwargs)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=safe_collate)

    mode_names = (
        "gt_metric", "gt_normalized", "pred_raw", "pred_rescaled",
        "gtK_predT_scaled", "predK_gtT")
    records = {name: [] for name in mode_names}
    failures = {name: 0 for name in mode_names}

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            images = batch["image"].to(device)
            depth = batch["depth"].to(device).float()
            mask = ((depth > 1e-3) & (depth <= head.depth_max)).float()
            intrinsic_gt = batch["IntM"].to(device).float()
            if isinstance(batch["poses"], (list, tuple)):
                extrinsic_gt = torch.stack(batch["poses"], dim=1)
            else:
                extrinsic_gt = batch["poses"]
            extrinsic_gt = extrinsic_gt.to(device).float()

            captured = capture_decoder_inputs(model, images)
            features = captured["features"][head.volume_level]
            extrinsic_pred = captured["extrinsics"]
            intrinsic_pred = captured["intrinsics"]
            if extrinsic_pred is None or intrinsic_pred is None:
                raise RuntimeError("GEM returned no camera prediction")
            extrinsic_pred = extrinsic_pred.float()
            intrinsic_pred = intrinsic_pred.float()

            descriptors = head.matcher(features)
            descriptors = descriptors / (
                descriptors.norm(dim=1, keepdim=True) + 1e-5)
            batch_size, frame_count = images.shape[:2]
            _, _, height, width = descriptors.shape
            descriptors = descriptors.reshape(
                batch_size, frame_count, -1, height, width)
            depth_feature = F.interpolate(
                depth.flatten(0, 1), size=(height, width), mode="nearest"
            ).unflatten(0, (batch_size, frame_count))
            mask_feature = F.interpolate(
                mask.flatten(0, 1), size=(height, width), mode="nearest"
            ).unflatten(0, (batch_size, frame_count))

            intrinsic_gt_frames = intrinsic_gt[:, None].expand(
                -1, frame_count, -1, -1)
            intrinsic_gt_feature = scale_intrinsics(
                intrinsic_gt_frames, images.shape[-2:], (height, width))
            intrinsic_pred_feature = scale_intrinsics(
                intrinsic_pred, images.shape[-2:], (height, width))
            extrinsic_normalized, depth_normalized, scene_scale = (
                normalize_ground_truth_camera(
                    depth, mask, intrinsic_gt, extrinsic_gt))
            depth_normalized_feature = F.interpolate(
                depth_normalized.flatten(0, 1), size=(height, width),
                mode="nearest").unflatten(0, (batch_size, frame_count))
            extrinsic_pred_scaled = _scale_translation(
                extrinsic_pred, scene_scale)

            metric_samples = depth_hypotheses(
                head.depth_min, head.depth_max, head.num_sample,
                height, width, device, torch.float32).expand(
                    batch_size, -1, -1, -1)
            normalized_samples = metric_samples / scene_scale[:, None, None, None]
            modes = {
                "gt_metric": (intrinsic_gt_feature, extrinsic_gt,
                              depth_feature, metric_samples),
                "gt_normalized": (intrinsic_gt_feature, extrinsic_normalized,
                                  depth_normalized_feature, normalized_samples),
                "pred_raw": (intrinsic_pred_feature, extrinsic_pred,
                             depth_feature, metric_samples),
                "pred_rescaled": (intrinsic_pred_feature, extrinsic_pred_scaled,
                                  depth_feature, metric_samples),
                "gtK_predT_scaled": (intrinsic_gt_feature,
                                     extrinsic_pred_scaled,
                                     depth_feature, metric_samples),
                "predK_gtT": (intrinsic_pred_feature, extrinsic_gt,
                              depth_feature, metric_samples),
            }

            for offset in head.warp_offsets:
                first = max(0, -offset)
                last = min(frame_count, frame_count - offset)
                if last <= first:
                    continue
                ref = torch.arange(first, last, device=device)
                src = ref + offset
                count = int(ref.numel())
                flat = batch_size * count
                source = descriptors[:, src].reshape(
                    flat, -1, height, width).float()
                reference = descriptors[:, ref].reshape(
                    flat, -1, 1, height, width).float()

                for name, (intrinsics, extrinsics, target, samples) in modes.items():
                    try:
                        pair_samples = samples[:, None].expand(
                            -1, count, -1, -1, -1).reshape(
                                flat, head.num_sample, height, width)
                        warped, valid = plane_sweep_warp(
                            source, pair_samples,
                            intrinsics[:, ref].reshape(flat, 3, 3),
                            intrinsics[:, src].reshape(flat, 3, 3),
                            extrinsics[:, ref].reshape(flat, 4, 4),
                            extrinsics[:, src].reshape(flat, 4, 4))
                        correlation = groupwise_correlation(
                            warped, reference, head.num_groups)
                        scores = correlation.mean(dim=1)
                        target_pair = target[:, ref].reshape(
                            flat, 1, height, width)
                        mask_pair = mask_feature[:, ref].reshape(
                            flat, 1, height, width)
                        records[name].append(volume_quality(
                            scores, valid, target_pair, mask_pair,
                            pair_samples))
                    except (RuntimeError, torch.linalg.LinAlgError):
                        failures[name] += 1

    summary = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.ckpt),
        "batches": min(args.max_batches, len(loader)),
        "hypotheses": {
            "count": head.num_sample,
            "depth_min": head.depth_min,
            "depth_max": head.depth_max,
        },
        "interpretation": {
            "continue_geometry_path_if": (
                "gt_metric has useful top-k retrieval and positive margin"),
            "fix_camera_before_training_if": (
                "gt_metric works but pred_rescaled/gtK_predT_scaled fail"),
            "gauge_check": (
                "gt_metric and gt_normalized should agree within numerical error"),
        },
        "modes": {name: _aggregate(records[name], failures[name])
                  for name in mode_names},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
