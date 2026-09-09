"""Test whether backbone matching features contain usable metric-depth evidence.

This is a zero-training gate for further cost-volume work.  It captures the exact
features and GEM camera tensors immediately before the configured decoder, then
ranks the ground-truth depth hypothesis under controlled camera substitutions.
Poor ``gt_metric`` retrieval localises a problem to features/sampling/visibility;
it does NOT disprove StereoGRU. A learned matcher trained with broken cameras
can itself be damaged. Flat volumes must score at chance, not 100% top-1.

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_ROOT = os.path.join(ROOT, "evaluation", "inference")
for path in (ROOT, INFERENCE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset.dataset_mix import DepthVideoDataset  # noqa: E402
from model.factory import build_gemdepth_from_config  # noqa: E402
from model.util.cost_volume import depth_hypotheses, groupwise_correlation  # noqa: E402
from model.util.warp import plane_sweep_warp, scale_intrinsics  # noqa: E402
from protocol import load_experiment_config  # noqa: E402
from diagnose_gem_camera import (diagnostic_batches,  # noqa: E402
                                 normalize_ground_truth_camera)


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


def _target_bin_support(target_depth, target_mask, samples):
    target = target_depth[:, 0]
    target_mask = (target_mask[:, 0].bool() & torch.isfinite(target)
                   & (target > 0))
    in_range = (target_mask & (target >= samples.amin(dim=1))
                & (target <= samples.amax(dim=1)))
    # Lookup is in inverse-depth INDEX units, not nearest metric-depth distance.
    true_index = (samples.reciprocal()
                  - target.clamp_min(1e-12).reciprocal()[:, None]).abs().argmin(dim=1)
    return true_index, target_mask, in_range


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
    true_index, target_mask, in_range = _target_bin_support(
        target_depth, target_mask, samples)
    true_valid = valid.gather(1, true_index[:, None]).squeeze(1)
    eligible = in_range & true_valid
    eligible_count = int(eligible.sum())
    target_count = int(target_mask.sum())
    base = {
        "eligible_pixels": eligible_count,
        "target_pixels": target_count,
        "in_range_pixels": int(in_range.sum()),
        "valid_fraction": eligible_count / max(target_count, 1),
    }
    if eligible_count == 0:
        return dict(base, **{key: float("nan") for key in (
            "top1", "top3", "top1_chance", "top3_chance", "unique_top1",
            "tied_top_fraction", "mean_valid_bins", "mean_rank", "mean_bin_error",
            "true_bin_margin", "score_std")})

    masked_scores = scores.masked_fill(~valid, -torch.inf)
    true_score = masked_scores.gather(1, true_index[:, None]).squeeze(1)
    better = ((masked_scores > true_score[:, None]) & valid).sum(dim=1)
    tied = ((masked_scores == true_score[:, None]) & valid).sum(dim=1).clamp_min(1)
    rank = 1 + better + (tied - 1).float() / 2
    winner = masked_scores.argmax(dim=1)
    valid_count = valid.sum(dim=1).clamp_min(1)
    # Fractional credit is the expected hit rate under uniform tie breaking.
    # In particular an all-equal D-bin volume gets 1/D, not 1, at top-1.
    top1 = ((1 - better).float() / tied).clamp(0, 1)
    top3 = ((3 - better).float() / tied).clamp(0, 1)

    other_scores = masked_scores.clone()
    other_scores.scatter_(1, true_index[:, None], -torch.inf)
    margin = true_score - other_scores.max(dim=1).values
    finite_margin = margin[eligible & torch.isfinite(margin)]
    score_mean = scores.masked_fill(~valid, 0).sum(dim=1) / valid_count
    score_variance = ((scores - score_mean[:, None]).square().masked_fill(~valid, 0)
                      .sum(dim=1) / valid_count)
    best_score = masked_scores.amax(dim=1, keepdim=True)
    tied_best = ((masked_scores == best_score) & valid).sum(dim=1) > 1

    return {
        **base,
        "top1": float(top1[eligible].mean()),
        "top3": float(top3[eligible].mean()),
        "top1_chance": float((1.0 / valid_count[eligible]).mean()),
        "top3_chance": float((3.0 / valid_count[eligible]).clamp(max=1).mean()),
        "unique_top1": float(((better == 0) & (tied == 1))[eligible].float().mean()),
        "tied_top_fraction": float(tied_best[eligible].float().mean()),
        "mean_valid_bins": float(valid_count[eligible].float().mean()),
        "mean_rank": float(rank[eligible].float().mean()),
        "mean_bin_error": float(
            (winner[eligible] - true_index[eligible]).abs().float().mean()),
        "true_bin_margin": float(finite_margin.mean()) if finite_margin.numel() else float("nan"),
        "score_std": float(score_variance[eligible].sqrt().mean()),
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


def trace_decoder(head, features, images, extrinsics, intrinsics):
    """Observe the unchanged eval decoder using removable hooks, without training.

    Record RAW volume, regularised logits and every lookup/update separately.
    Initial output loss alone cannot establish whether the volume is informative.
    """
    report = {"lookups": [], "updates": []}

    def stats(value):
        value = value.detach().float()
        finite = torch.isfinite(value)
        selected = value[finite]
        out = {"finite_fraction": float(finite.float().mean())}
        if selected.numel():
            out.update(min=float(selected.min()), max=float(selected.max()),
                       mean=float(selected.mean()), std=float(selected.std(unbiased=False)))
        return out

    def raw_hook(_module, args):
        raw = args[0].detach().float()
        report["raw_volume"] = dict(
            stats(raw), depth_std=float(raw.std(dim=2, unbiased=False).mean()),
            zero_pixel_fraction=float((raw.abs().sum(dim=(1, 2)) == 0).float().mean()))

    def logits_hook(_module, _args, output):
        logits = output.detach().float()
        probability = logits.softmax(dim=2)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=2)
        bins = torch.arange(logits.shape[2], device=logits.device).view(1, 1, -1, 1, 1)
        report["geometry_logits"] = dict(
            stats(logits), depth_std=float(logits.std(dim=2, unbiased=False).mean()),
            normalized_entropy=float(entropy.mean() / np.log(logits.shape[2])))
        report["initial_lowres_index"] = stats((probability * bins).sum(dim=2))

    def lookup_hook(_module, args):
        index, indexed = args
        report["lookups"].append(dict(
            stats(index),
            outside_volume_fraction=float(
                ((index < 0) | (index > head.num_sample - 1)).float().mean()),
            sampled_zero_fraction=float((indexed == 0).float().mean())))

    def update_hook(_module, _args, output):
        report["updates"].append(stats(output))

    handles = [head.volume_stem.register_forward_pre_hook(raw_hook),
               head.classifier.register_forward_hook(logits_hook),
               head.encoder.register_forward_pre_hook(lookup_hook),
               head.index_head.register_forward_hook(update_hook)]
    try:
        with torch.no_grad():
            patch_size = head.volume_stride // (2 ** head.volume_level)
            output = head(features, images.shape[-2] // patch_size, images.shape[-1] // patch_size,
                          images.shape[1], images=images,
                          extrinsics=extrinsics, intrinsics=intrinsics)
        if isinstance(output, (list, tuple)):
            output = output[-1]
        report["final_raw_output"] = dict(
            stats(output), relu_zero_fraction=float((output <= 0).float().mean()),
            above_volume_fraction=float((output > 1).float().mean()),
            training_floor_fraction=float((output < 5e-3).float().mean()))
    finally:
        for handle in handles:
            handle.remove()
    return report


def _aggregate(records, failures):
    eligible = sum(item["eligible_pixels"] for item in records)
    targets = sum(item["target_pixels"] for item in records)
    summary = {
        "pairs": len(records),
        "failures": failures,
        "eligible_pixels": eligible,
        "target_pixels": targets,
        "in_range_pixels": sum(item["in_range_pixels"] for item in records),
        "valid_fraction": eligible / max(targets, 1),
    }
    for key in ("top1", "top3", "top1_chance", "top3_chance", "unique_top1",
                "tied_top_fraction", "mean_valid_bins", "mean_rank", "mean_bin_error",
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
    parser.add_argument("--descriptor", choices=("matcher", "backbone"), default="matcher",
                        help="learned matching projection or its same-checkpoint backbone input")
    parser.add_argument("--trace-head", action="store_true",
                        help="also trace unchanged raw volume / logits / GRU lookup and updates")
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

    mode_names = (
        "gt_metric", "gt_normalized", "pred_raw", "pred_rescaled",
        "gtK_predT_raw", "gtK_predT_scaled", "predK_gtT", "gtF_center_gtT")
    records = {name: [] for name in mode_names}
    common_records = {name: [] for name in mode_names}
    paired_records = {name: {"gt_metric": [], "mode": []} for name in mode_names}
    failures = {name: 0 for name in mode_names}
    failure_examples = []
    clips = []

    with torch.no_grad():
        for sample_index, batch in diagnostic_batches(dataset, args.max_batches):
            images = batch["image"].to(device)
            depth = batch["depth"].to(device).float()
            mask = ((depth > 1e-3) & (depth <= head.depth_max)
                    & torch.isfinite(depth) & batch["mask"].to(device).bool()).float()
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

            descriptors = head.matcher(features) if args.descriptor == "matcher" else features.float()
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
            extrinsic_normalized, depth_normalized, metric_unit = (
                normalize_ground_truth_camera(
                    depth, mask, intrinsic_gt, extrinsic_gt))
            depth_normalized_feature = F.interpolate(
                depth_normalized.flatten(0, 1), size=(height, width),
                mode="nearest").unflatten(0, (batch_size, frame_count))
            extrinsic_pred_scaled = _scale_translation(
                extrinsic_pred, metric_unit)
            centered_gt = intrinsic_gt_frames.clone()
            centered_gt[..., 0, 2] = images.shape[-1] / 2
            centered_gt[..., 1, 2] = images.shape[-2] / 2
            centered_gt_feature = scale_intrinsics(
                centered_gt, images.shape[-2:], (height, width))
            clips.append({
                "sample_index": sample_index,
                "paths": batch["path"],
                "translation_unit_metres": metric_unit.cpu().tolist(),
                "intrinsic_gt": intrinsic_gt.cpu().tolist(),
                "intrinsic_pred": intrinsic_pred.cpu().tolist(),
                "nonfinite_intrinsic_frames": int(
                    (~torch.isfinite(intrinsic_pred).flatten(2).all(2)).sum()),
            })
            if args.trace_head:
                clips[-1]["head_trace"] = trace_decoder(
                    head, captured["features"], images, extrinsic_pred, intrinsic_pred)

            metric_samples = depth_hypotheses(
                head.depth_min, head.depth_max, head.num_sample,
                height, width, device, torch.float32).expand(
                    batch_size, -1, -1, -1)
            normalized_samples = metric_samples / metric_unit[:, None, None, None]
            modes = {
                "gt_metric": (intrinsic_gt_feature, extrinsic_gt,
                              depth_feature, metric_samples),
                "gt_normalized": (intrinsic_gt_feature, extrinsic_normalized,
                                  depth_normalized_feature, normalized_samples),
                "pred_raw": (intrinsic_pred_feature, extrinsic_pred,
                             depth_feature, metric_samples),
                "pred_rescaled": (intrinsic_pred_feature, extrinsic_pred_scaled,
                                  depth_feature, metric_samples),
                "gtK_predT_raw": (intrinsic_gt_feature, extrinsic_pred,
                                  depth_feature, metric_samples),
                "gtK_predT_scaled": (intrinsic_gt_feature,
                                     extrinsic_pred_scaled,
                                     depth_feature, metric_samples),
                "predK_gtT": (intrinsic_pred_feature, extrinsic_gt,
                              depth_feature, metric_samples),
                "gtF_center_gtT": (centered_gt_feature, extrinsic_gt,
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

                pair_results = {}
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
                        pair_results[name] = (
                            scores, valid, target_pair, mask_pair, pair_samples)
                    except (RuntimeError, torch.linalg.LinAlgError) as exc:
                        failures[name] += 1
                        if len(failure_examples) < 16:
                            failure_examples.append({
                                "sample_index": sample_index, "offset": offset,
                                "mode": name, "error": str(exc),
                            })

                # Pairwise shared support remains informative even if one broken mode
                # (e.g. infinite predicted focal length) empties the eight-way overlap.
                if "gt_metric" in pair_results:
                    gt_result = pair_results["gt_metric"]
                    for name, mode_result in pair_results.items():
                        pair_mask = torch.ones_like(gt_result[3], dtype=torch.bool)
                        for scores, valid, target_pair, mask_pair, pair_samples in (gt_result, mode_result):
                            true_index, _, in_range = _target_bin_support(
                                target_pair, mask_pair, pair_samples)
                            true_valid = (valid[:, 0].bool() & torch.isfinite(scores)).gather(
                                1, true_index[:, None])
                            pair_mask &= in_range[:, None] & true_valid
                        for label, result in (("gt_metric", gt_result), ("mode", mode_result)):
                            scores, valid, target_pair, _, pair_samples = result
                            paired_records[name][label].append(volume_quality(
                                scores, valid, target_pair, pair_mask, pair_samples))

                # Mode-specific visibility can make a broken camera appear better by
                # rejecting difficult pixels. Also compare on the shared true-bin
                # support; an empty intersection is reported, never filled by retries.
                if len(pair_results) == len(mode_names):
                    common = torch.ones_like(mask_pair, dtype=torch.bool)
                    for scores, valid, target_pair, mask_pair, pair_samples in pair_results.values():
                        true_index, _, in_range = _target_bin_support(
                            target_pair, mask_pair, pair_samples)
                        true_valid = (valid[:, 0].bool() & torch.isfinite(scores)).gather(
                            1, true_index[:, None])
                        common &= in_range[:, None] & true_valid
                    for name, (scores, valid, target_pair, _, pair_samples) in pair_results.items():
                        common_records[name].append(volume_quality(
                            scores, valid, target_pair, common, pair_samples))

    summary = {
        "diagnostic_version": 2,
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.ckpt),
        "descriptor": args.descriptor,
        "batches": len(clips),
        "clips": clips,
        "failure_examples": failure_examples,
        "hypotheses": {
            "count": head.num_sample,
            "depth_min": head.depth_min,
            "depth_max": head.depth_max,
        },
        "interpretation": {
            "matching_evidence": (
                "compare top-k to chance, margin and coverage on shared support; "
                "low GT-camera retrieval can reflect a damaged trained matcher, "
                "sampling, occlusion or dynamics, not a refutation of StereoGRU"),
            "fix_camera_before_training_if": (
                "gt_metric works but pred_rescaled/gtK_predT_scaled fail"),
            "gauge_check": (
                "gt_metric and gt_normalized should agree within numerical error"),
            "translation_scale": (
                "rescaling uses BOTH camera-loss normalisations, not mean scene depth; "
                "all GT-camera modes are diagnostic oracles, not RGB-only scores"),
            "tie_handling": (
                "top-k gives uniform fractional credit for exact ties; score_std is "
                "per-pixel variation across depth, not variation across image pixels"),
            "visibility_limit": (
                "in-bounds only; moving objects and true occlusion remain in the sample"),
        },
        "modes": {name: _aggregate(records[name], failures[name])
                  for name in mode_names},
        "shared_support_modes": {
            name: _aggregate(common_records[name], failures[name]) for name in mode_names},
        "paired_gt_support_modes": {
            name: {label: _aggregate(items, failures[name]) for label, items in pair.items()}
            for name, pair in paired_records.items()},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
