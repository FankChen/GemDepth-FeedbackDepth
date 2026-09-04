"""Compare sequence-wise and frame-wise affine alignment on saved predictions.

The standard evaluator fits one scale/shift pair to an entire sequence. The
training objective also has a frame-wise affine-invariant term. This diagnostic
quantifies the gap between those gauges before spending a training run on a
CARVE-style joint objective.

Example:
    python scripts/diagnose_alignment.py \
      --benchmark /mnt/data/PROJECT_CHEN/data/eval \
      --pred-root output_eval/ms_norm --dataset kitti
"""

import argparse
import csv
import json
import os
import sys

import cv2
import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ROOT = os.path.join(ROOT, "evaluation", "eval")
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from alignment import require_finite, stable_scale_and_shift  # noqa: E402


DATASETS = {
    "kitti": dict(base="kitti", json="kitti_video.json", max_len=110,
                  max_depth=80.0, crop=(0, 374, 0, 1242)),
    "kitti_500": dict(base="kitti", json="kitti_video_500.json", max_len=500,
                      max_depth=80.0, crop=(0, 374, 0, 1242)),
    "sintel": dict(base="sintel", json="sintel_video.json", max_len=100,
                   max_depth=70.0, crop=(0, 436, 0, 1024)),
    "bonn": dict(base="bonn", json="bonn_video.json", max_len=110,
                 max_depth=10.0, crop=(0, 480, 0, 640)),
    "bonn_500": dict(base="bonn", json="bonn_video_500.json", max_len=500,
                     max_depth=10.0, crop=(0, 480, 0, 640)),
    "scannet": dict(base="scannet", json="scannet_video.json", max_len=90,
                    max_depth=10.0, crop=(8, -8, 11, -11)),
    "scannet_500": dict(base="scannet", json="scannet_video_500.json", max_len=500,
                        max_depth=10.0, crop=(8, -8, 11, -11)),
}


def _depth(path, factor):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".npy":
        value = np.load(path)
    elif suffix == ".dsp5":
        with h5py.File(path, "r") as handle:
            disparity = handle["disparity"][()][::2, ::2]
        value = np.full_like(disparity, -1.0, dtype=np.float32)
        valid = disparity > 0
        value[valid] = 1.0 / disparity[valid]
    else:
        value = cv2.imread(path, -1).astype(np.float32)
    value = value.astype(np.float32) / float(factor)
    value[value == 0] = -1
    return value


def _prediction(path, target_shape):
    value = require_finite(np.load(path), "prediction", path).astype(np.float32)
    if value.shape != target_shape:
        value = cv2.resize(value, (target_shape[1], target_shape[0]))
    return np.clip(value, 1e-3, None)


def _metrics(aligned_disparity, depth, valid, max_depth):
    prediction = 1.0 / np.clip(aligned_disparity, 1e-6, None)
    prediction = np.clip(prediction, 1e-3, max_depth)
    p, g = prediction[valid], depth[valid]
    return {
        "absrel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(np.maximum(p / g, g / p) < 1.25)),
    }


def compare_alignment(prediction, depth, valid, max_depth):
    """Return metrics and calibration drift under both alignment gauges."""
    prediction = require_finite(prediction, "prediction").astype(np.float64)
    depth = require_finite(depth, "depth").astype(np.float64)
    target = np.zeros_like(depth)
    target[valid] = 1.0 / depth[valid]

    sequence_scale, sequence_shift = stable_scale_and_shift(
        prediction[valid], target[valid])
    sequence_aligned = sequence_scale * prediction + sequence_shift

    frame_aligned = np.empty_like(prediction)
    frame_scales, frame_shifts = [], []
    for index in range(prediction.shape[0]):
        frame_valid = valid[index]
        if not frame_valid.any():
            frame_aligned[index] = sequence_aligned[index]
            frame_scales.append(sequence_scale)
            frame_shifts.append(sequence_shift)
            continue
        scale, shift = stable_scale_and_shift(
            prediction[index][frame_valid], target[index][frame_valid])
        frame_aligned[index] = scale * prediction[index] + shift
        frame_scales.append(scale)
        frame_shifts.append(shift)

    frame_scales = np.asarray(frame_scales)
    frame_shifts = np.asarray(frame_shifts)
    target_median = float(np.median(target[valid]))
    scale_denominator = max(abs(float(frame_scales.mean())), 1e-8)
    shift_denominator = max(abs(target_median), 1e-8)
    return {
        "sequence": _metrics(sequence_aligned, depth, valid, max_depth),
        "frame": _metrics(frame_aligned, depth, valid, max_depth),
        "sequence_scale": float(sequence_scale),
        "sequence_shift": float(sequence_shift),
        "frame_scale_mean": float(frame_scales.mean()),
        "frame_scale_std": float(frame_scales.std()),
        "frame_scale_cv": float(frame_scales.std() / scale_denominator),
        "frame_shift_mean": float(frame_shifts.mean()),
        "frame_shift_std": float(frame_shifts.std()),
        "frame_shift_normalized_std": float(frame_shifts.std() / shift_denominator),
        "frame_scale_step": float(np.mean(np.abs(np.diff(frame_scales))))
            if len(frame_scales) > 1 else 0.0,
        "frame_shift_step": float(np.mean(np.abs(np.diff(frame_shifts))))
            if len(frame_shifts) > 1 else 0.0,
    }


def _sequence_arrays(value, spec, benchmark, pred_root):
    predictions, depths = [], []
    a, b, c, d = spec["crop"]
    for frame in value[:spec["max_len"]]:
        image = frame["image"]
        pred_path = os.path.join(pred_root, spec["base"], image)
        pred_path = os.path.splitext(pred_path)[0] + ".npy"
        gt_path = os.path.join(benchmark, spec["base"], frame["gt_depth"])
        depth = _depth(gt_path, frame["factor"])[a:b, c:d]
        predictions.append(_prediction(pred_path, depth.shape))
        depths.append(depth)
    prediction = np.stack(predictions)
    depth = np.stack(depths)
    valid = (depth > 1e-3) & (depth < spec["max_depth"])
    return prediction, depth, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="kitti")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    json_path = os.path.join(args.benchmark, spec["base"], spec["json"])
    with open(json_path, "r") as handle:
        sequences = json.load(handle)[spec["base"]]

    rows = []
    for item in sequences:
        for name, value in item.items():
            prediction, depth, valid = _sequence_arrays(
                value, spec, args.benchmark, args.pred_root)
            result = compare_alignment(
                prediction, depth, valid, spec["max_depth"])
            row = {"sequence": name, "frames": len(prediction)}
            for gauge in ("sequence", "frame"):
                for metric, score in result[gauge].items():
                    row[f"{gauge}_{metric}"] = score
            row.update({key: value for key, value in result.items()
                        if key not in ("sequence", "frame")})
            row["absrel_gain_frame_minus_sequence"] = (
                row["sequence_absrel"] - row["frame_absrel"])
            rows.append(row)

    output = args.output or os.path.join(
        args.pred_root, f"alignment_diagnostic_{args.dataset}.csv")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"dataset={args.dataset} sequences={len(rows)} output={output}")
    for key in ("sequence_absrel", "frame_absrel", "sequence_rmse",
                "frame_rmse", "frame_scale_cv",
                "frame_shift_normalized_std"):
        values = np.asarray([row[key] for row in rows])
        print(f"{key}: mean={values.mean():.6f} median={np.median(values):.6f}")
    gain = np.asarray([
        row["absrel_gain_frame_minus_sequence"] for row in rows])
    print(f"frame_alignment_absrel_gain: mean={gain.mean():.6f} "
          f"median={np.median(gain):.6f}")


if __name__ == "__main__":
    main()
