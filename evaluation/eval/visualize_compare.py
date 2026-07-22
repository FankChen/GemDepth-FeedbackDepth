"""Side-by-side visual comparison of two depth predictions against GT.

This mirrors ``eval.py``'s alignment EXACTLY (treat each ``.npy`` as inverse
depth / disparity, fit ``scale * pred + shift`` against ``1/GT`` with
``stable_scale_and_shift``, then invert back to depth) so the depth maps and
per-pixel AbsRel error maps shown here match the numbers in the metric table.

Output: one figure per sequence, rows = sampled frames, columns =
``[RGB | GT | A depth | B depth | A AbsRel | B AbsRel]``.

Example (run where the prediction ``.npy`` files live, e.g. Alibaba DSW):

    python evaluation/eval/visualize_compare.py \
        --benchmark_path /mnt/workspace/gemdepth_eval \
        --pred_a /path/to/temporal_baseline/output_eval  --label_a "temporal 0.105" \
        --pred_b /path/to/native_multiscale/output_eval  --label_b "native-ms 0.170" \
        --invert_b \
        --num_frames 4 --out compare_kitti.png

``--invert_b`` inverts B's ``.npy`` (metric depth -> 1/D) before the disparity
alignment. Use it only if B's on-disk ``.npy`` are RAW METRIC depth; skip it if
you already inverted them for the 0.170 eval.
"""

import argparse
import json
import os
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)
from alignment import stable_scale_and_shift  # noqa: E402

# Per-dataset crop + max depth, copied from eval.py so alignment matches.
DATASET_CFG = {
    "kitti":   dict(a=0, b=374, c=0, d=1242, max_depth=80.0),
    "sintel":  dict(a=0, b=436, c=0, d=1024, max_depth=70.0),
    "bonn":    dict(a=0, b=480, c=0, d=640,  max_depth=10.0),
    "scannet": dict(a=8, b=-8, c=11, d=-11,  max_depth=10.0),
}


def _crop(arr, cfg):
    return arr[cfg["a"]:cfg["b"], cfg["c"]:cfg["d"]]


def load_gt(path, factor, cfg):
    ext = path.split(".")[-1]
    if ext == "npy":
        gt = np.load(path)
    else:
        gt = np.array(cv2.imread(path, -1))
    gt = gt.astype(np.float32) / factor
    gt[gt == 0] = -1.0
    return _crop(gt, cfg)


def load_rgb(path, cfg):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _crop(img, cfg)


def load_pred_disp(path, target_hw, invert):
    pred = np.load(path).astype(np.float32)
    if invert:  # metric depth -> disparity, matching the L2-run eval prep
        pred = 1.0 / np.clip(pred, 1e-3, None)
    if pred.shape[:2] != target_hw:
        pred = cv2.resize(pred, (target_hw[1], target_hw[0]))
    return pred


def align_to_depth(pred_disp, gt, max_depth):
    """Replicate eval_depthcrafter: disparity-space lstsq align -> depth."""
    valid = np.logical_and(gt > 1e-3, gt < max_depth)
    gt_disp = 1.0 / (gt[valid].reshape(-1, 1).astype(np.float64) + 1e-8)
    pd = np.clip(pred_disp, 1e-3, None)
    scale, shift = stable_scale_and_shift(
        pd[valid].reshape(-1, 1).astype(np.float64), gt_disp)
    aligned = np.clip(scale * pd + shift, 1e-3, None)
    depth = np.clip(1.0 / aligned, 1e-3, max_depth)
    return depth, valid


def abs_rel(pred_depth, gt, valid):
    err = np.zeros_like(gt, dtype=np.float32)
    err[valid] = np.abs(pred_depth[valid] - gt[valid]) / gt[valid]
    mean = float(err[valid].mean()) if valid.any() else float("nan")
    return err, mean


def colorize_depth(depth, valid, dvmin, dvmax, cmap="magma"):
    disp = 1.0 / np.clip(depth, 1e-3, None)
    n = np.clip((disp - dvmin) / (dvmax - dvmin + 1e-8), 0, 1)
    rgb = (plt.get_cmap(cmap)(n)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def colorize_err(err, valid, vmax, cmap="turbo"):
    n = np.clip(err / vmax, 0, 1)
    rgb = (plt.get_cmap(cmap)(n)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def pick_frames(n_total, num_frames, frames_arg):
    if frames_arg:
        idx = [int(x) for x in frames_arg.split(",") if x.strip() != ""]
        return [i for i in idx if 0 <= i < n_total]
    num = min(num_frames, n_total)
    if num <= 1:
        return [0]
    return list(np.linspace(0, n_total - 1, num).round().astype(int))


def infer_npy_path(root, dataset, image_rel):
    p = os.path.join(root, dataset, image_rel)
    return p.replace(".jpg", ".npy").replace(".png", ".npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_path", required=True, help="GT/RGB/json root (same as eval --benchmark_path)")
    ap.add_argument("--pred_a", required=True, help="prediction A npy root (same as its eval --infer_path)")
    ap.add_argument("--pred_b", required=True, help="prediction B npy root")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--dataset", default="kitti", choices=list(DATASET_CFG.keys()))
    ap.add_argument("--invert_a", action="store_true", help="invert A npy (metric->disp) before alignment")
    ap.add_argument("--invert_b", action="store_true", help="invert B npy (metric->disp) before alignment")
    ap.add_argument("--sequence", default="", help="sequence key to render (default: first)")
    ap.add_argument("--num_frames", type=int, default=4)
    ap.add_argument("--frames", default="", help="explicit comma-separated frame indices, e.g. 0,10,20")
    ap.add_argument("--err_vmax", type=float, default=0.3, help="AbsRel colormap upper bound")
    ap.add_argument("--out", default="compare.png")
    args = ap.parse_args()

    cfg = DATASET_CFG[args.dataset]
    json_file = os.path.join(args.benchmark_path, args.dataset, f"{args.dataset}_video.json")
    root_path = os.path.join(args.benchmark_path, args.dataset)
    with open(json_file, "r") as fs:
        json_data = json.load(fs)[args.dataset]

    # Locate the requested sequence (or the first one).
    chosen = None
    for data in json_data:
        for key, value in data.items():
            if not args.sequence or key == args.sequence:
                chosen = (key, value)
                break
        if chosen:
            break
    if chosen is None:
        raise SystemExit(f"sequence '{args.sequence}' not found in {json_file}")
    seq_key, frames = chosen

    idxs = pick_frames(len(frames), args.num_frames, args.frames)
    print(f"[viz] dataset={args.dataset} sequence={seq_key} frames={idxs}")

    n = len(idxs)
    fig, axes = plt.subplots(n, 6, figsize=(6 * 3.0, n * 1.9))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["RGB", "GT", f"{args.label_a} depth", f"{args.label_b} depth",
                  f"{args.label_a} AbsRel", f"{args.label_b} AbsRel"]

    mean_a_all, mean_b_all = [], []
    for row, fi in enumerate(idxs):
        item = frames[fi]
        rgb = load_rgb(os.path.join(root_path, item["image"]), cfg)
        gt = load_gt(os.path.join(root_path, item["gt_depth"]), item["factor"], cfg)
        pa = load_pred_disp(infer_npy_path(args.pred_a, args.dataset, item["image"]), gt.shape, args.invert_a)
        pb = load_pred_disp(infer_npy_path(args.pred_b, args.dataset, item["image"]), gt.shape, args.invert_b)

        depth_a, valid = align_to_depth(pa, gt, cfg["max_depth"])
        depth_b, _ = align_to_depth(pb, gt, cfg["max_depth"])
        err_a, ma = abs_rel(depth_a, gt, valid)
        err_b, mb = abs_rel(depth_b, gt, valid)
        mean_a_all.append(ma)
        mean_b_all.append(mb)

        # Shared disparity range from GT (5/95 pct) so all depth panels compare.
        gt_disp = 1.0 / np.clip(gt[valid], 1e-3, None)
        dvmin, dvmax = np.percentile(gt_disp, [5, 95])

        panels = [
            rgb,
            colorize_depth(gt, valid, dvmin, dvmax),
            colorize_depth(depth_a, valid, dvmin, dvmax),
            colorize_depth(depth_b, valid, dvmin, dvmax),
            colorize_err(err_a, valid, args.err_vmax),
            colorize_err(err_b, valid, args.err_vmax),
        ]
        sub = ["", "", f"AbsRel={ma:.3f}", f"AbsRel={mb:.3f}", "", ""]
        for col, (img, s) in enumerate(zip(panels, sub)):
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)
            if s:
                ax.set_xlabel(s, fontsize=8)
        axes[row, 0].set_ylabel(f"frame {fi}", fontsize=8)

    fig.suptitle(
        f"{args.dataset} · {seq_key} · {args.label_a} mean AbsRel={np.mean(mean_a_all):.3f} "
        f"vs {args.label_b}={np.mean(mean_b_all):.3f}  (AbsRel map: 0=blue .. {args.err_vmax}=red)",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150)
    print(f"[viz] saved {args.out}")
    print(f"[viz] {args.label_a} mean AbsRel over shown frames = {np.mean(mean_a_all):.4f}")
    print(f"[viz] {args.label_b} mean AbsRel over shown frames = {np.mean(mean_b_all):.4f}")


if __name__ == "__main__":
    main()
