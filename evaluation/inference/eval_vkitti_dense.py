"""VKITTI2 held-out DENSE eval + dense visualization (in-domain, dense GT).

Reuses the EXACT tested inference path (model.infer_video_depth via the shared protocol) and the
same disparity-space alignment as the KITTI eval (stable_scale_and_shift), so the vkitti numbers are
directly comparable to the KITTI Eigen numbers. VKITTI2 depth GT is dense (16-bit PNG in cm; /100 ->
meters; 65535 = sky -> invalid), so BOTH the metrics and the figure are dense.

Held-out split matches the training dataloader (mode != 'train' -> last ~10% frames per sequence).

Run on the node that has the vkitti data + checkpoints:
    python evaluation/inference/eval_vkitti_dense.py \
        --config config/scratch_ed_dinov3vits_ms_C_native_video.yaml \
        --ckpt   checkpoint/scratch_ed_dinov3vits_ms_C_native_video/final_model.pth \
        --out_viz runlogs/viz/vkitti_vits_C.png
    #   add --invert for the METRIC-output runs (A / B / D); omit for video / temporal (C).
"""
import argparse
import glob
import os
import sys

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))        # evaluation/inference
_EVAL = os.path.join(os.path.dirname(_HERE), "eval")       # evaluation/eval
_ROOT = os.path.dirname(os.path.dirname(_HERE))            # repo root
for _p in (_ROOT, _HERE, _EVAL):
    if _p not in sys.path:
        sys.path.append(_p)
from model.factory import build_gemdepth_from_config          # noqa: E402
from protocol import infer_video_with_protocol, resolve_inference_clip_len  # noqa: E402
from alignment import stable_scale_and_shift                  # noqa: E402


def list_vkitti_heldout(root, seq_len, holdout=0.1, limit_seqs=None):
    """[(name, [rgb_paths], [depth_paths]), ...] for the last `holdout` frames of each vkitti2
    sequence -- same glob + val split as dataset/dataset_mix.py (mode != 'train')."""
    rgb_suffix = os.path.join("frames", "rgb", "Camera_0")
    rgb_dirs = sorted(glob.glob(os.path.join(root, "**", rgb_suffix), recursive=True))
    out = []
    for rgb_dir in rgb_dirs:
        if not rgb_dir.endswith(rgb_suffix):
            continue
        base = rgb_dir[:-len(rgb_suffix)]
        depth_dir = os.path.join(base, "frames", "depth", "Camera_0")
        if not os.path.isdir(depth_dir):
            continue
        trimmed = base.rstrip(os.sep)
        name = f"{os.path.basename(os.path.dirname(trimmed))}_{os.path.basename(trimmed)}"
        rgb_by = {}
        for fn in os.listdir(rgb_dir):
            if fn.endswith(".jpg") or fn.endswith(".png"):
                rgb_by[int(os.path.splitext(fn)[0].split("_")[-1])] = os.path.join(rgb_dir, fn)
        dep_by = {}
        for fn in os.listdir(depth_dir):
            if fn.endswith(".png"):
                dep_by[int(os.path.splitext(fn)[0].split("_")[-1])] = os.path.join(depth_dir, fn)
        frames = sorted(set(rgb_by) & set(dep_by))
        if len(frames) < seq_len:
            continue
        seq_num = len(frames) - seq_len + 1
        start = round(seq_num * (1.0 - holdout)) + 1          # dataloader val split
        held = frames[start:]
        n = (len(held) // seq_len) * seq_len                  # whole non-overlapping windows
        if n < seq_len:
            continue
        held = held[:n]
        out.append((name, [rgb_by[f] for f in held], [dep_by[f] for f in held]))
    return out[:limit_seqs] if limit_seqs else out


def load_gt(path, max_depth):
    d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 100.0  # cm -> m
    d[d >= 655] = 0.0                                          # VKITTI2 sky sentinel -> invalid
    valid = (d > 1e-3) & (d < max_depth)
    return d, valid


def align_to_depth(pred, gt, valid, invert, max_depth):
    p = pred.astype(np.float32)
    if p.shape != gt.shape:
        p = cv2.resize(p, (gt.shape[1], gt.shape[0]))
    pd = 1.0 / np.clip(p, 1e-3, None) if invert else p        # metric -> disparity for A/B/D
    pd = np.clip(pd, 1e-3, None)
    gt_disp = 1.0 / (gt[valid].reshape(-1, 1).astype(np.float64) + 1e-8)
    scale, shift = stable_scale_and_shift(pd[valid].reshape(-1, 1).astype(np.float64), gt_disp)
    aligned = np.clip(scale * pd + shift, 1e-3, None)
    return np.clip(1.0 / aligned, 1e-3, max_depth)


def metrics(pred_depth, gt, valid):
    p, g = pred_depth[valid], gt[valid]
    absrel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    d1 = float(np.mean(np.maximum(p / g, g / p) < 1.25))
    return absrel, rmse, d1


def colorize(depth, range_valid, show_valid=None, cmap="magma"):
    disp = 1.0 / np.clip(depth, 1e-3, None)
    v = disp[range_valid]
    lo, hi = (np.percentile(v, [5, 95]) if v.size else (0.0, 1.0))
    n = np.clip((disp - lo) / (hi - lo + 1e-8), 0, 1)
    rgb = (plt.get_cmap(cmap)(n)[..., :3] * 255).astype(np.uint8)
    if show_valid is not None:
        rgb[~show_valid] = 0
    return rgb


def colorize_err(pred, gt, valid, vmax=0.3):
    err = np.zeros_like(gt)
    err[valid] = np.abs(pred[valid] - gt[valid]) / gt[valid]
    rgb = (plt.get_cmap("turbo")(np.clip(err / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vkitti_root",
                    default=os.environ.get("VKITTI_ROOT", "/mnt/workspace/vkitti/vkitti/"))
    ap.add_argument("--invert", action="store_true",
                    help="metric output (A/B/D): take 1/D before alignment; omit for video/temporal")
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--limit_seqs", type=int, default=8, help="cap #held-out sequences for speed")
    ap.add_argument("--viz_n", type=int, default=6, help="#frames (one per sequence) in the montage")
    ap.add_argument("--out_viz", default="runlogs/viz/vkitti_dense.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.load(args.config)
    model = build_gemdepth_from_config(cfg, load_backbone_pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False), strict=True)
    model = model.to(device).eval()
    clip_len = resolve_inference_clip_len(cfg)
    seq_len = int(OmegaConf.select(cfg, "dataset.train.seq_len", default=4))

    seqs = list_vkitti_heldout(args.vkitti_root, seq_len, limit_seqs=args.limit_seqs)
    if not seqs:
        raise SystemExit(f"[vkitti] no held-out sequences under {args.vkitti_root} "
                         f"(expected <Scene>/<variation>/frames/rgb/Camera_0/...)")
    print(f"[vkitti] {len(seqs)} held-out sequences from {args.vkitti_root}  (invert={args.invert})")

    all_absrel, all_rmse, all_d1, viz = [], [], [], []
    for name, rgb_paths, dep_paths in seqs:
        videos = np.stack([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in rgb_paths], 0)
        depths, _ = infer_video_with_protocol(
            model, videos, 1, input_size=args.input_size, device=device, fp32=True,
            clip_len=clip_len, dataset="vkitti", sequence=name)
        seq_frames = []
        for i in range(len(rgb_paths)):
            gt, valid = load_gt(dep_paths[i], args.max_depth)
            if not valid.any():
                continue
            pred_depth = align_to_depth(depths[i], gt, valid, args.invert, args.max_depth)
            a, r, d = metrics(pred_depth, gt, valid)
            all_absrel.append(a); all_rmse.append(r); all_d1.append(d)
            seq_frames.append((i, gt, pred_depth, valid, a))
        if seq_frames and len(viz) < args.viz_n:
            i, gt, pred, valid, a = seq_frames[len(seq_frames) // 2]
            rgb = cv2.cvtColor(cv2.imread(rgb_paths[i]), cv2.COLOR_BGR2RGB)
            viz.append((name, rgb, gt, pred, valid, a))

    print(f"[vkitti DENSE] AbsRel={np.mean(all_absrel):.4f}  RMSE={np.mean(all_rmse):.4f}  "
          f"delta1={np.mean(all_d1):.4f}   (over {len(all_absrel)} frames, {len(seqs)} seqs)")

    n = len(viz)
    if n:
        fig, axes = plt.subplots(n, 4, figsize=(4 * 3.2, n * 1.9))
        if n == 1:
            axes = axes[None, :]
        titles = ["RGB", "GT (dense)", "pred depth (dense, 未mask)", "AbsRel"]
        for row, (name, rgb, gt, pred, valid, a) in enumerate(viz):
            panels = [
                rgb,
                colorize(gt, valid, show_valid=valid),          # GT: dense minus sky
                colorize(pred, valid, show_valid=None),         # pred: EVERY pixel -> proves dense
                colorize_err(pred, gt, valid),
            ]
            for col, img in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(titles[col], fontsize=9)
            axes[row, 0].set_ylabel(name, fontsize=7)
            axes[row, 2].set_xlabel(f"AbsRel={a:.3f}", fontsize=8)
        fig.suptitle(f"VKITTI2 held-out (DENSE, in-domain) · mean AbsRel={np.mean(all_absrel):.4f}",
                     fontsize=10)
        os.makedirs(os.path.dirname(args.out_viz) or ".", exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(args.out_viz, dpi=150)
        print(f"[vkitti] dense viz -> {args.out_viz}")


if __name__ == "__main__":
    main()
