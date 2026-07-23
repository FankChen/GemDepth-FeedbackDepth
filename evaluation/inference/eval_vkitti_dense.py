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


def align_to_depth(pred, gt, valid, invert, max_depth, space="disparity"):
    """Affine-align a prediction to the dense GT and return metric depth.

    `invert` describes the RAW model output (True = metric depth for A/B/D, False = disparity for
    C/temporal); `space` chooses WHERE the scale+shift is solved:
      - "disparity" (default, matches the KITTI/MiDaS eval protocol): align in inverse-depth space.
      - "depth": align in metric-depth space (fair sanity check for the metric-L2 models A/B/D).
    The two are orthogonal, so any (output, protocol) combination is handled consistently.
    """
    p = pred.astype(np.float32)
    if p.shape != gt.shape:
        p = cv2.resize(p, (gt.shape[1], gt.shape[0]))
    p = np.clip(p, 1e-3, None)
    if invert:                                   # A/B/D output METRIC depth
        pred_depth_raw, pred_disp_raw = p, 1.0 / p
    else:                                        # C/temporal output DISPARITY (inverse depth)
        pred_disp_raw, pred_depth_raw = p, 1.0 / p
    if space == "depth":                         # metric-space affine align (fair for A/B/D)
        s, t = stable_scale_and_shift(pred_depth_raw[valid].reshape(-1, 1).astype(np.float64),
                                      gt[valid].reshape(-1, 1).astype(np.float64))
        depth = s * pred_depth_raw + t
    else:                                        # disparity-space align (default, KITTI protocol)
        gt_disp = 1.0 / (gt[valid].reshape(-1, 1).astype(np.float64) + 1e-8)
        s, t = stable_scale_and_shift(pred_disp_raw[valid].reshape(-1, 1).astype(np.float64), gt_disp)
        depth = 1.0 / np.clip(s * pred_disp_raw + t, 1e-6, None)
    return np.clip(depth, 1e-3, max_depth)


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


def load_model(config, ckpt, device):
    cfg = OmegaConf.load(config)
    model = build_gemdepth_from_config(cfg, load_backbone_pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=True)
    model = model.to(device).eval()
    clip_len = resolve_inference_clip_len(cfg)
    seq_len = int(OmegaConf.select(cfg, "dataset.train.seq_len", default=4))
    return model, clip_len, seq_len


def eval_model(model, seqs, invert, max_depth, input_size, clip_len, device, space="disparity"):
    """Run one model over all held-out sequences.

    Returns (absrel_list, rmse_list, d1_list, viz) where viz maps
    seq_name -> ((mid_i, gt, pred_depth, valid, absrel), rgb_paths). The viz frame is the middle
    valid frame of each sequence; because `valid` depends only on the GT, the same seq_name selects
    the SAME frame across models -> two models are directly comparable per-frame.
    """
    absrel, rmse, d1, viz = [], [], [], {}
    for name, rgb_paths, dep_paths in seqs:
        videos = np.stack([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in rgb_paths], 0)
        depths, _ = infer_video_with_protocol(
            model, videos, 1, input_size=input_size, device=device, fp32=True,
            clip_len=clip_len, dataset="vkitti", sequence=name)
        seq_frames = []
        for i in range(len(rgb_paths)):
            gt, valid = load_gt(dep_paths[i], max_depth)
            if not valid.any():
                continue
            pred_depth = align_to_depth(depths[i], gt, valid, invert, max_depth, space)
            a, r, d = metrics(pred_depth, gt, valid)
            absrel.append(a); rmse.append(r); d1.append(d)
            seq_frames.append((i, gt, pred_depth, valid, a))
        if seq_frames:
            viz[name] = (seq_frames[len(seq_frames) // 2], rgb_paths)
    return absrel, rmse, d1, viz


def _summary(tag, absrel, rmse, d1, nseq):
    print(f"[vkitti DENSE {tag}] AbsRel={np.mean(absrel):.4f}  RMSE={np.mean(rmse):.4f}  "
          f"delta1={np.mean(d1):.4f}   (over {len(absrel)} frames, {nseq} seqs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vkitti_root",
                    default=os.environ.get("VKITTI_ROOT", "/mnt/workspace/vkitti/vkitti/"))
    ap.add_argument("--invert", action="store_true",
                    help="metric output (A/B/D): take 1/D before alignment; omit for video/temporal")
    ap.add_argument("--align_space", choices=["disparity", "depth"], default="disparity",
                    help="scale+shift space: 'disparity' = KITTI/MiDaS protocol (default); "
                         "'depth' = metric-space align, fair sanity check for the L2 models A/B/D")
    # optional second model -> C-vs-temporal style DENSE comparison figure
    ap.add_argument("--config2", default=None, help="second model config -> dense comparison figure")
    ap.add_argument("--ckpt2", default=None)
    ap.add_argument("--invert2", action="store_true", help="invert (1/D) for the second model")
    ap.add_argument("--label1", default="model1")
    ap.add_argument("--label2", default="model2")
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--limit_seqs", type=int, default=8, help="cap #held-out sequences for speed")
    ap.add_argument("--viz_n", type=int, default=6, help="#frames (one per sequence) in the montage")
    ap.add_argument("--out_viz", default="runlogs/viz/vkitti_dense.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model1, clip_len1, seq_len = load_model(args.config, args.ckpt, device)
    seqs = list_vkitti_heldout(args.vkitti_root, seq_len, limit_seqs=args.limit_seqs)
    if not seqs:
        raise SystemExit(f"[vkitti] no held-out sequences under {args.vkitti_root} "
                         f"(expected <Scene>/<variation>/frames/rgb/Camera_0/...)")
    print(f"[vkitti] {len(seqs)} held-out sequences from {args.vkitti_root}")

    a1, r1, d1, viz1 = eval_model(model1, seqs, args.invert, args.max_depth,
                                  args.input_size, clip_len1, device, args.align_space)
    os.makedirs(os.path.dirname(args.out_viz) or ".", exist_ok=True)

    # ---------------- single-model mode ----------------
    if not args.config2:
        _summary(args.label1, a1, r1, d1, len(seqs))
        rows = list(viz1.items())[:args.viz_n]
        n = len(rows)
        if n:
            fig, axes = plt.subplots(n, 4, figsize=(4 * 3.2, n * 1.9))
            if n == 1:
                axes = axes[None, :]
            titles = ["RGB", "GT (dense)", "pred depth (dense, 未mask)", "AbsRel"]
            for row, (name, ((i, gt, pred, valid, a), rgb_paths)) in enumerate(rows):
                rgb = cv2.cvtColor(cv2.imread(rgb_paths[i]), cv2.COLOR_BGR2RGB)
                panels = [rgb, colorize(gt, valid, show_valid=valid),
                          colorize(pred, valid, show_valid=None), colorize_err(pred, gt, valid)]
                for col, img in enumerate(panels):
                    ax = axes[row, col]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
                    if row == 0:
                        ax.set_title(titles[col], fontsize=9)
                axes[row, 0].set_ylabel(name, fontsize=7)
                axes[row, 2].set_xlabel(f"AbsRel={a:.3f}", fontsize=8)
            fig.suptitle(f"VKITTI2 held-out (DENSE) · {args.label1} mean AbsRel={np.mean(a1):.4f}",
                         fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(args.out_viz, dpi=150)
            print(f"[vkitti] dense viz -> {args.out_viz}")
        return

    # ---------------- two-model comparison (e.g. C vs temporal) ----------------
    model2, clip_len2, _ = load_model(args.config2, args.ckpt2, device)
    a2, r2, d2, viz2 = eval_model(model2, seqs, args.invert2, args.max_depth,
                                  args.input_size, clip_len2, device, args.align_space)
    _summary(args.label1, a1, r1, d1, len(seqs))
    _summary(args.label2, a2, r2, d2, len(seqs))

    names = [nm for nm in viz1 if nm in viz2][:args.viz_n]
    n = len(names)
    if n:
        fig, axes = plt.subplots(n, 6, figsize=(6 * 3.0, n * 1.9))
        if n == 1:
            axes = axes[None, :]
        titles = ["RGB", "GT (dense)", args.label1, args.label2,
                  f"AbsRel {args.label1}", f"AbsRel {args.label2}"]
        for row, name in enumerate(names):
            (i, gt, pred1, valid, aa1), rgb_paths = viz1[name]
            (_, _, pred2, _, aa2), _ = viz2[name]
            rgb = cv2.cvtColor(cv2.imread(rgb_paths[i]), cv2.COLOR_BGR2RGB)
            panels = [rgb,
                      colorize(gt, valid, show_valid=valid),
                      colorize(pred1, valid, show_valid=None),   # both preds UNMASKED -> prove dense
                      colorize(pred2, valid, show_valid=None),
                      colorize_err(pred1, gt, valid),
                      colorize_err(pred2, gt, valid)]
            for col, img in enumerate(panels):
                ax = axes[row, col]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(titles[col], fontsize=9)
            axes[row, 0].set_ylabel(name, fontsize=7)
            axes[row, 2].set_xlabel(f"AbsRel={aa1:.3f}", fontsize=8)
            axes[row, 3].set_xlabel(f"AbsRel={aa2:.3f}", fontsize=8)
        fig.suptitle(f"VKITTI2 held-out (DENSE) · {args.label1} {np.mean(a1):.4f}  vs  "
                     f"{args.label2} {np.mean(a2):.4f}  (mean AbsRel)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(args.out_viz, dpi=150)
        print(f"[vkitti] dense comparison viz -> {args.out_viz}")


if __name__ == "__main__":
    main()
