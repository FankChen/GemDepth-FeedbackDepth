"""Visualize the coarse-to-fine multi-scale refine head: per-scale depth pyramid + delta_Z per level.

Reads the eval-only `viz_cache` that DPTHeadMultiScaleRefine / DPTHeadMultiScaleRefineConvNeXt stash
on every forward, so we can see EACH level's depth (coarse -> fine) and the delta_Z residual it adds
on top of the previous (upsampled) level, plus per-level delta statistics -- a direct degeneration
check: if the deep levels' delta std ~ 0, those levels aren't actually refining anything.

    python evaluation/inference/eval_vkitti_multilevel.py \
        --config config/scratch/dinov3_convnext/scratch_ed_dinov3convnext_ms_C_native_video.yaml \
        --ckpt   checkpoint/scratch_ed_dinov3convnext_ms_C_native_video/final_model.pth \
        --out_viz runlogs/viz/vkitti_multilevel_convnext_C.png
"""
import argparse
import os
import sys

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.append(_p)
from protocol import infer_video_with_protocol                    # noqa: E402
from eval_vkitti_dense import list_vkitti_heldout, load_model     # noqa: E402


def _to2d(t):
    return t.squeeze().float().cpu().numpy()


def colorize_depth(x, lo, hi, cmap="magma"):
    n = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return (plt.get_cmap(cmap)(n)[..., :3] * 255).astype(np.uint8)


def colorize_delta(x, vmax, cmap="RdBu_r"):
    n = np.clip(x / (vmax + 1e-8), -1, 1) * 0.5 + 0.5   # 0 -> white, + -> red, - -> blue
    return (plt.get_cmap(cmap)(n)[..., :3] * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vkitti_root",
                    default=os.environ.get("VKITTI_ROOT", "/mnt/workspace/vkitti/vkitti/"))
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--seq_idx", type=int, default=0, help="which held-out sequence to use")
    ap.add_argument("--out_viz", default="runlogs/viz/vkitti_multilevel.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, clip_len, seq_len = load_model(args.config, args.ckpt, device)
    head = next((m for m in model.modules() if hasattr(m, "delta_heads")), None)
    if head is None:
        raise SystemExit("no multi-scale refine head (delta_heads) found -> is this a multiscale config?")

    seqs = list_vkitti_heldout(args.vkitti_root, seq_len, limit_seqs=args.seq_idx + 1)
    if not seqs:
        raise SystemExit(f"no vkitti held-out sequences under {args.vkitti_root}")
    name, rgb_paths, dep_paths = seqs[args.seq_idx]
    frames = rgb_paths[:clip_len]                       # one clip -> one clean forward window
    videos = np.stack([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in frames], 0)

    head.viz_cache = None
    infer_video_with_protocol(model, videos, 1, input_size=args.input_size, device=device,
                              fp32=True, clip_len=clip_len, dataset="vkitti", sequence=name)
    cache = getattr(head, "viz_cache", None)
    if not cache:
        raise SystemExit("viz_cache empty -> forward didn't populate it (check model.eval() / head type)")

    ml = cache["multilevel_native"]     # list of N: [B*T, 1, h_i, w_i], coarse -> fine
    dl = cache["deltas"]
    n_scales = len(ml)
    fi = ml[0].shape[0] // 2            # middle frame of the clip

    # ---- per-scale delta statistics = degeneration check ----
    print(f"[multilevel] {name}  frame {fi}/{ml[0].shape[0]}  ({n_scales} scales, coarse->fine)")
    for i in range(n_scales):
        d = _to2d(ml[i][fi]); dz = _to2d(dl[i][fi])
        print(f"  scale {i}: depth {d.shape[0]:>3}x{d.shape[1]:<3}  "
              f"delta std={dz.std():.4f} mean={dz.mean():+.4f} "
              f"[min {dz.min():+.3f}, max {dz.max():+.3f}]   "
              f"depth[min/med/max]={d.min():.2f}/{np.median(d):.2f}/{d.max():.2f}")

    # symmetric delta color range shared across scales (so magnitudes are comparable)
    dmax = max(np.percentile(np.abs(_to2d(dl[i][fi])), 99) for i in range(n_scales))

    rgb = cv2.cvtColor(cv2.imread(frames[fi]), cv2.COLOR_BGR2RGB)
    cols = 1 + n_scales
    fig, axes = plt.subplots(2, cols, figsize=(cols * 3.0, 2 * 2.6))
    axes[0, 0].imshow(rgb); axes[0, 0].set_title("RGB", fontsize=9)
    for i in range(n_scales):
        d = _to2d(ml[i][fi]); dz = _to2d(dl[i][fi])
        lo_i, hi_i = np.percentile(d, [5, 95])          # per-scale range -> structure is legible
        axes[0, 1 + i].imshow(colorize_depth(d, lo_i, hi_i))
        axes[0, 1 + i].set_title(f"depth s{i} ({d.shape[0]}x{d.shape[1]})", fontsize=8)
        axes[1, 1 + i].imshow(colorize_delta(dz, dmax))
        axes[1, 1 + i].set_title(f"delta s{i}  std={dz.std():.3f}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    axes[1, 0].axis("off")
    axes[0, 0].set_ylabel("depth pyramid (coarse->fine)", fontsize=9)
    fig.suptitle(f"multi-level refine · {name} · per-level depth + delta_Z "
                 f"(red=+ / blue=-, |delta|<= {dmax:.2f})", fontsize=10)
    os.makedirs(os.path.dirname(args.out_viz) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out_viz, dpi=150)
    print(f"[multilevel] viz -> {args.out_viz}")


if __name__ == "__main__":
    main()
