#!/usr/bin/env python3
"""Visualise HOG / warp / error-map signals to verify correctness BEFORE training.

Loads one real VKITTI sample, builds each warp signal (rgb / hog), runs the
differentiable reprojection with **GT depth + GT pose**, and dumps PNG panels so you
can eyeball three things:

  1. WARP GEOMETRY (most important): the GT-depth+pose warped neighbour should
     align with the target frame -> small residual. If it is clearly misaligned,
     the warp / pose convention / intrinsics scaling has a bug.
  2. HOG DESCRIPTOR: orientation-coloured map should respond on edges/texture.
  3. ERROR MAP + VALID MASK per signal.

This uses the built-in ``capture`` hook of ``signal_error_map`` (no model needed).

Usage (Aliyun, system python):
  /usr/local/bin/python3 scripts/viz_errormap_signals.py \
      --vkitti_root /mnt/workspace/vkitti2/vkitti \
      --seq_len 3 --sample_idx 0 --out viz_out
"""
import os
import sys
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.dataset_mix import DepthVideoDataset          # noqa: E402
from model.util.warp import signal_error_map               # noqa: E402
from model.util.hog import hog_feature_map                 # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denorm(img):
    """(T,3,H,W) ImageNet-normalised -> (T,3,H,W) in [0,1]."""
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def chw_to_hwc(x):
    x = x.detach().cpu().float()
    if x.dim() == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    return x.squeeze(-1).numpy() if x.dim() == 3 and x.shape[-1] == 1 else x.numpy()


def norm01(x):
    x = x.detach().cpu().float()
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def signal_to_gray(sig_chw):
    """Any (C,H,W) signal -> (H,W) grayscale for display (mean over channels)."""
    return norm01(sig_chw.mean(dim=0)).numpy()


def hog_to_rgb(hog_chw):
    """(nbins,H,W) -> (H,W,3): argmax orientation -> hue, summed magnitude -> value."""
    nbins = hog_chw.shape[0]
    mag = norm01(hog_chw.sum(0))
    ori = hog_chw.argmax(0).float() / max(nbins - 1, 1)
    hsv = torch.stack([ori, torch.ones_like(ori), mag], dim=-1).cpu().numpy()
    return mcolors.hsv_to_rgb(hsv)


def colorize_depth(depth_hw):
    d = norm01(depth_hw).numpy()
    return plt.cm.magma(d)[..., :3]


def dump_signal_panels(name, cap, rgb01, out):
    """For each offset record, show target vs warped neighbour + residual + valid."""
    is_rgb = (name == 'rgb')
    for rec in cap:
        off = rec['offset']
        # take the first valid target frame in this offset batch
        b, n = 0, 0
        tgt = rec['target'][b, n]      # (C,H,W)
        src = rec['source'][b, n]
        wrp = rec['warped'][b, n]
        err = rec['error'][b, n, 0]    # (H,W)
        val = rec['valid'][b, n, 0]

        if is_rgb:
            tgt_img, src_img, wrp_img = chw_to_hwc(tgt), chw_to_hwc(src), chw_to_hwc(wrp)
            absdiff = (tgt - wrp).abs().mean(0)
        else:
            tgt_img, src_img, wrp_img = signal_to_gray(tgt), signal_to_gray(src), signal_to_gray(wrp)
            absdiff = norm01((tgt - wrp).abs().mean(0))

        panels = [
            (tgt_img, f'target signal ({name})'),
            (src_img, f'neighbour t{off:+d} (source)'),
            (wrp_img, f'warped t{off:+d} -> t  (should match target)'),
            (chw_to_hwc(absdiff) if is_rgb else absdiff, '|target - warped|'),
            (norm01(err).numpy(), 'error map (min-reproj)'),
            (val.numpy(), 'valid mask'),
        ]
        fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
        for ax, (im, title) in zip(axes, panels):
            cmap = None if (im.ndim == 3) else 'gray'
            ax.imshow(im, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        fig.suptitle(f'[{name}] offset t{off:+d}  —  warp geometry check', fontsize=10)
        fig.tight_layout()
        fp = os.path.join(out, f'signal_{name}_off{off:+d}.png')
        fig.savefig(fp, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  wrote {fp}')


def dump_hog_orientation(hog_tchw, rgb01, out):
    """Per-frame HOG orientation-coloured map next to the RGB frame."""
    T = hog_tchw.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6))
    if T == 1:
        axes = axes.reshape(2, 1)
    for t in range(T):
        axes[0, t].imshow(chw_to_hwc(rgb01[t]))
        axes[0, t].set_title(f'frame {t} (rgb)', fontsize=8)
        axes[0, t].axis('off')
        axes[1, t].imshow(hog_to_rgb(hog_tchw[t]))
        axes[1, t].set_title(f'frame {t} (HOG orient)', fontsize=8)
        axes[1, t].axis('off')
    fig.suptitle('HOG descriptor sanity (hue=orientation, brightness=magnitude)', fontsize=10)
    fig.tight_layout()
    fp = os.path.join(out, 'hog_orientation.png')
    fig.savefig(fp, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {fp}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vkitti_root', required=True, help='path containing "vkitti" (2.0.3 layout)')
    ap.add_argument('--seq_len', type=int, default=3)
    ap.add_argument('--sample_idx', type=int, default=0)
    ap.add_argument('--crop_size', type=int, default=518)
    ap.add_argument('--signals', nargs='+', default=['rgb', 'hog'], choices=['rgb', 'hog'])
    ap.add_argument('--out', default='viz_out')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = DepthVideoDataset('train', data_dirs=[args.vkitti_root],
                           crop_size=args.crop_size, seq_len=args.seq_len)
    print(f'dataset samples: {len(ds)}   (using idx {args.sample_idx})')
    s = ds[args.sample_idx]

    image = s['image']                                        # (T,3,H,W) normalised
    depth = s['depth']                                        # (T,1,H,W)
    IntM = torch.as_tensor(np.asarray(s['IntM'])).float()     # (3,3)
    poses = torch.stack([torch.as_tensor(np.asarray(p)).float() for p in s['poses']])  # (T,4,4)
    T = image.shape[0]
    print(f'sample: T={T}  HxW={tuple(image.shape[-2:])}  '
          f'depth[min,max]=[{depth[depth>0].min():.2f},{depth.max():.2f}]  label={s.get("label")}')

    rgb01 = denorm(image)                                     # (T,3,H,W) [0,1]

    B = 1
    imgs_b = rgb01[None]                                      # (1,T,3,H,W)
    depth_b = depth[None]                                     # (1,T,1,H,W)
    K = IntM[None, None].repeat(B, T, 1, 1)                   # (1,T,3,3)
    ext = poses[None]                                         # (1,T,4,4)

    # dump reference: rgb frames + depth
    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6))
    if T == 1:
        axes = axes.reshape(2, 1)
    for t in range(T):
        axes[0, t].imshow(chw_to_hwc(rgb01[t])); axes[0, t].set_title(f'rgb {t}', fontsize=8); axes[0, t].axis('off')
        axes[1, t].imshow(colorize_depth(depth[t, 0])); axes[1, t].set_title(f'GT depth {t}', fontsize=8); axes[1, t].axis('off')
    fig.suptitle('input frames + GT depth', fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'input_frames.png'), dpi=120, bbox_inches='tight'); plt.close(fig)
    print(f'  wrote {args.out}/input_frames.png')

    signal_map = {}
    if 'rgb' in args.signals:
        signal_map['rgb'] = imgs_b
    if 'hog' in args.signals:
        signal_map['hog'] = hog_feature_map(imgs_b, nbins=9)  # (1,T,9,H,W)

    for name, sig in signal_map.items():
        cap = []
        err, valid = signal_error_map(sig, depth_b, K, ext, offsets=(-1, 1), capture=cap, tag=name)
        cov = float(valid.mean())
        print(f'[{name}] error mean(valid)={float((err*valid).sum()/valid.sum().clamp(min=1)):.4f}  '
              f'valid coverage={cov:.3f}  captures={len(cap)}')
        dump_signal_panels(name, cap, rgb01, args.out)

    if 'hog' in signal_map:
        dump_hog_orientation(signal_map['hog'][0], rgb01, args.out)

    print(f'\nDONE -> {args.out}/  (scp back and eyeball; warped should match target)')


if __name__ == '__main__':
    main()
