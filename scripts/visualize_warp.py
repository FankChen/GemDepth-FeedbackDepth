#!/usr/bin/env python3
"""Visualise every inverse-warp used by the error-map DPT heads.

The error-map heads (``dpt_errormap.py`` / ``dpt_errormap_coattn.py``) repeatedly inverse-warp
neighbouring video frames into each target frame and turn the residual into an "error map".
This script renders every one of those warps as an image panel so you can inspect, frame by
frame, exactly what the warp produced.

Two modes
---------
* **GT-geometry** (default, CPU, no checkpoint):
  Load real VKITTI 2 frames (RGB + ground-truth metric depth + ground-truth pose) and call
  ``signal_error_map`` directly. Because the geometry is correct, the warped neighbour should
  align with the target and ``|error|`` should be near zero except at occlusions /
  disocclusions / moving objects. This validates the warp maths independently of any model.

* **In-model** (``--ckpt CKPT``, needs a GPU because of cuRoPE2D):
  Build ``GemDepth`` with the chosen head, enable ``head.capture_warps`` and run
  ``infer_video_depth`` on a short real clip, dumping every warp the decoder performs (per
  stage, per modality) using the model's *predicted* depth and GEM-predicted pose.

For every warp a multi-panel PNG is written into ``--out``:
    target | source | warped | overlay | |error| | valid | depth

Examples
--------
    # CPU, ground-truth geometry (recommended first run)
    python scripts/visualize_warp.py --frames 6 --start 0 --out output_viz/gt

    # In-model warps with a trained checkpoint (run on a GPU node)
    python scripts/visualize_warp.py --ckpt runs/em_rgb/final_model.pth \
        --head_type errormap_coattn --error_modalities rgb --frames 8 --out output_viz/em_rgb
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import torch  # noqa: E402

# Keep the shared login node calm.
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from model.util.warp import signal_error_map  # noqa: E402

# VKITTI 2 native intrinsics / resolution.
VK_W, VK_H = 1242, 375
VK_FX, VK_FY, VK_CX, VK_CY = 725.0087, 725.0087, 620.5, 187.0
SKY_DEPTH_M = 655.0  # uint16-cm sky sentinel -> treat >= as invalid

# ImageNet stats used by the model's NormalizeImage transform (for in-model de-norm).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --------------------------------------------------------------------------------------- data
def load_extrinsics(path):
    """Return {frame: 4x4 world->camera float32} for camera 0 of a VKITTI extrinsic.txt."""
    poses = {}
    with open(path) as f:
        for line in f.readlines()[1:]:
            p = line.split()
            if len(p) < 18:
                continue
            frame, cam = int(float(p[0])), int(float(p[1]))
            if cam != 0:
                continue
            poses[frame] = np.array(list(map(float, p[2:18])), dtype=np.float32).reshape(4, 4)
    return poses


def find_sequence(root, scene, variation):
    """Return (scene, variation) that exists, falling back to the first available sequence."""
    if os.path.isdir(os.path.join(root, scene, variation, "frames", "rgb", "Camera_0")):
        return scene, variation
    hits = sorted(glob.glob(os.path.join(root, "*", "*", "frames", "rgb", "Camera_0")))
    if not hits:
        raise FileNotFoundError(f"No VKITTI rgb sequences under {root}")
    parts = hits[0].split(os.sep)
    print(f"[data] {scene}/{variation} not found; using {parts[-5]}/{parts[-4]}")
    return parts[-5], parts[-4]


def load_clip(root, scene, variation, start, count, height):
    """Load ``count`` consecutive VKITTI frames as model-ready tensors + raw display images."""
    rgb_dir = os.path.join(root, scene, variation, "frames", "rgb", "Camera_0")
    dep_dir = os.path.join(root, scene, variation, "frames", "depth", "Camera_0")
    poses = load_extrinsics(os.path.join(root, scene, variation, "extrinsic.txt"))

    raw_rgb, raw_dep, exts = [], [], []
    for i in range(start, start + count):
        rp = os.path.join(rgb_dir, f"rgb_{i:05d}.jpg")
        dp = os.path.join(dep_dir, f"depth_{i:05d}.png")
        bgr = cv2.imread(rp, cv2.IMREAD_COLOR)
        d16 = cv2.imread(dp, cv2.IMREAD_ANYDEPTH)
        if bgr is None or d16 is None or i not in poses:
            raise FileNotFoundError(f"Missing frame {i}: {rp} / {dp} / pose")
        raw_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        dep = d16.astype(np.float32) / 100.0  # cm -> m
        dep[dep >= SKY_DEPTH_M] = 0.0
        raw_dep.append(dep)
        exts.append(poses[i])

    h0, w0 = raw_rgb[0].shape[:2]
    width = int(round(w0 * height / h0))
    rs_rgb = [cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR) for im in raw_rgb]
    rs_dep = [cv2.resize(dp, (width, height), interpolation=cv2.INTER_NEAREST) for dp in raw_dep]

    sx, sy = width / w0, height / h0
    K = np.array([[VK_FX * sx, 0.0, VK_CX * sx],
                  [0.0, VK_FY * sy, VK_CY * sy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    T = count
    sig = torch.from_numpy(np.stack(rs_rgb)).permute(0, 3, 1, 2).float() / 255.0  # T,3,H,W in [0,1]
    dep = torch.from_numpy(np.stack(rs_dep)).unsqueeze(1).float()                 # T,1,H,W
    ext = torch.from_numpy(np.stack(exts)).float()                                # T,4,4
    Kt = torch.from_numpy(K)[None].repeat(T, 1, 1)                                # T,3,3
    disp_rgb = [im.astype(np.float32) / 255.0 for im in rs_rgb]                    # list HWC [0,1]
    return sig[None], dep[None], Kt[None], ext[None], disp_rgb


# ------------------------------------------------------------------------------------ display
def to_disp(chw):
    """Map a (C,H,W) tensor to an RGB image (C==3) or a normalised grayscale map (C!=3)."""
    arr = chw.detach().cpu().numpy()
    if arr.shape[0] == 3:
        return np.clip(arr.transpose(1, 2, 0), 0.0, 1.0)
    g = arr.mean(0)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    return np.clip((g - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def save_records(records, out_dir, disp_rgb, frame_base, denorm_rgb=False):
    """Write one multi-panel PNG per captured warp."""
    os.makedirs(out_dir, exist_ok=True)
    n_saved = 0
    for r in records:
        tag = r["tag"]
        safe_tag = tag.replace("/", "_")
        offset = r["offset"]
        idx_t = r["idx_t"].tolist()
        idx_s = r["idx_s"].tolist()
        tgt = r["target"][0]          # (n,C,H,W)
        wrp = r["warped"][0]
        err = r["error"][0, :, 0]     # (n,H,W)
        val = r["valid"][0, :, 0]
        is_rgb = tgt.shape[1] == 3 and tag.endswith("rgb")

        if denorm_rgb and is_rgb:
            tgt = tgt * IMAGENET_STD + IMAGENET_MEAN
            wrp = wrp * IMAGENET_STD + IMAGENET_MEAN

        depth = r.get("depth", None)  # (1,T,1,h,w) in-model only

        for j in range(tgt.shape[0]):
            ti, si = idx_t[j], idx_s[j]
            t_img = to_disp(tgt[j])
            w_img = to_disp(wrp[j])
            panels = [(t_img, f"target t={frame_base + ti}", "img")]
            if disp_rgb is not None:
                panels.append((disp_rgb[si], f"source t={frame_base + si}", "img"))
            panels.append((w_img, f"warped {frame_base + si}->{frame_base + ti}", "img"))
            if is_rgb:
                panels.append((np.clip(0.5 * t_img + 0.5 * w_img, 0, 1), "overlay t/warped", "img"))
            panels.append((err[j].numpy(), "|error|", "err"))
            panels.append((val[j].numpy(), "valid", "mask"))
            if depth is not None:
                panels.append((depth[0, ti, 0].numpy(), f"depth t={frame_base + ti}", "depth"))

            fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
            if len(panels) == 1:
                axes = [axes]
            for ax, (img, title, kind) in zip(axes, panels):
                if kind == "err":
                    im = ax.imshow(img, cmap="magma")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                elif kind == "mask":
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                elif kind == "depth":
                    im = ax.imshow(img, cmap="turbo")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.imshow(img)
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            fig.suptitle(
                f"{tag}   t={frame_base + ti} <- t={frame_base + si}  (offset {offset:+d})",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fname = f"{safe_tag}_t{frame_base + ti:03d}_from{frame_base + si:03d}.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=120)
            plt.close(fig)
            n_saved += 1
    return n_saved


# --------------------------------------------------------------------------------------- modes
def run_gt_mode(args):
    scene, variation = find_sequence(args.vkitti_root, args.scene, args.variation)
    sig, dep, K, ext, disp_rgb = load_clip(
        args.vkitti_root, scene, variation, args.start, args.frames, args.height
    )
    records = []
    signal_error_map(sig, dep, K, ext, offsets=tuple(args.offsets), capture=records, tag="rgb")
    # Attach the GT depth so each panel also shows what drove the warp.
    for r in records:
        r["depth"] = dep
    n = save_records(records, args.out, disp_rgb, args.start)
    print(f"[GT] {scene}/{variation} frames {args.start}..{args.start + args.frames - 1}: "
          f"{len(records)} warps -> {n} panels in {args.out}")


def run_model_mode(args):
    from model.gemdepth import GemDepth

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] in-model mode needs a GPU (cuRoPE2D); this will likely fail on CPU.")

    scene, variation = find_sequence(args.vkitti_root, args.scene, args.variation)
    rgb_dir = os.path.join(args.vkitti_root, scene, variation, "frames", "rgb", "Camera_0")
    videos = []
    for i in range(args.start, args.start + args.frames):
        bgr = cv2.imread(os.path.join(rgb_dir, f"rgb_{i:05d}.jpg"), cv2.IMREAD_COLOR)
        videos.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    videos = np.stack(videos, axis=0)

    cfg = {"vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
           "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])}[args.encoder]
    model = GemDepth(**cfg, head_type=args.head_type, error_modalities=args.error_modalities)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device).eval()

    model.head.capture_warps = []
    with torch.no_grad():
        model.infer_video_depth(videos, 1, input_size=args.input_size, device=device, fp32=True)
    records = model.head.capture_warps
    n = save_records(records, args.out, None, args.start, denorm_rgb=True)
    print(f"[model] {args.head_type}/{args.error_modalities}: {len(records)} warps -> "
          f"{n} panels in {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vkitti_root", default=os.environ.get("VKITTI_ROOT", "/home/izi2sgh/MYDATA/vkitti"))
    ap.add_argument("--scene", default="Scene01")
    ap.add_argument("--variation", default="clone")
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument("--frames", type=int, default=6, help="number of consecutive frames")
    ap.add_argument("--height", type=int, default=192, help="GT-mode render height (width by aspect)")
    ap.add_argument("--offsets", type=int, nargs="+", default=[-1, 1], help="temporal warp offsets")
    ap.add_argument("--out", default="output_viz/warp")
    # in-model mode
    ap.add_argument("--ckpt", default="", help="checkpoint -> enable in-model mode (GPU)")
    ap.add_argument("--encoder", default="vitl", choices=["vits", "vitl"])
    ap.add_argument("--head_type", default="errormap_coattn",
                    choices=["errormap", "errormap_coattn"])
    ap.add_argument("--error_modalities", default="rgb")
    ap.add_argument("--input_size", type=int, default=518)
    args = ap.parse_args()

    if args.ckpt:
        run_model_mode(args)
    else:
        run_gt_mode(args)


if __name__ == "__main__":
    main()
