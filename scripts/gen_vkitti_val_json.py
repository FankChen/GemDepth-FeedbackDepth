"""Generate a VKITTI 2.0.3 held-out (val) json for eval, in the kitti_video.json format.

Each Scene/variation sequence contributes its LAST ``val_ratio`` fraction of frames as one
eval clip (held-out): dataset_mix trains on the first 90%, so val = last 10% — no train/val
leakage. Output:

    {"vkitti": [ {"<Scene>/<variation>": [ {"image":.., "gt_depth":.., "factor":100.0}, ... ]}, ... ]}

Paths are relative to ``--vkitti-root`` (so root_path/json must sit at the VKITTI data root, e.g.
via a symlink ``gemdepth_eval/vkitti -> vkitti2/vkitti``). VKITTI depth is uint16 centimetres,
so meters = value / 100  ->  factor = 100.

Usage:
    python scripts/gen_vkitti_val_json.py \
        --vkitti-root /mnt/workspace/gemdepth_eval/vkitti \
        --out         /mnt/workspace/gemdepth_eval/vkitti/vkitti_video.json \
        --val-ratio 0.1 --max-frames 32
"""
import os
import glob
import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vkitti-root', required=True, help='VKITTI data root (contains Scene01..)')
    ap.add_argument('--out', required=True, help='output json path')
    ap.add_argument('--val-ratio', type=float, default=0.1, help='held-out fraction (last N%) per seq')
    ap.add_argument('--max-frames', type=int, default=32,
                    help='cap frames per seq (0 = all held-out frames); keeps eval fast')
    args = ap.parse_args()

    root = args.vkitti_root.rstrip('/')
    suffix = os.path.join('frames', 'rgb', 'Camera_0')
    rgb_dirs = sorted(glob.glob(os.path.join(root, '**', suffix), recursive=True))

    entries = []
    total_frames = 0
    for rgb_dir in rgb_dirs:
        if not rgb_dir.endswith(suffix):
            continue
        base = rgb_dir[:-len(suffix)].rstrip(os.sep)              # <root>/<Scene>/<variation>
        variation = os.path.basename(base)
        scene = os.path.basename(os.path.dirname(base))
        seq_key = f"{scene}/{variation}"
        depth_dir = os.path.join(base, 'frames', 'depth', 'Camera_0')
        if not os.path.isdir(depth_dir):
            continue

        rgb_by = {}
        for fn in os.listdir(rgb_dir):
            if fn.endswith('.jpg') or fn.endswith('.png'):
                fr = int(os.path.splitext(fn)[0].split('_')[-1])
                rgb_by[fr] = fn
        dep_by = {}
        for fn in os.listdir(depth_dir):
            if fn.endswith('.png'):
                fr = int(os.path.splitext(fn)[0].split('_')[-1])
                dep_by[fr] = fn

        frames = sorted(set(rgb_by) & set(dep_by))
        if len(frames) < 4:
            continue
        n = len(frames)
        start = int(round(n * (1.0 - args.val_ratio)))
        val_frames = frames[start:]
        if args.max_frames and len(val_frames) > args.max_frames:
            val_frames = val_frames[:args.max_frames]

        frame_list = []
        for fr in val_frames:
            img_rel = os.path.relpath(os.path.join(rgb_dir, rgb_by[fr]), root)
            dep_rel = os.path.relpath(os.path.join(depth_dir, dep_by[fr]), root)
            frame_list.append({'image': img_rel, 'gt_depth': dep_rel, 'factor': 100.0})
        if frame_list:
            entries.append({seq_key: frame_list})
            total_frames += len(frame_list)

    out = {'vkitti': entries}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"[gen] {len(entries)} seqs, {total_frames} frames -> {args.out}")


if __name__ == '__main__':
    main()
