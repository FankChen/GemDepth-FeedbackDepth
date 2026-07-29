# PointOdyssey loader (long synthetic videos, ~159 sequences).
#
# On-disk layout (verified on Bosch):
#   <root>/<split>/<seq>/rgbs/rgb_<frame:05d>.jpg
#   <root>/<split>/<seq>/depths/depth_<frame:05d>.png
#   <root>/<split>/<seq>/anno.npz   (intrinsics (N,3,3), extrinsics (N,4,4), trajs...)
#
# Depth: 16-bit PNG, metres = raw / 65535 * 1000  (verified by re-projecting the
#        3D trajectories in anno.npz: cam-space z matched raw/65535*1000).
# anno.npz: per-frame ``intrinsics`` (3x3) and ``extrinsics`` (4x4 world->camera,
#           used directly -- E @ X_world gives the correct camera-space point).
# Indoor-leaning scenes -> indoor MAX_DEPTH clamp.

import os

import cv2
import numpy as np

from dataset.loaders.base import BaseLoader, imread_rgb01, make_windows

_DEPTH_SCALE = 1000.0 / 65535.0  # uint16 -> metres


class PointOdysseyLoader(BaseLoader):
    LABEL = "pointodyssey"
    MAX_DEPTH = 80.0
    RATIO = 1  # already frame-rich (long videos)

    def matches(self, data_dir):
        d = data_dir.replace("-", "_").lower()
        return "point_odyssey" in d or "pointodyssey" in d

    def build_sequences(self, data_dir, mode, seq_len):
        clips = []
        for seq_dir in self._find_seq_dirs(data_dir):
            rgb_dir = os.path.join(seq_dir, "rgbs")
            dep_dir = os.path.join(seq_dir, "depths")
            anno = os.path.join(seq_dir, "anno.npz")
            names = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".jpg"))
            if not names:
                continue
            d = np.load(anno)  # npz: only the accessed arrays are decompressed
            Kint = d["intrinsics"]      # (N, 3, 3)
            Ext = d["extrinsics"]       # (N, 4, 4) world->cam
            n = min(len(names), len(Kint), len(Ext))
            refs = []
            for i in range(n):
                stem = os.path.splitext(names[i])[0]           # rgb_00000
                fidx = stem.split("_")[-1]                     # 00000
                dp = os.path.join(dep_dir, f"depth_{fidx}.png")
                if not os.path.isfile(dp):
                    continue
                refs.append(dict(
                    image=os.path.join(rgb_dir, names[i]),
                    depth=dp,
                    pose=Ext[i].astype(np.float32),
                    K=Kint[i].astype(np.float32)))
            for s in make_windows(len(refs), seq_len, mode):
                clips.append((self.LABEL, refs[s:s + seq_len]))
        return clips

    @staticmethod
    def _find_seq_dirs(data_dir):
        """Every directory holding both ``anno.npz`` and an ``rgbs`` folder,
        so the same code works whether data_dir points at the dataset root,
        a split (train/val/test), or a single sequence."""
        seqs = []
        for root, dirs, files in os.walk(data_dir):
            if "anno.npz" in files and os.path.isdir(os.path.join(root, "rgbs")):
                seqs.append(root)
                dirs[:] = []  # don't descend into a sequence's subfolders
        return sorted(seqs)

    def load_frame(self, ref):
        image = imread_rgb01(ref["image"])
        raw = cv2.imread(ref["depth"], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = (raw * _DEPTH_SCALE)[..., None]
        mask = (depth > 0) & (depth < self.MAX_DEPTH)
        depth[~mask] = 0.0
        return image, depth, mask, ref["pose"], ref["K"]
