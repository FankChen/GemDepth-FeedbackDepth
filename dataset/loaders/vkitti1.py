# VKITTI 1.3.1 loader (monocular driving video, 5 worlds x 10 variations).
#
# On-disk layout (verified on Bosch):
#   <root>/vkitti_1.3.1_rgb/<world>/<variation>/<frame:05d>.png
#   <root>/vkitti_1.3.1_depthgt/<world>/<variation>/<frame:05d>.png
#   <root>/vkitti_1.3.1_extrinsicsgt/<world>_<variation>.txt
#
# Depth: 16-bit PNG, metres = raw/100, sky sentinel 65535 (=655.35 m) -> invalid.
# Intrinsics: fx=fy=725, cx=620.5, cy=187 (same as the VKITTI2 branch).
# Extrinsics file: header row, then per frame
#   "frame r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3 0 0 0 1"  (world->camera).

import os

import cv2
import numpy as np

from dataset.loaders.base import BaseLoader, imread_rgb01, make_windows


class VKitti1Loader(BaseLoader):
    LABEL = "vkitti1"
    MAX_DEPTH = 200.0
    RATIO = 15  # ~21k frames, matched to the existing vkitti (VKITTI2) ratio

    _K = np.array([[725.0, 0.0, 620.5],
                   [0.0, 725.0, 187.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)

    def matches(self, data_dir):
        d = data_dir.lower()
        return "vkitti" in d and "1.3.1" in d

    def build_sequences(self, data_dir, mode, seq_len):
        rgb_root = os.path.join(data_dir, "vkitti_1.3.1_rgb")
        depth_root = os.path.join(data_dir, "vkitti_1.3.1_depthgt")
        extr_root = os.path.join(data_dir, "vkitti_1.3.1_extrinsicsgt")
        if not os.path.isdir(rgb_root):
            return []
        clips = []
        for world in sorted(os.listdir(rgb_root)):
            wdir = os.path.join(rgb_root, world)
            if not os.path.isdir(wdir):
                continue
            for var in sorted(os.listdir(wdir)):
                rgb_dir = os.path.join(rgb_root, world, var)
                depth_dir = os.path.join(depth_root, world, var)
                extr_file = os.path.join(extr_root, f"{world}_{var}.txt")
                if not (os.path.isdir(depth_dir) and os.path.isfile(extr_file)):
                    continue
                poses = self._parse_extrinsics(extr_file)
                frames = sorted(
                    int(os.path.splitext(f)[0])
                    for f in os.listdir(rgb_dir) if f.endswith(".png"))
                refs = []
                for fr in frames:
                    if fr not in poses:
                        continue
                    dp = os.path.join(depth_dir, f"{fr:05d}.png")
                    if not os.path.isfile(dp):
                        continue
                    refs.append(dict(
                        image=os.path.join(rgb_dir, f"{fr:05d}.png"),
                        depth=dp, pose=poses[fr], K=self._K))
                for s in make_windows(len(refs), seq_len, mode):
                    clips.append((self.LABEL, refs[s:s + seq_len]))
        return clips

    @staticmethod
    def _parse_extrinsics(path):
        poses = {}
        with open(path) as f:
            lines = f.readlines()
        for line in lines[1:]:  # skip header
            p = line.split()
            if len(p) < 17:
                continue
            fr = int(float(p[0]))
            vals = list(map(float, p[1:17]))
            poses[fr] = np.array(vals, dtype=np.float32).reshape(4, 4)
        return poses

    def load_frame(self, ref):
        image = imread_rgb01(ref["image"])
        depth = cv2.imread(ref["depth"],
                           cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 100.0
        depth = depth[..., None]
        depth[depth >= 655.0] = 0.0  # sky/invalid sentinel
        mask = depth > 0
        depth = np.minimum(depth, self.MAX_DEPTH)
        return image, depth, mask, ref["pose"], ref["K"]
