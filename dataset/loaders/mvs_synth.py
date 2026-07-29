# MVS-Synth loader (GTA-V rendered multi-view video, 120 sequences x 100 frames).
#
# On-disk layout (verified on Bosch):
#   <root>/GTAV_720/<seq>/images/<frame:04d>.png
#   <root>/GTAV_720/<seq>/depths/<frame:04d>.exr
#   <root>/GTAV_720/<seq>/poses/<frame:04d>.json
#
# Depth: EXR float32 metres. Sky is +inf -> masked invalid. Large outdoor depths
#        (median ~300 m) -> clamp with the outdoor MAX_DEPTH.
# Pose json: {"c_x","c_y","f_x","f_y","extrinsic": 4x4 world->camera}.
#            Intrinsics are per-frame (read from the json).
#
# NOTE: reading .exr requires OPENCV_IO_ENABLE_OPENEXR=1 to be set BEFORE cv2 is
# first imported; base.py / dataset_mix.py set it at module top.

import json
import os

import cv2
import numpy as np

from dataset.loaders.base import BaseLoader, imread_rgb01, make_windows


class MVSSynthLoader(BaseLoader):
    LABEL = "mvs_synth"
    MAX_DEPTH = 500.0   # outdoor GTA-V: real geometry extends well past 200m
                        # (only ~8% is sky/inf); 500m keeps ~69% of valid pixels
                        # while dropping the least-reliable >500m far-render tail.
    RATIO = 26  # ~12k frames, scaled up towards the ~310k/epoch of the big sets

    def matches(self, data_dir):
        d = data_dir.replace("-", "_").lower()
        return "mvs_synth" in d

    def build_sequences(self, data_dir, mode, seq_len):
        root = os.path.join(data_dir, "GTAV_720")
        if not os.path.isdir(root):
            root = data_dir
        if not os.path.isdir(root):
            return []
        clips = []
        for seq in sorted(os.listdir(root)):
            sdir = os.path.join(root, seq)
            img_dir = os.path.join(sdir, "images")
            dep_dir = os.path.join(sdir, "depths")
            pose_dir = os.path.join(sdir, "poses")
            if not os.path.isdir(img_dir):
                continue
            stems = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(img_dir) if f.endswith(".png"))
            refs = []
            for st in stems:
                pj = os.path.join(pose_dir, st + ".json")
                dp = os.path.join(dep_dir, st + ".exr")
                if not (os.path.isfile(pj) and os.path.isfile(dp)):
                    continue
                K, pose = self._parse_pose(pj)
                refs.append(dict(
                    image=os.path.join(img_dir, st + ".png"),
                    depth=dp, pose=pose, K=K))
            for s in make_windows(len(refs), seq_len, mode):
                clips.append((self.LABEL, refs[s:s + seq_len]))
        return clips

    @staticmethod
    def _parse_pose(pj):
        with open(pj) as f:
            d = json.load(f)
        K = np.array([[d["f_x"], 0.0, d["c_x"]],
                      [0.0, d["f_y"], d["c_y"]],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        pose = np.array(d["extrinsic"], dtype=np.float32).reshape(4, 4)  # world->cam
        return K, pose

    def load_frame(self, ref):
        image = imread_rgb01(ref["image"])
        depth = cv2.imread(ref["depth"], cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"failed to read EXR depth {ref['depth']} "
                          f"(is OPENCV_IO_ENABLE_OPENEXR=1 set before cv2 import?)")
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)[..., None]
        mask = np.isfinite(depth) & (depth > 0) & (depth < self.MAX_DEPTH)
        depth = np.nan_to_num(depth, posinf=0.0, neginf=0.0)
        depth[~mask] = 0.0
        return image, depth, mask, ref["pose"], ref["K"]
