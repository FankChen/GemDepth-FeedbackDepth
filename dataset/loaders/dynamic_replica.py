# Dynamic Replica loader (dynamic-scene stereo video; left/right treated as two
# independent monocular videos).
#
# On-disk layout (verified on Bosch):
#   <root>/data/<split>/<seq>/images/<seq>-<frame:04d>.png
#   <root>/data/<split>/<seq>/depths/<seq>_<frame:04d>.geometric.png
#   <root>/data/<split>/frame_annotations_<split>.jgz   (gzip JSON, camera params)
#
# Depth: 16-bit PNG whose raw bytes are the float16 depth bit-pattern
#        (CO3D/PyTorch3D convention) -> reinterpret uint16 bytes as float16,
#        metres. Verified: decoded range ~0..4 m (indoor) -> indoor MAX_DEPTH.
# Camera: per-frame ``viewpoint`` in the .jgz with PyTorch3D NDC-isotropic
#        intrinsics {R (3x3, row-vec world->cam), T, focal_length, principal_point}.
#        Converted here to pixel intrinsics + a world->camera 4x4 in the OpenCV
#        camera frame used by the rest of the pipeline.
#
# ⚠ PyTorch3D view space is (+X left, +Y up, +Z forward); the pipeline / K use
#   OpenCV (+X right, +Y down, +Z forward). We left-multiply the extrinsics by
#   diag(-1,-1,1) to convert. This axis flip is the one thing to sanity-check in
#   the smoke test (re-project depth across two frames and verify consistency).

import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from dataset.loaders.base import BaseLoader, imread_rgb01, make_windows

# OpenCV(view) = FLIP @ PyTorch3D(view)
_FLIP = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)


class DynamicReplicaLoader(BaseLoader):
    LABEL = "dynamic_replica"
    MAX_DEPTH = 80.0
    RATIO = 1

    def matches(self, data_dir):
        d = data_dir.replace("-", "_").lower()
        return "dynamic_replica" in d

    def build_sequences(self, data_dir, mode, seq_len):
        jgz_files = glob.glob(
            os.path.join(data_dir, "**", "frame_annotations_*.jgz"), recursive=True)
        clips = []
        for jpath in sorted(jgz_files):
            split_dir = os.path.dirname(jpath)
            with gzip.open(jpath, "rt") as f:
                entries = json.load(f)
            # group frames by their on-disk sequence folder (first path component)
            groups = defaultdict(list)
            for e in entries:
                groups[e["image"]["path"].split("/")[0]].append(e)
            for folder, es in sorted(groups.items()):
                es.sort(key=lambda e: e["frame_number"])
                refs = []
                for e in es:
                    img = os.path.join(split_dir, e["image"]["path"])
                    dep = os.path.join(split_dir, e["depth"]["path"])
                    if not (os.path.isfile(img) and os.path.isfile(dep)):
                        continue
                    H, W = e["image"]["size"]
                    K, pose = self._camera(e["viewpoint"], H, W)
                    refs.append(dict(image=img, depth=dep, pose=pose, K=K))
                for s in make_windows(len(refs), seq_len, mode):
                    clips.append((self.LABEL, refs[s:s + seq_len]))
        return clips

    @staticmethod
    def _camera(vp, H, W):
        # NDC-isotropic -> pixel intrinsics
        s = min(W, H) / 2.0
        fl = vp["focal_length"]
        pp = vp["principal_point"]
        fx = float(fl[0]) * s
        fy = float(fl[1]) * s
        cx = W / 2.0 - float(pp[0]) * s
        cy = H / 2.0 - float(pp[1]) * s
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        # PyTorch3D row-vec world->cam (X_cam = X_world @ R + T) -> column-vec 4x4
        R = np.array(vp["R"], dtype=np.float32)
        T = np.array(vp["T"], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.T
        pose[:3, 3] = T
        pose = _FLIP @ pose  # PyTorch3D view frame -> OpenCV view frame
        return K, pose.astype(np.float32)

    def load_frame(self, ref):
        image = imread_rgb01(ref["image"])
        arr = np.array(Image.open(ref["depth"]), dtype=np.uint16)
        depth = np.frombuffer(arr.tobytes(), dtype=np.float16).astype(np.float32)
        depth = depth.reshape(arr.shape)[..., None]
        mask = (depth > 0) & (depth < self.MAX_DEPTH) & np.isfinite(depth)
        depth = np.nan_to_num(depth, posinf=0.0, neginf=0.0)
        depth[~mask] = 0.0
        return image, depth, mask, ref["pose"], ref["K"]
