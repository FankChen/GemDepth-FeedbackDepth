# Modular dataset-loader base for GemDepth mix training.
#
# Each dataset (VKITTI1.3.1 / PointOdyssey / MVS-Synth / Dynamic Replica) lives in
# its own module and subclasses ``BaseLoader``. A loader only has to answer three
# questions, everything downstream (resize / crop / intrinsic rescale / tensor
# packing) is shared inside ``DepthVideoDataset.__getitem__``:
#
#   1. matches(data_dir)          -> is this my directory?
#   2. build_sequences(...)       -> list of (label, clip) where a clip is a list
#                                    of ``seq_len`` frame-refs (metadata only, no
#                                    pixels) -- parsed ONCE at dataset init.
#   3. load_frame(frame_ref)      -> (image, depth, mask, pose, K) read from disk
#                                    at __getitem__ time.
#
# Contract for load_frame return values (matches the existing vkitti/tartanair
# code path so the shared tail can treat every dataset identically):
#   image : HxWx3 float32, RGB, range [0, 1]
#   depth : HxWx1 float32, **metric metres**, invalid pixels set to 0
#   mask  : HxWx1 bool,    True where depth is valid
#   pose  : 4x4  float32,  **world -> camera** (same convention as vkitti/tartanair)
#   K     : 3x3  float32,  intrinsics at the ORIGINAL image resolution
#           (the shared tail rescales it by the resize factor and crop margins)

import os

# OpenEXR must be enabled before cv2 is first imported anywhere in the process
# (MVS-Synth depth is stored as .exr). dataset_mix.py sets this too, defensively.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


def imread_rgb01(path):
    """Read an 8-bit image as HxWx3 float32 RGB in [0, 1]."""
    img = cv2.imread(path).astype(np.float32) / 255.0
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_windows(num_frames, seq_len, mode, val_frac=0.9):
    """Sliding-window start indices, with the same train/val carve-out the
    original ``DepthVideoDataset`` used (last ~10% of windows -> val)."""
    seq_num = num_frames - seq_len + 1
    if seq_num <= 0:
        return []
    if mode == "train":
        start_idx = 0
        end_idx = round(seq_num)
    else:
        start_idx = round(seq_num * val_frac) + 1
        end_idx = seq_num
    return list(range(start_idx, max(start_idx, end_idx)))


class BaseLoader:
    #: label routed through __getitem__ (must be unique across datasets)
    LABEL = None
    #: metric depth clamp -- outdoor scenes 200, indoor 80
    MAX_DEPTH = 200.0
    #: mixing multiplier (clip list is repeated RATIO times, to balance datasets
    #: with very different frame counts -- mirrors vkitti_ratio / tartanair_ratio)
    RATIO = 1

    def matches(self, data_dir):
        raise NotImplementedError

    def build_sequences(self, data_dir, mode, seq_len):
        raise NotImplementedError

    def load_frame(self, frame_ref):
        raise NotImplementedError
