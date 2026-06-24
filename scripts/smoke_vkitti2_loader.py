"""CPU smoke test for the VKITTI 2.0.3 dataloader adaptation.

Run AFTER the VKITTI 2.0.3 archives are extracted under the data root.

Usage:
    python scripts/smoke_vkitti2_loader.py [DATA_ROOT] [SEQ_LEN]

Defaults: DATA_ROOT=/home/izi2sgh/MYDATA/vkitti/  SEQ_LEN=4
It builds DepthVideoDataset('train'), reports how many sequences were found,
pulls one sample, and prints shapes / value ranges so we can confirm rgb,
depth, mask, pose (4x4 world->camera) and intrinsics are read correctly.
"""
import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.dataset_mix import DepthVideoDataset


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else '/home/izi2sgh/MYDATA/vkitti/'
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if not data_root.endswith(os.sep):
        data_root += os.sep

    print(f"[cfg] data_root={data_root}  seq_len={seq_len}")
    ds = DepthVideoDataset('train', data_dirs=[data_root], crop_size=518, seq_len=seq_len)
    print(f"[info] raw vkitti sequences = {len(ds.vkitti_data_paths)}")
    print(f"[info] total (after ratio)  = {len(ds)}")
    assert len(ds) > 0, "No sequences discovered -- check extraction layout / paths!"

    # Inspect the discovery of one raw sequence (paths + pose matrix).
    label, set_paths = ds.vkitti_data_paths[0]
    img0, dep0, pose0 = set_paths[0]
    print(f"[seq0] label={label}")
    print(f"[seq0] rgb   = {img0}")
    print(f"[seq0] depth = {dep0}")
    print(f"[seq0] pose0 (world->cam) =\n{np.asarray(pose0)}")
    assert np.asarray(pose0).shape == (4, 4), "pose must be 4x4"

    sample = ds[0]
    img = sample['image']
    dep = sample['depth']
    msk = sample['mask']
    intm = sample['IntM']
    poses = sample['poses']
    print("\n=== sample[0] ===")
    print(f"image  {tuple(img.shape)}  dtype={img.dtype}  "
          f"range=[{img.min():.3f},{img.max():.3f}]")
    print(f"depth  {tuple(dep.shape)}  "
          f"range=[{float(dep.min()):.3f},{float(dep.max()):.3f}]")
    print(f"mask   {tuple(msk.shape)}  valid_frac={float(msk.float().mean()):.3f}")
    print(f"IntM   =\n{intm}")
    print(f"poses  len={len(poses)}  pose[0].shape={np.asarray(poses[0]).shape}")

    # Basic sanity
    assert img.shape[0] == seq_len, "image frame count mismatch"
    assert dep.shape[0] == seq_len, "depth frame count mismatch"
    assert float(dep.max()) <= ds.max_depth_outer + 1e-3, "depth not clipped"
    assert float(msk.float().mean()) > 0.05, "mask almost empty -- depth read wrong?"
    print("\nALL CHECKS PASSED")


if __name__ == '__main__':
    main()
