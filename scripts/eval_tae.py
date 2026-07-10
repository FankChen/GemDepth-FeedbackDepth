"""TAE (Temporal Alignment Error): multi-view temporal consistency of predicted depth.

Loads a trained GemDepth checkpoint, runs it on VKITTI sequences (which carry GT poses),
aligns the predicted disparity to metric depth per-sequence via GT scale/shift, then measures
the reprojection consistency across neighbour frames (model/util/temporal). Lower = temporally
more consistent. No retraining — a relative metric to compare arms (baseline vs em_single vs
perlayer ...). This is the temporal-consistency axis where the error-map / warp heads should win.

Usage:
  python scripts/eval_tae.py --ckpt checkpoint/single_a100_perlayer/final_model.pth \
      --head_type perlayer --warp_signal feat --data_dirs /mnt/workspace/vkitti2/vkitti \
      --seq_len 16 --num_seqs 40
"""
import os
import sys
import argparse
import random
import statistics

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from model.gemdepth import GemDepth
from dataset.dataset_mix import DepthVideoDataset, safe_collate
from model.util.temporal import geometric_temporal_consistency, align_pred_metric

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def _str2bool(s):
    return str(s).lower() not in ('false', '0', 'no', 'off')


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--head_type', default='temporal',
                   choices=['temporal', 'errormap', 'errormap_single', 'batlin', 'perlayer', 'perlayer_refine'])
    p.add_argument('--warp_signal', default='feat', choices=['rgb', 'feat', 'rgbfeat', 'hog'])
    p.add_argument('--use_warp', type=_str2bool, default=True)
    p.add_argument('--scales', type=str, nargs='+', default=['p2', 'p1'])
    p.add_argument('--encoder', default='vitl')
    p.add_argument('--data_dirs', nargs='+', default=['/mnt/workspace/vkitti2/vkitti'])
    p.add_argument('--seq_len', type=int, default=16)
    p.add_argument('--num_seqs', type=int, default=40)
    p.add_argument('--offsets', type=int, nargs='+', default=[1])
    p.add_argument('--tag', default='')
    return p.parse_args()


def main():
    a = parse()
    # Deterministic crop so TAE is reproducible AND comparable across arms: the dataset uses a
    # random StatefulRandomCrop, so we re-seed every run and load single-process.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GemDepth(**MODEL_CONFIGS[a.encoder], head_type=a.head_type,
                     warp_signal=a.warp_signal, scales=tuple(a.scales), use_warp=a.use_warp)
    ckpt = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {a.ckpt} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(dev)

    ds = DepthVideoDataset(mode='train', data_dirs=list(a.data_dirs), crop_size=518, seq_len=a.seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=safe_collate)

    taes = []
    for data in loader:
        if data is None:
            continue
        if len(taes) >= a.num_seqs:
            break
        image = data['image'].to(dev)
        depth_gt = data['depth'].to(dev)
        mask = (depth_gt > 0).float()
        K = data['IntM'].to(dev)
        ext = torch.stack(data['poses'], dim=1).to(dev)          # (B,T,4,4)
        with torch.no_grad():
            depth_pred = model(image)[0]                         # (B,T,H,W)
            if depth_pred.dim() == 4:
                depth_pred = depth_pred.unsqueeze(2)             # (B,T,1,H,W)
            pred_m = align_pred_metric(depth_pred.float(), depth_gt.float(), mask.float())
            tae = geometric_temporal_consistency(pred_m, K.float(), ext.float(), offsets=tuple(a.offsets))
        taes.append(tae.item())

    if not taes:
        print("TAE: no sequences evaluated")
        return
    mean = sum(taes) / len(taes)
    med = statistics.median(taes)
    label = a.tag or os.path.basename(os.path.dirname(a.ckpt))
    print(f"TAE[{label}] seqs={len(taes)} mean={mean:.5f} median={med:.5f} "
          f"(head={a.head_type} sig={a.warp_signal} warp={a.use_warp} offsets={a.offsets})")


if __name__ == '__main__':
    main()
