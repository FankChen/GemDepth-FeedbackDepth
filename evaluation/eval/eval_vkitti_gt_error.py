#!/usr/bin/env python3
"""Deterministic VKITTI held-out evaluation for scratch multiscale/GT-error heads.

GT-camera heads require camera matrices at inference, so this oracle stage is
measured on VKITTI where GT intrinsics/extrinsics are available.  The evaluator
also supports the plain multiscale baseline for an identical protocol.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.dataset_mix import DepthVideoDataset, safe_collate
from loss.videoloss import compute_scale_and_shift
from model.factory import build_gemdepth_from_config


def install_zero_error_hooks(model, mode):
    """Inference-time control that disables v2's error pathway without retraining.

    'all' feeds an all-zero error channel to every error encoder and the final
    correction. Because those modules are bias-free, the feedback and correction
    become exactly zero, so the trained v2 collapses to its pure anchored
    Temporal DPT readout. 'residual' zeros only the two reprojection-residual
    channels and keeps the validity channel, isolating whether the residual
    VALUES matter beyond the valid-pixel mask.
    """
    head = model.head

    def _pre_hook(module, inputs):
        (features,) = inputs
        if mode == 'all':
            return (torch.zeros_like(features),)
        if mode == 'residual':
            gated = features.clone()
            gated[:, :2] = 0.0
            return (gated,)
        raise ValueError(f"Unknown zero_error mode {mode!r}")

    handles = [encoder.register_forward_pre_hook(_pre_hook)
               for encoder in head.error_encoders.values()]
    handles.append(head.final_error_correction.register_forward_pre_hook(_pre_hook))
    return handles


def evaluate_batch(pred_inverse_depth, gt_depth, mask, max_depth):
    """Return sums of per-frame AbsRel/RMSE/delta1 after per-clip SSI alignment."""
    pred = pred_inverse_depth.float()
    gt = gt_depth.squeeze(2).float()
    valid = (mask.squeeze(2) > 0.5) & (gt > 1e-3) & (gt < max_depth)
    b, t, h, w = pred.shape
    scale, shift = compute_scale_and_shift(
        pred.flatten(1, 2), (1.0 / gt.clamp(min=1e-3)).flatten(1, 2),
        valid.float().flatten(1, 2))
    aligned_inverse = scale[:, None, None, None] * pred + shift[:, None, None, None]
    pred_depth = 1.0 / aligned_inverse.clamp(min=1e-3)
    pred_depth = pred_depth.clamp(min=1e-3, max=max_depth)

    totals = pred.new_zeros(3)
    count = 0
    for bi in range(b):
        for ti in range(t):
            m = valid[bi, ti]
            if int(m.sum()) == 0:
                continue
            p, g = pred_depth[bi, ti][m], gt[bi, ti][m]
            absrel = ((p - g).abs() / g).mean()
            rmse = torch.sqrt(((p - g) ** 2).mean())
            delta1 = (torch.maximum(p / g, g / p) < 1.25).float().mean()
            totals += torch.stack((absrel, rmse, delta1))
            count += 1
    return totals, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output', default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_clips', type=int, default=0,
                        help='0 evaluates all unique held-out clips')
    parser.add_argument('--max_depth', type=float, default=80.0)
    parser.add_argument('--seed', type=int, default=20260714)
    parser.add_argument('--zero_error', choices=['none', 'all', 'residual'],
                        default='none',
                        help='Inference-time error-pathway control for v2 heads '
                             '(no retraining). "all" disables the entire error '
                             'feedback (pure anchored Temporal DPT); "residual" '
                             'zeros only the reprojection residual and keeps validity.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.device('cuda')
    cfg = OmegaConf.load(args.config)
    crop_size = int(cfg.dataset.train.crop_size)
    seq_len = int(cfg.dataset.train.seq_len)
    heldout_scenes = list(OmegaConf.select(
        cfg, 'dataset.val.include_scenes', default=['Scene20']))
    dataset = DepthVideoDataset(
        mode='train', data_dirs=[args.data_dir], crop_size=crop_size,
        seq_len=seq_len, include_scenes=heldout_scenes,
        window_stride=seq_len)
    # Remove training-time vkitti_ratio replication. Keep only unique clips from
    # fully held-out scenes (the matching train configs explicitly exclude them).
    dataset.data_paths = list(dataset.vkitti_data_paths)
    transform = dataset.transform.get('vkitti')
    if transform is not None and hasattr(transform, 'set_random_seed'):
        transform.set_random_seed(args.seed)
    if args.max_clips > 0:
        dataset.data_paths = dataset.data_paths[:args.max_clips]
    if not dataset.data_paths:
        raise RuntimeError(f'No held-out VKITTI clips found under {args.data_dir}')
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=safe_collate,
        pin_memory=True, drop_last=False)

    # final_model.pth is the complete state_dict (backbone + LoRA + head). Build
    # offline, then strict-load every tensor; no HF access is needed for evaluation.
    model = build_gemdepth_from_config(cfg, load_backbone_pretrained=False)
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    head_type = str(cfg.model.head_type)
    if args.zero_error != 'none':
        if head_type != 'multiscale_gt_error_v2':
            raise ValueError('--zero_error is only supported for multiscale_gt_error_v2')
        install_zero_error_hooks(model, args.zero_error)
        print(f'[control] zero_error={args.zero_error}: error pathway disabled at '
              f'inference (same checkpoint, no retraining)')

    totals = torch.zeros(3, device=device)
    count = 0
    for batch in tqdm(loader, desc='VKITTI held-out'):
        images = batch['image'].to(device, non_blocking=True)
        gt = batch['depth'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        K = batch['IntM'].to(device, non_blocking=True)
        poses = batch['poses']
        ext = torch.stack(poses, dim=1).to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if head_type in ('multiscale_gt_error', 'multiscale_gt_error_v2'):
                pred, _, _, _ = model(images, gt_intrinsics=K, gt_extrinsics=ext)
            else:
                pred, _, _, _ = model(images)
        batch_totals, batch_count = evaluate_batch(pred, gt, mask, args.max_depth)
        totals += batch_totals
        count += batch_count

    metrics = (totals / max(count, 1)).cpu().tolist()
    result = {
        'config': args.config,
        'checkpoint': args.ckpt,
        # DepthVideoDataset.__len__ rounds down to a multiple of four, so report
        # what the DataLoader actually evaluated rather than the raw path count.
        'clips': len(dataset),
        'clips_discovered': len(dataset.data_paths),
        'heldout_scenes': heldout_scenes,
        'frames_evaluated': count,
        'seed': args.seed,
        'zero_error': args.zero_error,
        'abs_relative_difference': metrics[0],
        'rmse_linear': metrics[1],
        'delta1_acc': metrics[2],
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
