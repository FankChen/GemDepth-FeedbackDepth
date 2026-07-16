#!/usr/bin/env python3
"""Visualize all four v2 DPT predictions and their matching stage warps.

Accepts either a full training checkpoint (checkpoint_N.pth) or final_model.pth.
The figure is deliberately fail-fast: every p4/p3/p2/p1 row must contain a
finite depth, same-resolution target/warped RGB, non-empty validity and errors.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.dataset_mix import DepthVideoDataset
from loss.videoloss import compute_scale_and_shift
from model.factory import build_gemdepth_from_config


def align_depth(prediction, gt_depth, mask, max_depth=100.0):
    prediction = prediction.float()
    gt_depth = gt_depth.float()
    valid = (mask > 0.5) & (gt_depth > 1e-3) & (gt_depth <= max_depth)
    scale, shift = compute_scale_and_shift(
        prediction.flatten(1, 2),
        (1.0 / gt_depth.clamp(min=1e-3)).flatten(1, 2),
        valid.float().flatten(1, 2),
    )
    aligned = scale[:, None, None, None] * prediction + shift[:, None, None, None]
    depth = (1.0 / aligned.clamp(min=1e-3)).clamp(1e-3, max_depth)
    relative_error = ((depth - gt_depth).abs() / gt_depth.clamp(min=1e-3)) * valid
    return depth, relative_error, valid


def denormalize(images):
    mean = images.new_tensor((0.485, 0.456, 0.406)).reshape(1, 1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).reshape(1, 1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def percentile(values, q=99.0, floor=1e-6):
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    return max(float(np.percentile(finite, q)), floor) if finite.size else floor


def show(axis, image, title, cmap=None, vmin=None, vmax=None):
    axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=8)
    axis.axis('off')


def load_state(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        return payload['model_state_dict'], int(payload.get('total_step', -1))
    return payload, -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/scratch_ed_gt_error_rgbfeat_v2.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='/mnt/workspace/vkitti/vkitti')
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--frame_idx', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260714)
    parser.add_argument('--output', default='results/dpt_warp_v2_visualization/sample0')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    for path in (args.config, args.checkpoint, args.data_dir):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    if str(config.model.head_type) != 'multiscale_gt_error_v2':
        raise ValueError('Visualization requires head_type=multiscale_gt_error_v2')
    crop_size = int(config.dataset.train.crop_size)
    frames = int(config.dataset.train.seq_len)
    heldout = list(OmegaConf.select(
        config, 'dataset.val.include_scenes', default=['Scene20']))
    dataset = DepthVideoDataset(
        mode='train', data_dirs=[args.data_dir], crop_size=crop_size,
        seq_len=frames, include_scenes=heldout, window_stride=frames)
    dataset.data_paths = list(dataset.vkitti_data_paths)
    transform = dataset.transform.get('vkitti')
    if transform is not None and hasattr(transform, 'set_random_seed'):
        transform.set_random_seed(args.seed)
    if not 0 <= args.sample_idx < len(dataset):
        raise IndexError(args.sample_idx)
    if not 0 <= args.frame_idx < frames:
        raise IndexError(args.frame_idx)

    sample = dataset[args.sample_idx]
    device = torch.device('cuda')
    images = sample['image'].unsqueeze(0).to(device)
    gt_depth = sample['depth'].unsqueeze(0).to(device).squeeze(2)
    mask = sample['mask'].unsqueeze(0).to(device).squeeze(2)
    K = torch.as_tensor(sample['IntM']).unsqueeze(0).to(device)
    poses = torch.stack([torch.as_tensor(value) for value in sample['poses']])
    poses = poses.unsqueeze(0).to(device)

    model = build_gemdepth_from_config(config, load_backbone_pretrained=False)
    state, step = load_state(args.checkpoint)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.head.capture_warps = True
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        prediction, _, _, _ = model(
            images, gt_intrinsics=K, gt_extrinsics=poses)

    head = model.head
    expected_stages = ('p4', 'p3', 'p2', 'p1')
    for name, container in (
            ('stage_depths', head.stage_depths),
            ('metric_depths', dict(zip(expected_stages, head.metric_depths))),
            ('error_maps', head.error_maps),
            ('valid_maps', head.valid_maps),
            ('warp_visuals', head.warp_visuals)):
        if tuple(container) != expected_stages:
            raise RuntimeError(f'{name} stages={tuple(container)}, expected={expected_stages}')

    frame = args.frame_idx
    rgb_full = denormalize(images.float()).cpu()[0, frame].permute(1, 2, 0).numpy()
    gt_np = gt_depth.cpu()[0, frame].numpy()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {'checkpoint': args.checkpoint, 'step': step, 'sample_idx': args.sample_idx,
               'frame_idx': frame, 'stages': {}}

    for index, stage in enumerate(expected_stages):
        inverse = head.stage_depths[stage].squeeze(2).to(device)
        depth, rel_error, valid_gt = align_depth(inverse, gt_depth, mask)
        metric = head.metric_depths[index].detach().float().cpu()
        error = head.error_maps[stage].detach().float().cpu()
        valid = head.valid_maps[stage].detach().float().cpu()
        target = head.warp_visuals[stage]['target'].detach().float().cpu()
        warped = head.warp_visuals[stage]['warped'].detach().float().cpu()

        native_hw = metric.shape[-2:]
        if target.shape[-2:] != native_hw or warped.shape[-2:] != native_hw:
            raise RuntimeError(
                f'{stage}: image/depth mismatch target={target.shape[-2:]}, '
                f'warped={warped.shape[-2:]}, metric={native_hw}')
        tensors = (inverse, metric, error, valid, target, warped)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise FloatingPointError(f'{stage}: non-finite visualization tensor')
        valid_fraction = float(valid.mean())
        if valid_fraction <= 0:
            raise RuntimeError(f'{stage}: zero warp validity; stop training and inspect geometry')

        target_rgb = target[0, frame].permute(1, 2, 0).numpy()
        warped_rgb = warped[0, frame].permute(1, 2, 0).numpy()
        rows.append({
            'stage': stage,
            'target': target_rgb,
            'warped': warped_rgb,
            'warp_difference': np.abs(target_rgb - warped_rgb).mean(axis=2),
            'metric': metric[0, frame, 0].numpy(),
            'depth': depth.cpu()[0, frame].numpy(),
            'relative_error': rel_error.cpu()[0, frame].numpy(),
            'rgb_error': error[0, frame, 0].numpy(),
            'feature_error': error[0, frame, 1].numpy(),
            'valid': valid[0, frame, 0].numpy(),
        })
        summary['stages'][stage] = {
            'native_height': int(native_hw[0]),
            'native_width': int(native_hw[1]),
            'metric_min': float(metric.min()),
            'metric_mean': float(metric.mean()),
            'metric_max': float(metric.max()),
            'valid_fraction': valid_fraction,
            'relative_error_mean': float(
                rel_error[valid_gt].mean()) if bool(valid_gt.any()) else float('nan'),
            'rgb_error_mean': float(error[:, :, 0].sum() / valid.sum().clamp(min=1)),
            'feature_error_mean': float(error[:, :, 1].sum() / valid.sum().clamp(min=1)),
        }

    error_limit = percentile(np.concatenate([
        row['relative_error'].reshape(-1) for row in rows]))
    warp_limit = percentile(np.concatenate([
        row['warp_difference'].reshape(-1) for row in rows]))
    fig, axes = plt.subplots(4, 9, figsize=(27, 12))
    for row_index, row in enumerate(rows):
        show(axes[row_index, 0], row['target'], f"{row['stage']} target RGB")
        show(axes[row_index, 1], row['warped'], 'selected warped RGB')
        show(axes[row_index, 2], row['warp_difference'], '|target-warped|',
             'inferno', 0, warp_limit)
        show(axes[row_index, 3], row['metric'], 'metric depth (m)',
             'magma', 1e-3, 100)
        show(axes[row_index, 4], row['depth'], 'SSI-aligned depth',
             'magma', 1e-3, 100)
        show(axes[row_index, 5], row['relative_error'], 'depth relative error',
             'inferno', 0, error_limit)
        show(axes[row_index, 6], row['rgb_error'], 'RGB error [0,1]',
             'inferno', 0, 1)
        show(axes[row_index, 7], row['feature_error'], 'Feature cosine [0,1]',
             'inferno', 0, 1)
        show(axes[row_index, 8], row['valid'], 'validity', 'gray', 0, 1)
    fig.suptitle(
        f'Four-level DPT + matching warp diagnostic | step={step} | '
        f'sample={args.sample_idx} frame={frame}', fontsize=12)
    fig.tight_layout()
    figure_path = output / 'four_level_depth_warp.png'
    fig.savefig(figure_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    summary['final_prediction_shape'] = list(prediction.shape)
    summary['depth_paths'] = [str(path) for path in sample['path']]
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f'wrote {figure_path}')


if __name__ == '__main__':
    main()
