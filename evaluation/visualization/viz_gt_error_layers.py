#!/usr/bin/env python3
"""Fast, single-clip diagnosis for the GT-camera RGB+Feature experiment.

No retraining is required. The script loads the multiscale-only baseline and the
RGB+Feature checkpoint on one deterministic Scene20 crop, then exports:

* p4->p1 accumulated depth and per-scale error;
* p2/p1 raw + normalized RGB/Feature residuals and validity;
* p2 feedback norm and p1 signed correction;
* inference-time counterfactuals (normal / RGB removed / Feature removed /
  both residuals removed).

Counterfactuals diagnose checkpoint dependency only; they are not substitutes
for separately trained ablations.
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
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.dataset_mix import DepthVideoDataset
from loss.videoloss import compute_scale_and_shift
from model.factory import build_gemdepth_from_config
from model.util.gt_error import imagenet_denormalize


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(config_path, checkpoint_path, device):
    cfg = OmegaConf.load(config_path)
    model = build_gemdepth_from_config(cfg, load_backbone_pretrained=False)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg


def align_inverse_depth(pred, gt, mask, max_depth=80.0):
    """Clip-level SSI alignment, matching eval_vkitti_gt_error.py."""
    pred = pred.float()
    gt = gt.float()
    valid = (mask > 0.5) & (gt > 1e-3) & (gt < max_depth)
    scale, shift = compute_scale_and_shift(
        pred.flatten(1, 2),
        (1.0 / gt.clamp(min=1e-3)).flatten(1, 2),
        valid.float().flatten(1, 2),
    )
    aligned_inv = scale[:, None, None, None] * pred + shift[:, None, None, None]
    depth = (1.0 / aligned_inv.clamp(min=1e-3)).clamp(min=1e-3, max=max_depth)
    rel_error = ((depth - gt).abs() / gt.clamp(min=1e-3)) * valid

    frame_metrics = []
    for bi in range(depth.shape[0]):
        for ti in range(depth.shape[1]):
            m = valid[bi, ti]
            if not bool(m.any()):
                continue
            p, g = depth[bi, ti][m], gt[bi, ti][m]
            frame_metrics.append(torch.stack((
                ((p - g).abs() / g).mean(),
                torch.sqrt(((p - g) ** 2).mean()),
                (torch.maximum(p / g, g / p) < 1.25).float().mean(),
            )))
    metrics = torch.stack(frame_metrics).mean(0) if frame_metrics else depth.new_zeros(3)
    return depth, rel_error, valid, metrics


def align_aux_depth(aux, gt, mask, max_depth=80.0):
    if aux.ndim == 5:
        pred = aux.squeeze(2)
    else:
        raise ValueError(f'Expected (B,T,1,H,W), got {tuple(aux.shape)}')
    size = pred.shape[-2:]
    b, t = gt.shape[:2]
    gt_s = F.interpolate(gt.reshape(b * t, 1, *gt.shape[-2:]), size=size, mode='nearest')
    gt_s = gt_s.reshape(b, t, *size)
    mask_s = F.interpolate(mask.reshape(b * t, 1, *mask.shape[-2:]).float(),
                           size=size, mode='nearest')
    mask_s = mask_s.reshape(b, t, *size)
    return align_inverse_depth(pred, gt_s, mask_s, max_depth=max_depth)


def percentile_max(arrays, percentile=99.0, floor=1e-6):
    values = []
    for value in arrays:
        a = np.asarray(value)
        finite = a[np.isfinite(a)]
        if finite.size:
            values.append(finite)
    if not values:
        return floor
    return max(float(np.percentile(np.concatenate(values), percentile)), floor)


def show(ax, image, title, cmap=None, vmin=None, vmax=None):
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis('off')
    return im


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def tensor_cpu(value):
    return value.detach().float().cpu()


def stage_statistics(raw_maps, norm_maps):
    output = {}
    for stage in ('p2', 'p1'):
        rgb = raw_maps[stage]['rgb'].numpy().reshape(-1)
        feat = raw_maps[stage]['feat'].numpy().reshape(-1)
        valid = (raw_maps[stage]['rgb_valid'].numpy().reshape(-1) > 0.5)
        valid &= (raw_maps[stage]['feat_valid'].numpy().reshape(-1) > 0.5)
        rgb_v, feat_v = rgb[valid], feat[valid]
        if rgb_v.size > 1 and np.std(rgb_v) > 0 and np.std(feat_v) > 0:
            pearson = float(np.corrcoef(rgb_v, feat_v)[0, 1])
        else:
            pearson = float('nan')
        if rgb_v.size:
            rgb_threshold = np.quantile(rgb_v, 0.9)
            feat_threshold = np.quantile(feat_v, 0.9)
            rgb_top = rgb_v >= rgb_threshold
            feat_top = feat_v >= feat_threshold
            union = np.logical_or(rgb_top, feat_top).sum()
            top_iou = float(np.logical_and(rgb_top, feat_top).sum() / max(union, 1))
        else:
            top_iou = float('nan')
        normalized = norm_maps[stage]
        output[stage] = {
            'valid_fraction': float(valid.mean()),
            'raw_rgb_mean': float(rgb_v.mean()) if rgb_v.size else float('nan'),
            'raw_feat_mean': float(feat_v.mean()) if feat_v.size else float('nan'),
            'pearson_rgb_feat': pearson,
            'top10_iou_rgb_feat': top_iou,
            'rgb_saturation_fraction': float((normalized[:, :, 0] >= 0.999).float().mean()),
            'feat_saturation_fraction': float((normalized[:, :, 1] >= 0.999).float().mean()),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/mnt/workspace/vkitti/vkitti')
    parser.add_argument('--baseline_config', default='config/scratch_ed_gt_error_baseline.yaml')
    parser.add_argument('--baseline_ckpt',
                        default='checkpoint/scratch_ed_gt_error_baseline/final_model.pth')
    parser.add_argument('--rgbfeat_config', default='config/scratch_ed_gt_error_rgbfeat.yaml')
    parser.add_argument('--rgbfeat_ckpt',
                        default='checkpoint/scratch_ed_gt_error_rgbfeat/final_model.pth')
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--frame_idx', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260714)
    parser.add_argument('--max_depth', type=float, default=80.0)
    parser.add_argument('--output', default='results/gt_error_visualization/sample0')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    for required in (args.baseline_config, args.baseline_ckpt,
                     args.rgbfeat_config, args.rgbfeat_ckpt, args.data_dir):
        if not Path(required).exists():
            raise FileNotFoundError(required)

    seed_everything(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')

    rgbfeat_cfg = OmegaConf.load(args.rgbfeat_config)
    crop_size = int(rgbfeat_cfg.dataset.train.crop_size)
    seq_len = int(rgbfeat_cfg.dataset.train.seq_len)
    heldout = list(OmegaConf.select(
        rgbfeat_cfg, 'dataset.val.include_scenes', default=['Scene20']))
    dataset = DepthVideoDataset(
        mode='train', data_dirs=[args.data_dir], crop_size=crop_size,
        seq_len=seq_len, include_scenes=heldout, window_stride=seq_len)
    dataset.data_paths = list(dataset.vkitti_data_paths)
    transform = dataset.transform.get('vkitti')
    if transform is not None and hasattr(transform, 'set_random_seed'):
        transform.set_random_seed(args.seed)
    if not 0 <= args.sample_idx < len(dataset):
        raise IndexError(f'sample_idx={args.sample_idx}, dataset length={len(dataset)}')
    if not 0 <= args.frame_idx < seq_len:
        raise IndexError(f'frame_idx={args.frame_idx}, seq_len={seq_len}')

    sample = dataset[args.sample_idx]
    images = sample['image'].unsqueeze(0).to(device)
    gt = sample['depth'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)
    K = torch.as_tensor(sample['IntM']).unsqueeze(0).to(device)
    ext = torch.stack([torch.as_tensor(p) for p in sample['poses']], dim=0)
    ext = ext.unsqueeze(0).to(device)
    gt_hw = gt.squeeze(2)
    mask_hw = mask.squeeze(2)
    frame = args.frame_idx

    # Baseline first, then release it before loading the larger RGB+Feature arm.
    baseline_model, _ = load_model(args.baseline_config, args.baseline_ckpt, device)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        baseline_pred, _, _, _ = baseline_model(images)
    baseline_depth, baseline_error, _, baseline_metrics = align_inverse_depth(
        baseline_pred, gt_hw, mask_hw, args.max_depth)
    del baseline_model
    torch.cuda.empty_cache()

    model, _ = load_model(args.rgbfeat_config, args.rgbfeat_ckpt, device)
    head = model.head
    if getattr(head, 'error_signal', None) != 'rgbfeat':
        raise ValueError('This diagnosis requires an RGB+Feature checkpoint')

    intervention = {'mode': 'normal'}
    captured = {}

    def apply_intervention(error_tensor):
        mode = intervention['mode']
        if mode == 'normal':
            return error_tensor
        value = error_tensor.clone()
        if mode in ('no_rgb', 'zero_residuals'):
            value[:, 0] = 0
        if mode in ('no_feat', 'zero_residuals'):
            value[:, 1] = 0
        return value

    def p2_pre_hook(_module, inputs):
        if inputs[0].shape[1] != 3:
            raise ValueError(f'Expected three p2 error channels, got {inputs[0].shape[1]}')
        return (apply_intervention(inputs[0]),)

    def p1_pre_hook(_module, inputs):
        value = inputs[0]
        expected_channels = head.p1_correction[0].in_channels
        if value.shape[1] != expected_channels:
            raise ValueError(
                f'Expected {expected_channels} p1 correction channels, got {value.shape[1]}')
        feature, error = value[:, :-3], value[:, -3:]
        return (torch.cat((feature, apply_intervention(error)), dim=1),)

    def p2_capture_hook(_module, _inputs, output_tensor):
        if intervention['mode'] == 'normal':
            captured['p2_feedback'] = tensor_cpu(output_tensor)

    def p1_capture_hook(_module, _inputs, output_tensor):
        if intervention['mode'] == 'normal':
            captured['p1_correction'] = tensor_cpu(output_tensor)

    handles = [
        head.error_encoders['p2'].register_forward_pre_hook(p2_pre_hook),
        head.p1_correction.register_forward_pre_hook(p1_pre_hook),
        head.error_encoders['p2'].register_forward_hook(p2_capture_hook),
        head.p1_correction.register_forward_hook(p1_capture_hook),
    ]

    predictions = {}
    metrics = {}
    normal_cache = None
    for mode in ('normal', 'no_rgb', 'no_feat', 'zero_residuals'):
        intervention['mode'] = mode
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred, _, _, _ = model(images, gt_intrinsics=K, gt_extrinsics=ext)
        depth, error, valid, score = align_inverse_depth(
            pred, gt_hw, mask_hw, args.max_depth)
        predictions[mode] = {
            'inverse': tensor_cpu(pred),
            'depth': tensor_cpu(depth),
            'error': tensor_cpu(error),
            'valid': tensor_cpu(valid),
        }
        metrics[mode] = {
            'absrel': float(score[0]), 'rmse': float(score[1]), 'delta1': float(score[2])
        }
        if mode == 'normal':
            normal_cache = {
                'aux_depths': [tensor_cpu(v) for v in head.aux_depths],
                'metric_depths': [tensor_cpu(v) for v in head.metric_depths],
                'error_maps': {k: tensor_cpu(v) for k, v in head.error_maps.items()},
                'valid_maps': {k: tensor_cpu(v) for k, v in head.valid_maps.items()},
                'raw_error_maps': {
                    stage: {name: tensor_cpu(value) for name, value in maps.items()}
                    for stage, maps in head.raw_error_maps.items()
                },
            }

    for handle in handles:
        handle.remove()

    metrics['baseline'] = {
        'absrel': float(baseline_metrics[0]),
        'rmse': float(baseline_metrics[1]),
        'delta1': float(baseline_metrics[2]),
    }
    metrics['stage_statistics'] = stage_statistics(
        normal_cache['raw_error_maps'], normal_cache['error_maps'])
    metrics['sample'] = {
        'sample_idx': args.sample_idx,
        'frame_idx': frame,
        'seed': args.seed,
        'depth_paths': [str(p) for p in sample['path']],
    }
    (output / 'summary.json').write_text(
        json.dumps(metrics, indent=2, allow_nan=True) + '\n', encoding='utf-8')

    rgb = imagenet_denormalize(images.float()).cpu()[0, frame].permute(1, 2, 0).numpy()
    gt_np = gt_hw.cpu()[0, frame].numpy()
    valid_np = (mask_hw.cpu()[0, frame].numpy() > 0.5) & (gt_np < args.max_depth)
    gt_show = np.where(valid_np, gt_np, np.nan)

    # 1) High-level output + correction diagnosis.
    normal_depth = predictions['normal']['depth'][0, frame].numpy()
    normal_error = predictions['normal']['error'][0, frame].numpy()
    base_depth = baseline_depth.cpu()[0, frame].numpy()
    base_error = baseline_error.cpu()[0, frame].numpy()
    correction = captured['p1_correction']
    correction_bt = correction.reshape(1, seq_len, 1, *correction.shape[-2:])
    correction_full = F.interpolate(
        correction_bt.flatten(0, 1), size=predictions['normal']['inverse'].shape[-2:],
        mode='bilinear', align_corners=True).reshape(1, seq_len, *predictions['normal']['inverse'].shape[-2:])
    before_inv = predictions['normal']['inverse'] - correction_full
    before_depth, before_error, _, before_score = align_inverse_depth(
        before_inv.to(device), gt_hw, mask_hw, args.max_depth)
    metrics['before_p1_correction'] = {
        'absrel': float(before_score[0]), 'rmse': float(before_score[1]),
        'delta1': float(before_score[2]),
    }
    (output / 'summary.json').write_text(
        json.dumps(metrics, indent=2, allow_nan=True) + '\n', encoding='utf-8')
    improvement = base_error - normal_error
    signed_limit = percentile_max([np.abs(improvement[valid_np])], 99)

    fig, axes = plt.subplots(1, 8, figsize=(24, 3.5))
    show(axes[0], rgb, 'RGB target')
    show(axes[1], gt_show, 'GT depth', 'magma', 0, args.max_depth)
    show(axes[2], base_depth, f"B1 depth\nAbsRel={metrics['baseline']['absrel']:.3f}",
         'magma', 0, args.max_depth)
    show(axes[3], base_error, 'B1 relative error', 'inferno', 0,
         percentile_max([base_error[valid_np]], 99))
    show(axes[4], before_depth.cpu()[0, frame].numpy(),
         f"before p1 correction\nAbsRel={metrics['before_p1_correction']['absrel']:.3f}",
         'magma', 0, args.max_depth)
    show(axes[5], normal_depth, f"RGB+Feat depth\nAbsRel={metrics['normal']['absrel']:.3f}",
         'magma', 0, args.max_depth)
    show(axes[6], normal_error, 'RGB+Feat relative error', 'inferno', 0,
         percentile_max([normal_error[valid_np]], 99))
    show(axes[7], improvement, 'Improvement vs B1\ngreen=better', 'RdYlGn',
         -signed_limit, signed_limit)
    save_figure(fig, output / '01_overview.png')

    # 2) Four accumulated scale outputs (p4 -> p1).
    scale_depths, scale_errors = [], []
    for aux in normal_cache['aux_depths']:
        d, e, _, _ = align_aux_depth(aux.to(device), gt_hw, mask_hw, args.max_depth)
        scale_depths.append(d.cpu()[0, frame].numpy())
        scale_errors.append(e.cpu()[0, frame].numpy())
    error_limit = percentile_max(scale_errors, 99)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for index, stage in enumerate(('p4', 'p3', 'p2', 'p1')):
        show(axes[0, index], scale_depths[index], f'{stage} accumulated depth',
             'magma', 0, args.max_depth)
        show(axes[1, index], scale_errors[index], f'{stage} relative error',
             'inferno', 0, error_limit)
    save_figure(fig, output / '02_multiscale_z.png')

    # 3) Actual p2/p1 error inputs: raw and network-normalized.
    raw = normal_cache['raw_error_maps']
    normalized = normal_cache['error_maps']
    raw_rgb_limit = percentile_max([
        raw[s]['rgb'][0, frame, 0].numpy() for s in ('p2', 'p1')], 99)
    raw_feat_limit = percentile_max([
        raw[s]['feat'][0, frame, 0].numpy() for s in ('p2', 'p1')], 99)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for row, stage in enumerate(('p2', 'p1')):
        metric_index = 0 if stage == 'p2' else 1
        show(axes[row, 0], normal_cache['metric_depths'][metric_index][0, frame, 0].numpy(),
             f'{stage} metric depth', 'magma', 0, args.max_depth)
        show(axes[row, 1], raw[stage]['rgb'][0, frame, 0].numpy(),
             f'{stage} raw RGB residual', 'inferno', 0, raw_rgb_limit)
        show(axes[row, 2], normalized[stage][0, frame, 0].numpy(),
             f'{stage} normalized RGB', 'inferno', 0, 1)
        show(axes[row, 3], raw[stage]['feat'][0, frame, 0].numpy(),
             f'{stage} raw Feature residual', 'inferno', 0, raw_feat_limit)
        show(axes[row, 4], normalized[stage][0, frame, 1].numpy(),
             f'{stage} normalized Feature', 'inferno', 0, 1)
        show(axes[row, 5], normalized[stage][0, frame, 2].numpy(),
             f'{stage} shared validity', 'gray', 0, 1)
    save_figure(fig, output / '03_error_layers.png')

    # 4) What the two mechanisms inject.
    p2_feedback = captured['p2_feedback'].reshape(
        1, seq_len, captured['p2_feedback'].shape[1], *captured['p2_feedback'].shape[-2:])
    feedback_norm = torch.linalg.vector_norm(p2_feedback, dim=2)[0, frame].numpy()
    corr = correction_bt[0, frame, 0].numpy()
    corr_limit = percentile_max([np.abs(corr)], 99)
    before_error_np = before_error.cpu()[0, frame].numpy()
    corr_improvement = before_error_np - normal_error
    corr_imp_limit = percentile_max([np.abs(corr_improvement[valid_np])], 99)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    show(axes[0], feedback_norm, 'p2 feedback channel norm', 'viridis', 0,
         percentile_max([feedback_norm], 99))
    show(axes[1], corr, 'p1 signed correction', 'coolwarm', -corr_limit, corr_limit)
    show(axes[2], before_error_np, 'Error before p1 correction', 'inferno', 0,
         percentile_max([before_error_np[valid_np]], 99))
    show(axes[3], normal_error, 'Error after p1 correction', 'inferno', 0,
         percentile_max([normal_error[valid_np]], 99))
    show(axes[4], corr_improvement, 'p1 correction effect\ngreen=better', 'RdYlGn',
         -corr_imp_limit, corr_imp_limit)
    save_figure(fig, output / '04_feedback_correction.png')

    # 5) Fast inference-time counterfactuals on the same trained checkpoint.
    order = ('normal', 'no_rgb', 'no_feat', 'zero_residuals')
    cf_error_limit = percentile_max([
        predictions[name]['error'][0, frame].numpy()[valid_np] for name in order], 99)
    fig, axes = plt.subplots(2, len(order), figsize=(14, 7))
    labels = {
        'normal': 'normal RGB+Feature',
        'no_rgb': 'RGB slot = 0',
        'no_feat': 'Feature slot = 0',
        'zero_residuals': 'both residuals = 0',
    }
    for col, name in enumerate(order):
        show(axes[0, col], predictions[name]['depth'][0, frame].numpy(),
             f"{labels[name]}\nAbsRel={metrics[name]['absrel']:.3f}",
             'magma', 0, args.max_depth)
        show(axes[1, col], predictions[name]['error'][0, frame].numpy(),
             'relative error', 'inferno', 0, cf_error_limit)
    save_figure(fig, output / '05_counterfactual.png')

    np.savez_compressed(
        output / 'diagnostic_maps.npz',
        rgb=rgb,
        gt_depth=gt_np,
        baseline_depth=base_depth,
        rgbfeat_depth=normal_depth,
        baseline_error=base_error,
        rgbfeat_error=normal_error,
        p2_raw_rgb=raw['p2']['rgb'][0, frame, 0].numpy(),
        p2_raw_feat=raw['p2']['feat'][0, frame, 0].numpy(),
        p2_normalized=normalized['p2'][0, frame].numpy(),
        p1_raw_rgb=raw['p1']['rgb'][0, frame, 0].numpy(),
        p1_raw_feat=raw['p1']['feat'][0, frame, 0].numpy(),
        p1_normalized=normalized['p1'][0, frame].numpy(),
        p2_feedback_norm=feedback_norm,
        p1_correction=corr,
    )
    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f'DONE: {output}')


if __name__ == '__main__':
    main()
