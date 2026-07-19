#!/usr/bin/env python3
"""Minimal causal validation of the multi-level error-map method.

Gate 1 (no training): for p4/p3/p2/p1, calibrate the *same* DPT prediction d_s
with a detached GT clip-level affine transform, convert it to the exact metric
Z_s passed to warp, and compare GT / predicted / locally perturbed depth warps.

Gate 2 (100 tiny steps by default): freeze the model and overfit an error-gated
correction on one deliberately corrupted p2 depth. The trained corrector is then
evaluated with normal, zero and spatially shifted errors. This tests whether the
error signal carries causal spatial information; it is not a model benchmark.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.dataset_mix import DepthVideoDataset
from loss.videoloss import compute_scale_and_shift
from model.factory import build_gemdepth_from_config
from model.util.gt_error import imagenet_denormalize
from model.util.warp_v2 import scale_intrinsics_v2, temporal_signal_error_v2


STAGES = ('p4', 'p3', 'p2', 'p1')
CONDITIONS = ('gt', 'predicted', 'perturbed')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        return payload['model_state_dict'], int(payload.get('total_step', -1))
    return payload, -1


def load_v2_model(config, checkpoint, device):
    model = build_gemdepth_from_config(config, load_backbone_pretrained=False)
    state, step = load_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = (
        'head.metric_depth_heads.',
        'head.error_encoders.',
        'head.final_error_correction.',
    )
    invalid_missing = [name for name in missing if not name.startswith(allowed_missing)]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f'Checkpoint is not a compatible temporal/v2 model: '
            f'missing={invalid_missing[:20]} unexpected={list(unexpected)[:20]}')
    print(f'[load] step={step} missing_v2={len(missing)} unexpected=0')
    model = model.to(device).eval()
    model.head.capture_warps = True
    return model, step


def resize_bt(value, size, mode):
    batch, frames = value.shape[:2]
    resized = F.interpolate(
        value.flatten(0, 1), size=size, mode=mode,
        align_corners=False if mode in ('bilinear', 'bicubic') else None)
    return resized.unflatten(0, (batch, frames))


def calibrate_stage_depth(raw_full, gt_full, mask_full, native_size,
                          min_depth=1e-3, max_depth=100.0):
    """Return native d_s, aligned inverse depth and the exact Z_s used by warp."""
    raw = resize_bt(raw_full.float(), native_size, 'bilinear')
    gt = resize_bt(gt_full.float(), native_size, 'nearest')
    valid = resize_bt(mask_full.float(), native_size, 'nearest')
    valid = valid * (gt > min_depth).float() * (gt <= max_depth).float()
    target_inverse = torch.zeros_like(gt)
    positive = valid > 0.5
    target_inverse[positive] = 1.0 / gt[positive]
    batch = raw.shape[0]
    with torch.no_grad():
        scale, shift = compute_scale_and_shift(
            raw.squeeze(2).flatten(1, 2),
            target_inverse.squeeze(2).flatten(1, 2),
            valid.squeeze(2).flatten(1, 2))
    aligned_inverse = scale[:, None, None, None, None] * raw + shift[:, None, None, None, None]
    bounded_inverse = aligned_inverse.clamp(min=1.0 / max_depth, max=1.0 / min_depth)
    metric = 1.0 / bounded_inverse
    return {
        'raw_inverse': raw,
        'aligned_inverse': aligned_inverse,
        'metric': metric,
        'gt_metric': gt.clamp(min=min_depth, max=max_depth),
        'valid_gt': valid,
        'scale': scale,
        'shift': shift,
        'floor_fraction': float((metric <= min_depth * 1.01).float().mean()),
        'ceiling_fraction': float((metric >= max_depth * 0.99).float().mean()),
    }


def make_perturbation(metric, frame_index):
    corrupted = metric.clone()
    height, width = metric.shape[-2:]
    y0, y1 = height // 4, 3 * height // 4
    x0, x1 = width // 4, 3 * width // 4
    region = torch.zeros_like(metric, dtype=torch.bool)
    region[:, frame_index, :, y0:y1, x0:x1] = True
    corrupted[region] = corrupted[region] * 0.5
    return corrupted, region


def stage_signals(images, feature, batch, frames, native_size):
    rgb = imagenet_denormalize(images.detach().float())
    rgb = resize_bt(rgb, native_size, 'bilinear')
    feat = feature.detach().reshape(batch, frames, feature.shape[1], *native_size).float()
    return rgb, feat


def compute_errors(rgb, feature, metric, K, poses, offsets, border_margin,
                   occlusion_rel, occlusion_abs, diagnostics=True):
    rgb_result = temporal_signal_error_v2(
        rgb, metric, K, poses, offsets=offsets, distance='rgb_l1',
        border_margin=border_margin, occlusion_rel=occlusion_rel,
        occlusion_abs=occlusion_abs, return_diagnostics=diagnostics)
    if diagnostics:
        rgb_error, rgb_valid, visual = rgb_result
    else:
        rgb_error, rgb_valid = rgb_result
        visual = None
    feature_error, feature_valid = temporal_signal_error_v2(
        feature, metric, K, poses, offsets=offsets, distance='feature_cosine',
        border_margin=border_margin, occlusion_rel=occlusion_rel,
        occlusion_abs=occlusion_abs)
    valid = torch.minimum(rgb_valid, feature_valid)
    return {
        'rgb_error': rgb_error * valid,
        'feature_error': feature_error * valid,
        'valid': valid,
        'visual': visual,
    }


def masked_mean(value, mask):
    return float((value * mask).sum() / mask.sum().clamp(min=1.0))


def percentile(values, q=99.0, floor=1e-6):
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    return max(float(np.percentile(finite, q)), floor) if finite.size else floor


def show(axis, image, title, cmap=None, vmin=None, vmax=None):
    axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=7)
    axis.axis('off')


def save_gate1_figures(results, frame, output):
    raw_values = np.concatenate([
        results[stage]['calibration']['raw_inverse'][0, frame, 0].cpu().numpy().reshape(-1)
        for stage in STAGES])
    raw_limit = percentile(raw_values)
    for condition in CONDITIONS:
        figure, axes = plt.subplots(4, 11, figsize=(33, 12))
        for row, stage in enumerate(STAGES):
            item = results[stage]
            calibration = item['calibration']
            condition_data = item['conditions'][condition]
            visual = condition_data['visual']
            selected_offset = visual['selected_offset'][0, frame, 0].cpu().numpy()
            metric = condition_data['metric'][0, frame, 0].cpu().numpy()
            show(axes[row, 0], visual['target'][0, frame].permute(1, 2, 0).cpu().numpy(),
                 f'{stage} target RGB')
            show(axes[row, 1], visual['source'][0, frame].permute(1, 2, 0).cpu().numpy(),
                 'selected source RGB')
            show(axes[row, 2], calibration['raw_inverse'][0, frame, 0].cpu().numpy(),
                 'raw SSI score d_s', 'viridis', 0, raw_limit)
            show(axes[row, 3], calibration['aligned_inverse'][0, frame, 0].cpu().numpy(),
                 f"aligned inv\na={float(calibration['scale'][0]):.3g} "
                 f"b={float(calibration['shift'][0]):.3g}", 'viridis', 0, 1.0)
            show(axes[row, 4], metric, 'EXACT Z used by warp (m)', 'magma', 1e-3, 100)
            show(axes[row, 5], calibration['gt_metric'][0, frame, 0].cpu().numpy(),
                 'GT Z (m)', 'magma', 1e-3, 100)
            show(axes[row, 6], visual['warped'][0, frame].permute(1, 2, 0).cpu().numpy(),
                 'selected warped RGB')
            show(axes[row, 7], condition_data['rgb_error'][0, frame, 0].cpu().numpy(),
                 'RGB error [0,1]', 'inferno', 0, 1)
            show(axes[row, 8], condition_data['feature_error'][0, frame, 0].cpu().numpy(),
                 'Feature error [0,1]', 'inferno', 0, 1)
            show(axes[row, 9], condition_data['valid'][0, frame, 0].cpu().numpy(),
                 'validity', 'gray', 0, 1)
            show(axes[row, 10], selected_offset, 'selected offset', 'coolwarm', -1, 1)
        figure.suptitle(
            f'Gate 1 | condition={condition} | displayed d_s -> aligned inverse -> '
            f'EXACT same Z passed to warp', fontsize=12)
        figure.tight_layout()
        path = output / f'gate1_{condition}.png'
        figure.savefig(path, dpi=170, bbox_inches='tight')
        plt.close(figure)
        print(f'wrote {path}')


class ErrorGatedCorrector(nn.Module):
    """Tiny diagnostic corrector: zero residual guarantees exact zero output."""

    def __init__(self, feature_channels):
        super().__init__()
        self.content = nn.Sequential(
            nn.Conv2d(feature_channels + 3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        nn.init.zeros_(self.content[-1].weight)
        nn.init.zeros_(self.content[-1].bias)

    def forward(self, feature, error_channels):
        gate = error_channels[:, :2].amax(dim=1, keepdim=True)
        gate = gate * error_channels[:, 2:3]
        return gate * self.content(torch.cat((feature, error_channels), dim=1))


def run_gate2(stage_item, frame, steps, learning_rate, output):
    calibration = stage_item['calibration']
    predicted = stage_item['conditions']['predicted']
    perturbed = stage_item['conditions']['perturbed']
    feature = stage_item['feature'][:, frame].detach()
    valid = perturbed['valid'][:, frame].detach()
    errors = torch.cat((
        perturbed['rgb_error'][:, frame],
        perturbed['feature_error'][:, frame],
        valid,
    ), dim=1).detach()
    region = stage_item['perturb_region'][:, frame].float().detach()
    original_inverse = 1.0 / calibration['metric'][:, frame].detach()
    corrupted_inverse = 1.0 / perturbed['metric'][:, frame].detach()
    target_correction = original_inverse - corrupted_inverse

    torch.manual_seed(20260717)
    model = ErrorGatedCorrector(feature.shape[1]).to(feature.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = 1.0 + 9.0 * region
    history = []
    snapshots = {}
    for step in range(steps + 1):
        correction = model(feature, errors)
        loss = ((correction - target_correction).square() * weights).sum() / weights.sum()
        if step in (0, 10, 50, 100, steps):
            roi_mse = ((correction - target_correction).square() * region).sum() / region.sum().clamp(min=1)
            history.append({'step': step, 'loss': float(loss), 'roi_mse': float(roi_mse)})
            snapshots[step] = correction.detach().cpu()
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    zero_errors = errors.clone()
    zero_errors[:, :2] = 0
    shifted_errors = errors.clone()
    shifted_errors[:, :2] = torch.roll(
        shifted_errors[:, :2], shifts=(errors.shape[-2] // 3, errors.shape[-1] // 3),
        dims=(-2, -1))
    with torch.no_grad():
        outputs = {
            'normal': model(feature, errors),
            'zero': model(feature, zero_errors),
            'shifted': model(feature, shifted_errors),
        }
    metrics = {}
    for name, correction in outputs.items():
        roi_mse = ((correction - target_correction).square() * region).sum() / region.sum().clamp(min=1)
        corrected_inverse = (corrupted_inverse + correction).clamp(min=1.0 / 100, max=1.0 / 1e-3)
        corrected_metric = 1.0 / corrected_inverse
        roi_rel = (((corrected_metric - calibration['metric'][:, frame]).abs()
                    / calibration['metric'][:, frame].clamp(min=1e-3)) * region).sum()
        roi_rel = roi_rel / region.sum().clamp(min=1)
        metrics[name] = {'roi_mse': float(roi_mse), 'roi_relative_depth_error': float(roi_rel)}
    metrics['raw_max_difference_normal_zero'] = float(
        (outputs['normal'] - outputs['zero']).abs().max())
    normal_mse = metrics['normal']['roi_mse']
    metrics['pass'] = bool(
        normal_mse < 0.8 * metrics['zero']['roi_mse']
        and normal_mse < 0.9 * metrics['shifted']['roi_mse']
        and metrics['raw_max_difference_normal_zero'] > 1e-6)
    metrics['history'] = history

    figure, axes = plt.subplots(2, 5, figsize=(16, 6))
    show(axes[0, 0], target_correction[0, 0].cpu().numpy(), 'target inv-depth correction', 'coolwarm')
    show(axes[0, 1], errors[0, 0].cpu().numpy(), 'perturbed RGB error', 'inferno', 0, 1)
    show(axes[0, 2], errors[0, 1].cpu().numpy(), 'perturbed Feature error', 'inferno', 0, 1)
    show(axes[0, 3], errors[0, 2].cpu().numpy(), 'validity', 'gray', 0, 1)
    show(axes[0, 4], region[0, 0].cpu().numpy(), 'corrupted region', 'gray', 0, 1)
    for column, name in enumerate(('normal', 'zero', 'shifted')):
        show(axes[1, column], outputs[name][0, 0].cpu().numpy(),
             f"{name} correction\nROI MSE={metrics[name]['roi_mse']:.3g}", 'coolwarm')
    steps_x = [item['step'] for item in history]
    losses = [item['roi_mse'] for item in history]
    axes[1, 3].plot(steps_x, losses, marker='o')
    axes[1, 3].set_title('normal error ROI MSE', fontsize=8)
    axes[1, 3].set_yscale('log')
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 4].axis('off')
    axes[1, 4].text(
        0.05, 0.8,
        f"PASS={metrics['pass']}\nnormal={metrics['normal']['roi_mse']:.4g}\n"
        f"zero={metrics['zero']['roi_mse']:.4g}\n"
        f"shifted={metrics['shifted']['roi_mse']:.4g}\n"
        f"max diff={metrics['raw_max_difference_normal_zero']:.4g}",
        fontsize=10, va='top')
    figure.suptitle('Gate 2 | frozen model, tiny error-gated correction overfit', fontsize=12)
    figure.tight_layout()
    path = output / 'gate2_causal_overfit.png'
    figure.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(figure)
    print(f'wrote {path}')
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/scratch_ed_gt_error_rgbfeat_v2.yaml')
    parser.add_argument('--checkpoint', required=True,
                        help='A temporal baseline or v2 model/final training checkpoint')
    parser.add_argument('--data_dir', default='/mnt/workspace/vkitti/vkitti')
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--frame_idx', type=int, default=1)
    parser.add_argument('--gate2_stage', choices=STAGES, default='p2')
    parser.add_argument('--gate2_steps', type=int, default=100)
    parser.add_argument('--gate2_lr', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=20260717)
    parser.add_argument('--output', default='results/errmap_minimal_validation/sample0')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    for path in (args.config, args.checkpoint, args.data_dir):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    seed_everything(args.seed)
    config = OmegaConf.load(args.config)
    if str(config.model.head_type) != 'multiscale_gt_error_v2':
        raise ValueError('Minimal validation requires multiscale_gt_error_v2 config')

    crop_size = int(config.dataset.train.crop_size)
    frames = int(config.dataset.train.seq_len)
    heldout = list(OmegaConf.select(config, 'dataset.val.include_scenes', default=['Scene20']))
    dataset = DepthVideoDataset(
        mode='train', data_dirs=[args.data_dir], crop_size=crop_size,
        seq_len=frames, include_scenes=heldout, window_stride=frames)
    dataset.data_paths = list(dataset.vkitti_data_paths)
    transform = dataset.transform.get('vkitti')
    if transform is not None and hasattr(transform, 'set_random_seed'):
        transform.set_random_seed(args.seed)
    sample = dataset[args.sample_idx]

    device = torch.device('cuda')
    images = sample['image'].unsqueeze(0).to(device)
    gt = sample['depth'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)
    K = torch.as_tensor(sample['IntM']).unsqueeze(0).to(device)
    poses = torch.stack([torch.as_tensor(value) for value in sample['poses']])
    poses = poses.unsqueeze(0).to(device)
    model, step = load_v2_model(config, args.checkpoint, device)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        model(images, gt_intrinsics=K, gt_extrinsics=poses)
    head = model.head
    if tuple(head.stage_features) != STAGES or tuple(head.stage_depths) != STAGES:
        raise RuntimeError(
            f'Incomplete stages features={tuple(head.stage_features)} depths={tuple(head.stage_depths)}')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    summary = {
        'checkpoint': args.checkpoint,
        'checkpoint_step': step,
        'sample_idx': args.sample_idx,
        'frame_idx': args.frame_idx,
        'depth_paths': [str(path) for path in sample['path']],
        'gate1': {},
    }

    batch = images.shape[0]
    for stage in STAGES:
        feature_bt = head.stage_features[stage]
        native_size = feature_bt.shape[-2:]
        calibration = calibrate_stage_depth(
            head.stage_depths[stage], gt, mask, native_size,
            min_depth=1e-3, max_depth=100.0)
        rgb, feature = stage_signals(images, feature_bt, batch, frames, native_size)
        K_stage = scale_intrinsics_v2(K.unsqueeze(1).expand(-1, frames, -1, -1),
                                      images.shape[-2:], native_size)
        predicted_metric = calibration['metric']
        perturbed_metric, perturb_region = make_perturbation(
            predicted_metric, args.frame_idx)
        condition_metrics = {
            'gt': calibration['gt_metric'],
            'predicted': predicted_metric,
            'perturbed': perturbed_metric,
        }
        conditions = {}
        for condition, metric in condition_metrics.items():
            conditions[condition] = compute_errors(
                rgb, feature, metric, K_stage, poses,
                offsets=tuple(config.model.warp_offsets),
                border_margin=float(config.model.warp_border_margin),
                occlusion_rel=float(config.model.warp_occlusion_rel),
                occlusion_abs=float(config.model.warp_occlusion_abs),
                diagnostics=True)
            conditions[condition]['metric'] = metric

        frame_region = perturb_region[:, args.frame_idx].float()
        pred_condition = conditions['predicted']
        bad_condition = conditions['perturbed']
        ratios = {}
        for key in ('rgb_error', 'feature_error'):
            pred_roi = masked_mean(pred_condition[key][:, args.frame_idx], frame_region)
            bad_roi = masked_mean(bad_condition[key][:, args.frame_idx], frame_region)
            pred_out = masked_mean(
                pred_condition[key][:, args.frame_idx], 1.0 - frame_region)
            bad_out = masked_mean(
                bad_condition[key][:, args.frame_idx], 1.0 - frame_region)
            ratios[key] = {
                'predicted_roi': pred_roi,
                'perturbed_roi': bad_roi,
                'roi_ratio': bad_roi / max(pred_roi, 1e-8),
                'outside_delta': bad_out - pred_out,
            }
        stage_pass = bool(
            float(conditions['gt']['valid'].mean()) > 0.05
            and max(ratios['rgb_error']['roi_ratio'],
                    ratios['feature_error']['roi_ratio']) > 1.05)
        summary['gate1'][stage] = {
            'native_size': list(native_size),
            'scale': float(calibration['scale'][0]),
            'shift': float(calibration['shift'][0]),
            'metric_min': float(predicted_metric.min()),
            'metric_median': float(predicted_metric.median()),
            'metric_max': float(predicted_metric.max()),
            'floor_fraction': calibration['floor_fraction'],
            'ceiling_fraction': calibration['ceiling_fraction'],
            'gt_valid_fraction': float(conditions['gt']['valid'].mean()),
            'predicted_valid_fraction': float(conditions['predicted']['valid'].mean()),
            'perturbed_valid_fraction': float(conditions['perturbed']['valid'].mean()),
            'perturbation_response': ratios,
            'pass': stage_pass,
        }
        results[stage] = {
            'feature': feature,
            'calibration': calibration,
            'conditions': conditions,
            'perturb_region': perturb_region,
        }

    save_gate1_figures(results, args.frame_idx, output)
    summary['gate1_pass'] = bool(sum(
        int(summary['gate1'][stage]['pass']) for stage in STAGES) >= 3)
    summary['gate2'] = run_gate2(
        results[args.gate2_stage], args.frame_idx,
        args.gate2_steps, args.gate2_lr, output)
    summary['overall_pass'] = bool(
        summary['gate1_pass'] and summary['gate2']['pass'])
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"OVERALL {'PASS' if summary['overall_pass'] else 'FAIL'}: {output}")


if __name__ == '__main__':
    main()
