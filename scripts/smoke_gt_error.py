#!/usr/bin/env python3
"""GPU smoke for all GT-camera error-channel arms.

Runs the *full* GemDepth forward and one backward at a small resolution without
loading/downloading pretrained weights.  It checks shape/finite values, baseline
initialization, metric-depth loss, gradients, and equal parameter counts.
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.factory import build_gemdepth_from_config  # pyright: ignore[reportMissingImports]
from train import compute_aux_depth_loss_disp, compute_metric_depth_loss  # pyright: ignore[reportMissingImports]


def make_geometry(batch, frames, size, device):
    K = torch.tensor(
        [[0.9 * size, 0.0, (size - 1) / 2],
         [0.0, 0.9 * size, (size - 1) / 2],
         [0.0, 0.0, 1.0]], device=device).unsqueeze(0).repeat(batch, 1, 1)
    ext = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch, frames, 1, 1)
    for i in range(frames):
        ext[:, i, 0, 3] = 0.05 * i
    return K, ext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=112)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required (temporal motion modules use CUDA attention kernels)')
    if args.size % 14:
        raise ValueError('--size must be divisible by DINOv2 patch size 14')

    device = torch.device('cuda')
    signals = ('rgb', 'feat', 'rgbfeat', 'geom')
    counts = []
    for signal in signals:
        cfg = OmegaConf.load(f'config/scratch_ed_gt_error_{signal}.yaml')
        model = build_gemdepth_from_config(cfg, load_backbone_pretrained=False).to(device).train()
        batch, frames = 1, 4
        images = torch.randn(batch, frames, 3, args.size, args.size, device=device)
        depth_gt = torch.rand(batch, frames, 1, args.size, args.size, device=device) * 40 + 2
        mask = torch.ones_like(depth_gt)
        K, ext = make_geometry(batch, frames, args.size, device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            depth, _, _, _ = model(images, gt_intrinsics=K, gt_extrinsics=ext)
        if depth.shape != (batch, frames, args.size, args.size):
            raise AssertionError((signal, depth.shape))
        if not torch.isfinite(depth).all():
            raise FloatingPointError(f'{signal}: non-finite output')
        head = model.head
        aux = compute_aux_depth_loss_disp(head.aux_depths, depth_gt, mask)
        metric = compute_metric_depth_loss(head.metric_depths, depth_gt, mask)

        # Metric-depth supervision is intentionally isolated from the main
        # inverse-depth feature path (metric heads consume detached p2/p1).
        model.zero_grad(set_to_none=True)
        metric.backward(retain_graph=True)
        leaked = [name for name, param in head.named_parameters()
                  if not name.startswith('metric_depth_heads.')
                  and param.grad is not None and float(param.grad.abs().max()) > 0]
        if leaked:
            raise AssertionError(f'{signal}: metric loss leaked outside metric heads: {leaked[:10]}')
        model.zero_grad(set_to_none=True)
        loss = depth.float().mean() + aux + 0.1 * metric
        loss.backward()

        grad_names = ('p1_correction.4.weight', 'error_encoders.p2.2.weight',
                      'metric_depth_heads.p1.4.weight')
        grads = {}
        for name, param in head.named_parameters():
            if name in grad_names:
                grads[name] = 0.0 if param.grad is None else param.grad.abs().max().item()
        if set(grads) != set(grad_names) or not all(v > 0 for v in grads.values()):
            raise AssertionError(f'{signal}: missing/zero branch gradients: {grads}')
        count = sum(p.numel() for p in head.parameters())
        counts.append(count)
        valid = {k: float(v.mean()) for k, v in head.valid_maps.items()}
        print(f'{signal:7s} OK shape={tuple(depth.shape)} loss={loss.item():.4f} '
              f'grads={grads} valid={valid} params={count}')
        del model, depth, loss
        torch.cuda.empty_cache()

    if len(set(counts)) != 1:
        raise AssertionError(f'Channel arms have unequal head sizes: {counts}')
    print('ALL GT-ERROR GPU SMOKES PASSED')


if __name__ == '__main__':
    main()
