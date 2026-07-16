"""Pixel-center-consistent temporal warp used by the baseline-anchored DPT v2.

This module is intentionally independent from ``warp.py`` so historical
experiments remain reproducible. All resize/projection/sampling operations use
PyTorch's ``align_corners=False`` pixel-center convention.
"""

import torch
import torch.nn.functional as F


def scale_intrinsics_v2(K, src_hw, dst_hw):
    """Scale intrinsics between pixel-center grids (align_corners=False)."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    scaled = K.clone()
    scaled[..., 0, 0] = K[..., 0, 0] * sx
    scaled[..., 1, 1] = K[..., 1, 1] * sy
    scaled[..., 0, 2] = (K[..., 0, 2] + 0.5) * sx - 0.5
    scaled[..., 1, 2] = (K[..., 1, 2] + 0.5) * sy - 0.5
    return scaled


def _project_target_to_source_v2(depth_t, inv_K_t, K_s, ext_t, ext_s,
                                 border_margin=1.0, eps=1e-6):
    """Project target pixels into a source frame using world-to-camera poses."""
    n, _, height, width = depth_t.shape
    device, dtype = depth_t.device, depth_t.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing='ij',
    )
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=0)
    pixels = pixels.reshape(3, -1).unsqueeze(0).expand(n, -1, -1)

    camera_t = torch.bmm(inv_K_t, pixels) * depth_t.reshape(n, 1, -1)
    camera_t_h = torch.cat((
        camera_t,
        torch.ones(n, 1, height * width, device=device, dtype=dtype),
    ), dim=1)
    relative = torch.bmm(ext_s, torch.linalg.inv(ext_t))
    camera_s = torch.bmm(relative, camera_t_h)[:, :3]
    projected = torch.bmm(K_s, camera_s)
    z_source = camera_s[:, 2:3]
    safe_z = z_source.clamp(min=eps)
    u_source = projected[:, 0:1] / safe_z
    v_source = projected[:, 1:2] / safe_z

    # Pixel-center normalization required by grid_sample(..., align_corners=False).
    grid_x = 2.0 * (u_source + 0.5) / float(width) - 1.0
    grid_y = 2.0 * (v_source + 0.5) / float(height) - 1.0
    grid = torch.cat((grid_x, grid_y), dim=1)
    grid = grid.permute(0, 2, 1).reshape(n, height, width, 2)

    margin = float(border_margin)
    valid = (
        (z_source > eps)
        & (u_source >= margin)
        & (u_source <= (width - 1 - margin))
        & (v_source >= margin)
        & (v_source <= (height - 1 - margin))
    )
    valid = valid.reshape(n, 1, height, width) & (depth_t > eps)
    return grid, z_source.reshape(n, 1, height, width), valid


def _signal_distance(target, warped, kind, eps):
    if kind == 'rgb_l1':
        return (target - warped).abs().mean(dim=1, keepdim=True).clamp(0.0, 1.0)
    if kind == 'feature_cosine':
        target_unit = F.normalize(target, p=2, dim=1, eps=eps)
        warped_unit = F.normalize(warped, p=2, dim=1, eps=eps)
        cosine = (target_unit * warped_unit).sum(dim=1, keepdim=True)
        return ((1.0 - cosine.clamp(-1.0, 1.0)) * 0.5).clamp(0.0, 1.0)
    raise ValueError(f'Unknown signal distance: {kind!r}')


def temporal_signal_error_v2(signal, metric_depth, K, extrinsics, offsets=(-1, 1),
                             distance='rgb_l1', border_margin=1.0,
                             occlusion_rel=0.05, occlusion_abs=0.10,
                             eps=1e-6, return_diagnostics=False):
    """Warp temporal signals and return min-reprojection residual in ``[0,1]``.

    Source metric depth supplies an occlusion test for *all* signal modalities.
    Invalid candidates cannot win minimum reprojection. Diagnostics contain the
    target and pixel-wise selected warped signal at exactly this stage size.
    """
    batch, frames, channels, height, width = signal.shape
    if metric_depth.shape != (batch, frames, 1, height, width):
        raise ValueError(
            f'Expected metric depth {(batch, frames, 1, height, width)}, '
            f'got {tuple(metric_depth.shape)}')
    if K.shape != (batch, frames, 3, 3):
        raise ValueError(f'Expected K {(batch, frames, 3, 3)}, got {tuple(K.shape)}')
    if extrinsics.shape != (batch, frames, 4, 4):
        raise ValueError(
            f'Expected extrinsics {(batch, frames, 4, 4)}, got {tuple(extrinsics.shape)}')

    dtype, device = signal.dtype, signal.device
    K = K.to(device=device, dtype=dtype)
    extrinsics = extrinsics.to(device=device, dtype=dtype)
    metric_depth = metric_depth.to(device=device, dtype=dtype)
    inv_K = torch.linalg.inv(K)

    candidate_errors = []
    candidate_valids = []
    candidate_warped = []
    candidate_offsets = []
    for offset in tuple(int(value) for value in offsets):
        if offset == 0:
            raise ValueError('Temporal offsets must be non-zero')
        start, end = max(0, -offset), min(frames, frames - offset)
        if end <= start:
            continue
        target_indices = torch.arange(start, end, device=device)
        source_indices = target_indices + offset
        count = int(target_indices.numel())
        flattened = batch * count

        target = signal[:, target_indices].reshape(flattened, channels, height, width)
        source = signal[:, source_indices].reshape(flattened, channels, height, width)
        target_depth = metric_depth[:, target_indices].reshape(flattened, 1, height, width)
        source_depth = metric_depth[:, source_indices].reshape(flattened, 1, height, width)
        grid, projected_z, valid = _project_target_to_source_v2(
            target_depth,
            inv_K[:, target_indices].reshape(flattened, 3, 3),
            K[:, source_indices].reshape(flattened, 3, 3),
            extrinsics[:, target_indices].reshape(flattened, 4, 4),
            extrinsics[:, source_indices].reshape(flattened, 4, 4),
            border_margin=border_margin,
            eps=eps,
        )
        warped = F.grid_sample(
            source, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled_source_depth = F.grid_sample(
            source_depth, grid, mode='bilinear', padding_mode='zeros',
            align_corners=False)
        valid = valid & (sampled_source_depth > eps)
        visible = projected_z <= (
            sampled_source_depth * (1.0 + occlusion_rel) + occlusion_abs)
        valid = valid & visible
        error = _signal_distance(target, warped, distance, eps)

        full_error = signal.new_full((batch, frames, 1, height, width), float('inf'))
        full_valid = torch.zeros(
            (batch, frames, 1, height, width), device=device, dtype=torch.bool)
        full_error[:, target_indices] = torch.where(
            valid, error, torch.full_like(error, float('inf'))
        ).reshape(batch, count, 1, height, width)
        full_valid[:, target_indices] = valid.reshape(batch, count, 1, height, width)
        candidate_errors.append(full_error)
        candidate_valids.append(full_valid)
        if return_diagnostics:
            full_warped = signal.new_zeros(
                (batch, frames, channels, height, width))
            full_offset = torch.zeros(
                (batch, frames, 1, height, width), device=device,
                dtype=torch.int16)
            full_warped[:, target_indices] = warped.reshape(
                batch, count, channels, height, width)
            full_offset[:, target_indices] = torch.full(
                (batch, count, 1, height, width), offset,
                device=device, dtype=torch.int16)
            candidate_warped.append(full_warped)
            candidate_offsets.append(full_offset)

    if not candidate_errors:
        zeros = signal.new_zeros((batch, frames, 1, height, width))
        diagnostics = {
            'target': signal.detach(),
            'warped': signal.new_zeros(signal.shape).detach(),
            'selected_offset': torch.zeros_like(zeros, dtype=torch.int16),
        }
        return (zeros, zeros, diagnostics) if return_diagnostics else (zeros, zeros)

    errors = torch.stack(candidate_errors, dim=0)
    valids = torch.stack(candidate_valids, dim=0)
    best_error, winner = errors.min(dim=0)
    valid = valids.any(dim=0)
    best_error = torch.where(valid, best_error, torch.zeros_like(best_error))

    diagnostics = None
    if return_diagnostics:
        warped_candidates = torch.stack(candidate_warped, dim=0)
        offset_candidates = torch.stack(candidate_offsets, dim=0)
        warped_index = winner.unsqueeze(0).expand(
            1, batch, frames, channels, height, width)
        selected_warped = torch.gather(warped_candidates, 0, warped_index).squeeze(0)
        offset_index = winner.unsqueeze(0)
        selected_offset = torch.gather(offset_candidates, 0, offset_index).squeeze(0)
        selected_warped = selected_warped * valid.to(dtype)
        diagnostics = {
            'target': signal.detach(),
            'warped': selected_warped.detach(),
            'selected_offset': selected_offset.detach(),
        }
    if return_diagnostics:
        return best_error, valid.to(dtype), diagnostics
    return best_error, valid.to(dtype)


def temporal_depth_error_v2(metric_depth, K, extrinsics, offsets=(-1, 1),
                            border_margin=1.0, occlusion_rel=0.05,
                            occlusion_abs=0.10, eps=1e-6):
    """Fixed-range geometric consistency residual for metric depth."""
    batch, frames, _, height, width = metric_depth.shape
    dtype, device = metric_depth.dtype, metric_depth.device
    K = K.to(device=device, dtype=dtype)
    extrinsics = extrinsics.to(device=device, dtype=dtype)
    inv_K = torch.linalg.inv(K)
    candidates, valid_candidates = [], []

    for offset in tuple(int(value) for value in offsets):
        start, end = max(0, -offset), min(frames, frames - offset)
        if end <= start:
            continue
        target_indices = torch.arange(start, end, device=device)
        source_indices = target_indices + offset
        count = int(target_indices.numel())
        flattened = batch * count
        target_depth = metric_depth[:, target_indices].reshape(flattened, 1, height, width)
        source_depth = metric_depth[:, source_indices].reshape(flattened, 1, height, width)
        grid, projected_z, valid = _project_target_to_source_v2(
            target_depth,
            inv_K[:, target_indices].reshape(flattened, 3, 3),
            K[:, source_indices].reshape(flattened, 3, 3),
            extrinsics[:, target_indices].reshape(flattened, 4, 4),
            extrinsics[:, source_indices].reshape(flattened, 4, 4),
            border_margin=border_margin,
            eps=eps,
        )
        sampled = F.grid_sample(
            source_depth, grid, mode='bilinear', padding_mode='zeros',
            align_corners=False)
        valid = valid & (sampled > eps)
        valid = valid & (
            projected_z <= sampled * (1.0 + occlusion_rel) + occlusion_abs)
        error = (projected_z - sampled).abs() / (
            projected_z.abs() + sampled.abs() + eps)
        error = error.clamp(0.0, 1.0)

        full_error = metric_depth.new_full(
            (batch, frames, 1, height, width), float('inf'))
        full_valid = torch.zeros(
            (batch, frames, 1, height, width), device=device, dtype=torch.bool)
        full_error[:, target_indices] = torch.where(
            valid, error, torch.full_like(error, float('inf'))
        ).reshape(batch, count, 1, height, width)
        full_valid[:, target_indices] = valid.reshape(
            batch, count, 1, height, width)
        candidates.append(full_error)
        valid_candidates.append(full_valid)

    if not candidates:
        zeros = metric_depth.new_zeros((batch, frames, 1, height, width))
        return zeros, zeros
    stacked = torch.stack(candidates, dim=0)
    valid = torch.stack(valid_candidates, dim=0).any(dim=0)
    error = stacked.min(dim=0).values
    error = torch.where(valid, error, torch.zeros_like(error))
    return error, valid.to(dtype)
