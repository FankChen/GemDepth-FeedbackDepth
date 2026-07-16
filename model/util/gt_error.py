"""GT-camera error signals for multiscale depth refinement.

The functions in this module use *ground-truth camera intrinsics/extrinsics* but
never ground-truth depth as model input.  Metric depth is predicted by the model.
GT metric depth is used only by the training loss for that prediction branch.

This is an oracle-camera stage: it isolates the depth/error-map design before a
learned pose CNN is introduced.
"""

import torch
import torch.nn.functional as F

from .warp import cosine_error_map, photometric_error_map


def imagenet_denormalize(images: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet-normalized ``(B,T,3,H,W)`` images back to ``[0,1]`` RGB."""
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
    return (images * std + mean).clamp_(0.0, 1.0)


def normalize_error_map(error: torch.Tensor, valid: torch.Tensor,
                        clip: float = 5.0, eps: float = 1e-6,
                        mode: str = 'mean') -> torch.Tensor:
    """Scale a residual for network input.

    ``fixed`` preserves residuals already defined in ``[0,1]`` (RGB L1, cosine
    feature distance, geometric relative error). ``mean`` reproduces the legacy
    per-frame masked-mean normalization for old checkpoints/configs.
    """
    valid = valid.to(error.dtype)
    if mode == 'fixed':
        # RGB L1, cosine feature distance and geometric relative error are all
        # defined in [0,1]. Preserve their absolute confidence instead of
        # amplifying every frame to the same mean magnitude.
        return error.clamp(0.0, 1.0) * valid
    if mode != 'mean':
        raise ValueError(f"Unknown error normalization mode: {mode!r}")
    denom = ((error * valid).sum(dim=(-2, -1), keepdim=True)
             / valid.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0))
    normalized = (error / denom.detach().clamp(min=eps)).clamp(0.0, clip) / clip
    return normalized * valid


def _project_target_to_source(depth_t: torch.Tensor, inv_K_t: torch.Tensor,
                              K_s: torch.Tensor, ext_t: torch.Tensor,
                              ext_s: torch.Tensor, eps: float):
    """Project target-camera pixels into a source camera.

    Returns the source sampling grid, projected source-camera depth, and a basic
    in-bounds/front-facing validity mask.  Extrinsics are world-to-camera.
    """
    n, _, h, w = depth_t.shape
    device, dtype = depth_t.device, depth_t.dtype
    vv, uu = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing='ij',
    )
    pix = torch.stack((uu, vv, torch.ones_like(uu)), dim=0).reshape(3, -1)
    pix = pix.unsqueeze(0).expand(n, -1, -1)

    cam_t = torch.bmm(inv_K_t, pix) * depth_t.reshape(n, 1, -1)
    cam_t_h = torch.cat((cam_t, torch.ones(n, 1, h * w, device=device, dtype=dtype)), dim=1)
    rel_t_to_s = torch.bmm(ext_s, torch.linalg.inv(ext_t))
    cam_s = torch.bmm(rel_t_to_s, cam_t_h)[:, :3]
    proj = torch.bmm(K_s, cam_s)
    z_s = proj[:, 2:3]
    z_safe = z_s.clamp(min=eps)
    u_s = proj[:, 0:1] / z_safe
    v_s = proj[:, 1:2] / z_safe
    gx = 2.0 * u_s / max(w - 1, 1) - 1.0
    gy = 2.0 * v_s / max(h - 1, 1) - 1.0
    grid = torch.cat((gx, gy), dim=1).permute(0, 2, 1).reshape(n, h, w, 2)
    valid = ((gx.abs() <= 1.0) & (gy.abs() <= 1.0) & (z_s > eps))
    valid = valid.reshape(n, 1, h, w) & (depth_t > eps)
    return grid, z_s.reshape(n, 1, h, w), valid


def geometric_depth_error_map(depth: torch.Tensor, K: torch.Tensor,
                              extrinsics: torch.Tensor, offsets=(-1, 1),
                              eps: float = 1e-4,
                              occlusion_rel: float = 0.05,
                              occlusion_abs: float = 0.10):
    """Compute temporal metric-depth consistency error using GT cameras.

    ``depth`` is model-predicted positive metric depth ``(B,T,1,H,W)``. Target
    points are projected into neighbouring frames using GT cameras, then their
    projected source-camera depth is compared with sampled predicted source
    depth.  Points hidden behind a closer source surface are masked as occluded.

    The returned error is a symmetric relative depth difference in ``[0,1]``.
    """
    b, t, _, h, w = depth.shape
    dtype, device = depth.dtype, depth.device
    K = K.to(device=device, dtype=dtype)
    extrinsics = extrinsics.to(device=device, dtype=dtype)
    inv_K = torch.linalg.inv(K)

    error_candidates = []
    valid_candidates = []
    for offset in offsets:
        t0, t1 = max(0, -offset), min(t, t - offset)
        if t1 <= t0:
            continue
        idx_t = torch.arange(t0, t1, device=device)
        idx_s = idx_t + offset
        n_frames = int(idx_t.numel())
        n = b * n_frames

        depth_t = depth[:, idx_t].reshape(n, 1, h, w)
        depth_s = depth[:, idx_s].reshape(n, 1, h, w)
        grid, projected_z, valid = _project_target_to_source(
            depth_t,
            inv_K[:, idx_t].reshape(n, 3, 3),
            K[:, idx_s].reshape(n, 3, 3),
            extrinsics[:, idx_t].reshape(n, 4, 4),
            extrinsics[:, idx_s].reshape(n, 4, 4),
            eps,
        )
        sampled_source = F.grid_sample(
            depth_s, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        valid = valid & (sampled_source > eps)
        # A projected point substantially behind the observed source surface is occluded.
        visible = projected_z <= (sampled_source * (1.0 + occlusion_rel) + occlusion_abs)
        valid = valid & visible
        rel_error = (projected_z - sampled_source).abs() / (
            projected_z.abs() + sampled_source.abs() + eps)
        rel_error = rel_error.clamp(0.0, 1.0)

        full_error = depth.new_ones((b, t, 1, h, w))
        full_valid = torch.zeros((b, t, 1, h, w), device=device, dtype=torch.bool)
        full_error[:, idx_t] = rel_error.reshape(b, n_frames, 1, h, w)
        full_valid[:, idx_t] = valid.reshape(b, n_frames, 1, h, w)
        error_candidates.append(full_error)
        valid_candidates.append(full_valid)

    if not error_candidates:
        zeros = depth.new_zeros((b, t, 1, h, w))
        return zeros, zeros

    stacked_error = torch.stack(error_candidates, dim=0)
    stacked_valid = torch.stack(valid_candidates, dim=0)
    # Invalid candidates cannot win minimum reprojection.
    stacked_error = torch.where(stacked_valid, stacked_error, torch.ones_like(stacked_error))
    error = stacked_error.min(dim=0).values
    valid = stacked_valid.any(dim=0)
    return error * valid.to(dtype), valid.to(dtype)


def photometric_signal_error(signal: torch.Tensor, metric_depth: torch.Tensor,
                             K: torch.Tensor, extrinsics: torch.Tensor,
                             offsets=(-1, 1)):
    """Thin typed wrapper around the generic warp residual implementation."""
    return photometric_error_map(
        signal, metric_depth, K, extrinsics, offsets=tuple(offsets))


def feature_cosine_signal_error(signal: torch.Tensor, metric_depth: torch.Tensor,
                                K: torch.Tensor, extrinsics: torch.Tensor,
                                offsets=(-1, 1)):
    """Fixed-range cosine residual for dense decoder features."""
    return cosine_error_map(
        signal, metric_depth, K, extrinsics, offsets=tuple(offsets))
