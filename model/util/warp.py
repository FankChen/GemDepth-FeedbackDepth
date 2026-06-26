# Differentiable geometric/photometric warping utilities for the error-map DPT head.
#
# Given a target frame's depth, the camera intrinsics/extrinsics estimated by the GEM
# module, and neighbouring frames, we inverse-warp the neighbours onto the target frame
# and measure a per-pixel photometric residual. This residual ("error map") highlights
# regions where the current depth estimate is geometrically inconsistent with the video,
# and is fed forward into the next DPT fusion stage.

import torch
import torch.nn.functional as F


def scale_intrinsics(K, src_hw, dst_hw):
    """Scale a (B,T,3,3) intrinsic matrix from ``src_hw`` to ``dst_hw`` resolution.

    Args:
        K: (B,T,3,3) intrinsics defined at resolution ``src_hw = (H0, W0)``.
        src_hw: (H0, W0) the resolution the intrinsics were defined at.
        dst_hw: (h, w) the target feature resolution.
    Returns:
        (B,T,3,3) scaled intrinsics.
    """
    H0, W0 = src_hw
    h, w = dst_hw
    sx = float(w) / float(W0)
    sy = float(h) / float(H0)
    K2 = K.clone()
    K2[..., 0, 0] = K2[..., 0, 0] * sx
    K2[..., 0, 2] = K2[..., 0, 2] * sx
    K2[..., 1, 1] = K2[..., 1, 1] * sy
    K2[..., 1, 2] = K2[..., 1, 2] * sy
    return K2


def _inverse_warp(depth_t, img_s, inv_K_t, K_s, ext_t, ext_s, eps=1e-6):
    """Inverse-warp ``img_s`` onto the target frame using the target depth.

    All tensors are flattened over (B*n) into the first dim.

    Args:
        depth_t: (N,1,H,W) positive metric depth of the target frame.
        img_s:   (N,C,H,W) source frame image (the neighbour to warp).
        inv_K_t: (N,3,3) inverse intrinsics of the target frame at (H,W).
        K_s:     (N,3,3) intrinsics of the source frame at (H,W).
        ext_t:   (N,4,4) world->camera extrinsics of the target frame (OpenCV).
        ext_s:   (N,4,4) world->camera extrinsics of the source frame (OpenCV).
    Returns:
        warped: (N,C,H,W) source image sampled at target pixels.
        valid:  (N,1,H,W) float mask of pixels with a valid correspondence.
    """
    N, _, H, W = depth_t.shape
    device = depth_t.device
    dtype = depth_t.dtype

    vv, uu = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij',
    )
    ones = torch.ones_like(uu)
    pix = torch.stack([uu, vv, ones], dim=0).reshape(3, -1)  # (3, H*W)
    pix = pix.unsqueeze(0).expand(N, -1, -1)                 # (N,3,H*W)

    d = depth_t.reshape(N, 1, H * W)
    # Backproject target pixels to target camera coordinates.
    cam_t = torch.bmm(inv_K_t, pix) * d                      # (N,3,H*W)
    cam_t_h = torch.cat([cam_t, torch.ones(N, 1, H * W, device=device, dtype=dtype)], dim=1)

    # Relative transform cam_t -> cam_s : ext_s @ inv(ext_t).
    M = torch.bmm(ext_s, torch.inverse(ext_t))               # (N,4,4)
    cam_s = torch.bmm(M, cam_t_h)[:, :3]                      # (N,3,H*W)

    proj = torch.bmm(K_s, cam_s)                             # (N,3,H*W)
    z = proj[:, 2:3]
    z_safe = torch.where(z.abs() < eps, torch.full_like(z, eps), z)
    u_s = proj[:, 0:1] / z_safe
    v_s = proj[:, 1:2] / z_safe

    gx = 2.0 * u_s / (W - 1) - 1.0
    gy = 2.0 * v_s / (H - 1) - 1.0
    grid = torch.cat([gx, gy], dim=1).permute(0, 2, 1).reshape(N, H, W, 2)

    warped = F.grid_sample(img_s, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    in_bounds = (gx.abs() <= 1.0) & (gy.abs() <= 1.0) & (z > eps)
    valid = (in_bounds.reshape(N, 1, H, W) & (depth_t > 0)).to(dtype)
    return warped, valid


def signal_error_map(signal, depth, K, extrinsics, offsets=(-1, 1), eps=1e-6, big=1.0,
                     capture=None, tag=''):
    """Generic temporal-reprojection error map for an arbitrary per-pixel signal.

    Warps each neighbour's ``signal`` into the target frame using the (differentiable)
    depth + camera geometry, then measures the residual against the target signal. The
    signal can be raw RGB (photometric error), decoder features (feature error), HOG
    descriptors, etc. — the only difference between experiment arms is what is fed here.

    Uses the minimum-reprojection trick (monodepth2): for every target frame we warp
    each available neighbour and keep, per pixel, the smallest residual.

    Args:
        signal: (B,T,C,H,W) per-pixel signal at the current stage resolution.
        depth:  (B,T,1,H,W) positive metric depth of every frame.
        K:      (B,T,3,3) intrinsics at (H,W).
        extrinsics: (B,T,4,4) world->camera extrinsics (OpenCV).
        offsets: temporal neighbour offsets to warp from.
        capture: optional list. When provided, one record per offset is appended with the
            (detached, CPU) target / warped / error / valid tensors so a debug script can
            visualise every warp. Has no effect on the returned values or on training.
        tag: string label attached to each captured record (e.g. ``'s4/rgb'``).
    Returns:
        err:   (B,T,1,H,W) error (0 where no valid neighbour).
        valid: (B,T,1,H,W) float mask of frames/pixels with a valid neighbour.
    """
    B, T, C, H, W = signal.shape
    device = signal.device
    dtype = signal.dtype

    K = K.to(dtype)
    extrinsics = extrinsics.to(dtype)
    inv_K = torch.inverse(K)

    err_stack = []
    valid_stack = []
    for o in offsets:
        t0 = max(0, -o)
        t1 = min(T, T - o)
        if t1 <= t0:
            continue
        idx_t = torch.arange(t0, t1, device=device)
        idx_s = idx_t + o
        n = int(idx_t.numel())
        N = B * n

        sig_t = signal[:, idx_t].reshape(N, C, H, W)
        sig_s = signal[:, idx_s].reshape(N, C, H, W)
        d_t = depth[:, idx_t].reshape(N, 1, H, W)
        invKt = inv_K[:, idx_t].reshape(N, 3, 3)
        Ks = K[:, idx_s].reshape(N, 3, 3)
        Tt = extrinsics[:, idx_t].reshape(N, 4, 4)
        Ts = extrinsics[:, idx_s].reshape(N, 4, 4)

        warped, valid = _inverse_warp(d_t, sig_s, invKt, Ks, Tt, Ts, eps=eps)
        residual = (sig_t - warped).abs().mean(dim=1, keepdim=True)  # (N,1,H,W)

        if capture is not None:
            capture.append({
                'tag': tag,
                'offset': int(o),
                'idx_t': idx_t.detach().cpu(),
                'idx_s': idx_s.detach().cpu(),
                'target': sig_t.detach().reshape(B, n, C, H, W).cpu(),
                'source': sig_s.detach().reshape(B, n, C, H, W).cpu(),
                'warped': warped.detach().reshape(B, n, C, H, W).cpu(),
                'error': (residual * valid).detach().reshape(B, n, 1, H, W).cpu(),
                'valid': valid.detach().reshape(B, n, 1, H, W).cpu(),
            })

        residual = residual * valid + big * (1.0 - valid)

        full_err = signal.new_full((B, T, 1, H, W), big)
        full_valid = signal.new_zeros((B, T, 1, H, W))
        full_err[:, idx_t] = residual.reshape(B, n, 1, H, W)
        full_valid[:, idx_t] = valid.reshape(B, n, 1, H, W)
        err_stack.append(full_err)
        valid_stack.append(full_valid)

    if len(err_stack) == 0:
        zeros = signal.new_zeros((B, T, 1, H, W))
        return zeros, zeros

    err = torch.stack(err_stack, dim=0).min(dim=0).values  # (B,T,1,H,W)
    valid = torch.stack(valid_stack, dim=0).max(dim=0).values
    err = err * valid
    return err, valid


def photometric_error_map(images, depth, K, extrinsics, offsets=(-1, 1), eps=1e-6, big=1.0,
                          capture=None, tag=''):
    """Photometric (RGB) reprojection error map. Thin wrapper over :func:`signal_error_map`.

    Kept for backward compatibility with the v1 error-map head.
    """
    return signal_error_map(images, depth, K, extrinsics, offsets=offsets, eps=eps, big=big,
                            capture=capture, tag=tag)
