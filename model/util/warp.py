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


def photometric_error_map(images, depth, K, extrinsics, offsets=(-1, 1), eps=1e-6, big=1.0):
    """Compute a per-pixel photometric error map via inverse warping of neighbour frames.

    Uses the minimum-reprojection trick (monodepth2): for every target frame we warp
    each available neighbour and keep, per pixel, the smallest photometric residual.

    Args:
        images: (B,T,C,H,W) frames at the current stage resolution.
        depth:  (B,T,1,H,W) positive metric depth of every frame.
        K:      (B,T,3,3) intrinsics at (H,W).
        extrinsics: (B,T,4,4) world->camera extrinsics (OpenCV).
        offsets: temporal neighbour offsets to warp from.
    Returns:
        err:   (B,T,1,H,W) photometric error (0 where no valid neighbour).
        valid: (B,T,1,H,W) float mask of frames/pixels with a valid neighbour.
    """
    B, T, C, H, W = images.shape
    device = images.device
    dtype = images.dtype

    K = K.to(dtype)
    extrinsics = extrinsics.to(dtype)
    inv_K = torch.inverse(K)

    photo_stack = []
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

        img_t = images[:, idx_t].reshape(N, C, H, W)
        img_s = images[:, idx_s].reshape(N, C, H, W)
        d_t = depth[:, idx_t].reshape(N, 1, H, W)
        invKt = inv_K[:, idx_t].reshape(N, 3, 3)
        Ks = K[:, idx_s].reshape(N, 3, 3)
        Tt = extrinsics[:, idx_t].reshape(N, 4, 4)
        Ts = extrinsics[:, idx_s].reshape(N, 4, 4)

        warped, valid = _inverse_warp(d_t, img_s, invKt, Ks, Tt, Ts, eps=eps)
        photo = (img_t - warped).abs().mean(dim=1, keepdim=True)  # (N,1,H,W)
        photo = photo * valid + big * (1.0 - valid)

        full_photo = images.new_full((B, T, 1, H, W), big)
        full_valid = images.new_zeros((B, T, 1, H, W))
        full_photo[:, idx_t] = photo.reshape(B, n, 1, H, W)
        full_valid[:, idx_t] = valid.reshape(B, n, 1, H, W)
        photo_stack.append(full_photo)
        valid_stack.append(full_valid)

    if len(photo_stack) == 0:
        zeros = images.new_zeros((B, T, 1, H, W))
        return zeros, zeros

    photo = torch.stack(photo_stack, dim=0).min(dim=0).values  # (B,T,1,H,W)
    valid = torch.stack(valid_stack, dim=0).max(dim=0).values
    err = photo * valid
    return err, valid


# Generic per-layer signal error map. The photometric error map is the special case where
# the warped signal is the RGB image; here the signal can be any (B,T,C,H,W) feature tensor
# (e.g. a projected DPT feature or a HOG descriptor). The warping/residual computation is
# identical, so we reuse ``photometric_error_map`` which already averages over the channel dim.
signal_error_map = photometric_error_map


def plane_sweep_warp(src_feat, depth_samples, K_ref, K_src, ext_ref, ext_src, eps=1e-6):
    """Warp source features into the reference frame under a whole set of depth hypotheses.

    This is the plane sweep that turns matching between two arbitrarily posed views into a
    1D search: instead of scanning an epipolar line, we scan a list of candidate depths and
    record where each one lands. ``photometric_error_map`` above is the D=1 special case,
    evaluated at the depth the network currently believes.

    Args:
        src_feat: (N,C,H,W) source-frame features to sample.
        depth_samples: (N,D,H,W) positive metric depth hypotheses for the reference frame.
        K_ref, K_src: (N,3,3) intrinsics at (H,W).
        ext_ref, ext_src: (N,4,4) world->camera extrinsics (OpenCV).
    Returns:
        warped: (N,C,D,H,W) source features sampled at the reference pixels, per hypothesis.
        valid: (N,1,D,H,W) float mask of hypotheses that land in front of the source camera
            and inside its image.
    """
    N, C, H, W = src_feat.shape
    D = depth_samples.shape[1]
    device, dtype = src_feat.device, src_feat.dtype

    vv, uu = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij',
    )
    pix = torch.stack([uu, vv, torch.ones_like(uu)], dim=0).reshape(3, -1)
    pix = pix.unsqueeze(0).expand(N, -1, -1)                       # (N,3,H*W)

    # Compose reference-pixel -> source-pixel once, so the depth sweep is a single
    # broadcast multiply rather than D separate projections.
    M = torch.bmm(ext_src.to(dtype), torch.inverse(ext_ref.to(dtype)))
    KR = torch.bmm(K_src.to(dtype), M[:, :3, :3])                  # (N,3,3)
    Kt = torch.bmm(K_src.to(dtype), M[:, :3, 3:4])                 # (N,3,1)
    ray = torch.bmm(torch.inverse(K_ref.to(dtype)), pix)           # (N,3,H*W)
    base = torch.bmm(KR, ray)                                      # (N,3,H*W)

    proj = (base.unsqueeze(2) * depth_samples.reshape(N, 1, D, H * W)
            + Kt.reshape(N, 3, 1, 1))                              # (N,3,D,H*W)
    z = proj[:, 2:3]
    in_front = z > eps
    z_safe = torch.where(in_front, z, torch.full_like(z, eps))
    gx = 2.0 * (proj[:, 0:1] / z_safe) / (W - 1) - 1.0
    gy = 2.0 * (proj[:, 1:2] / z_safe) / (H - 1) - 1.0

    # A non-finite projection means there is no correspondence, which is the same thing
    # "out of frame" already means -- so map it out of bounds and let the shared valid
    # test below reject it. Without this, an intrinsics matrix that has not converged yet
    # (GEM's focal length can still overflow to inf early in training) makes grid_sample
    # return NaN, and NaN * valid is still NaN however carefully the mask is applied.
    out_of_bounds = torch.full_like(gx, 2.0)
    gx = torch.where(torch.isfinite(gx), gx, out_of_bounds)
    gy = torch.where(torch.isfinite(gy), gy, out_of_bounds)

    grid = torch.cat([gx, gy], dim=1).permute(0, 2, 3, 1)          # (N,D,H*W,2)
    warped = F.grid_sample(
        src_feat, grid.reshape(N, D * H, W, 2),
        mode='bilinear', padding_mode='zeros', align_corners=True)
    warped = warped.reshape(N, C, D, H, W)

    valid = (in_front & (gx.abs() <= 1.0) & (gy.abs() <= 1.0))
    return warped, valid.reshape(N, 1, D, H, W).to(dtype)
