"""Geometric temporal (cycle) consistency loss for video depth (2026-07-09).

Differentiates from the existing TemporalGradientMatchingLoss (which matches the temporal
*difference* of depth against GT) by enforcing MULTI-VIEW GEOMETRIC consistency: reproject
frame t's depth into frame t+o using the camera pose, and penalise the disagreement between
the projected depth and the neighbour frame's own depth at the reprojected pixel. This is the
bundle-adjustment-style constraint that the error-map / BAT+Lin heads are built around.

GemDepth predicts affine-invariant disparity, so geometric reprojection needs a metric depth;
``align_pred_metric`` converts the predicted disparity to metric depth via a per-sequence GT
scale/shift (scale/shift detached, gradient flows through the prediction).
"""
import torch
import torch.nn.functional as F


def align_pred_metric(pred_disp, gt_depth, mask, eps=1e-8):
    """Convert predicted disparity to metric depth via per-sequence GT scale/shift.

    pred_disp / gt_depth / mask: (B,T,1,H,W). scale/shift are detached (no grad), so the
    returned metric depth is a differentiable function of ``pred_disp`` only through the
    affine map — the alignment itself does not leak GT gradients.
    """
    B, T = pred_disp.shape[:2]
    out = []
    for i in range(B):
        p = pred_disp[i].reshape(-1)
        g = gt_depth[i].reshape(-1)
        m = (mask[i].reshape(-1) > 0) & (g > 1e-3)
        if m.sum() < 10:
            out.append(torch.ones_like(pred_disp[i]))
            continue
        with torch.no_grad():
            gd = 1.0 / (g[m] + eps)                       # GT disparity
            pd = p[m]
            A = torch.stack([pd, torch.ones_like(pd)], dim=1)
            X = torch.linalg.lstsq(A, gd.unsqueeze(1)).solution
            s, sh = X[0, 0], X[1, 0]
        aligned_disp = (s * pred_disp[i] + sh).clamp(min=1e-3)  # grad flows through pred
        out.append(1.0 / aligned_disp)                          # metric depth
    return torch.stack(out, dim=0)                              # (B,T,1,H,W)


def geometric_temporal_consistency(depth, K, extrinsics, offsets=(1,), eps=1e-6):
    """Multi-view reprojection consistency of metric depth across temporal neighbours.

    For each offset o: backproject frame t pixels with depth_t, transform to frame (t+o)'s
    camera via the GT pose, project, and compare the projected depth z with frame (t+o)'s
    depth sampled at the reprojected pixel. Returns a masked-L1 consistency loss.

    depth: (B,T,1,H,W) metric; K: (B,T,3,3) at depth resolution; extrinsics: (B,T,4,4) world->cam.
    """
    if depth.dim() == 4:
        depth = depth.unsqueeze(2)
    B, T, _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    K = K.to(dtype)
    ext = extrinsics.to(dtype)
    if K.dim() == 3:                        # (B,3,3) shared intrinsics -> per-frame (B,T,3,3)
        K = K.unsqueeze(1).expand(B, T, 3, 3)
    inv_K = torch.inverse(K)

    vv, uu = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij',
    )
    pix = torch.stack([uu, vv, torch.ones_like(uu)], dim=0).reshape(3, -1)   # (3,H*W)

    total = depth.new_zeros(())
    cnt = 0
    for o in offsets:
        t0, t1 = max(0, -o), min(T, T - o)
        if t1 <= t0:
            continue
        idx_t = torch.arange(t0, t1, device=device)
        idx_s = idx_t + o
        n = int(idx_t.numel())
        N = B * n

        d_t = depth[:, idx_t].reshape(N, 1, H * W)
        d_s = depth[:, idx_s].reshape(N, 1, H, W)
        invKt = inv_K[:, idx_t].reshape(N, 3, 3)
        Ks = K[:, idx_s].reshape(N, 3, 3)
        Tt = ext[:, idx_t].reshape(N, 4, 4)
        Ts = ext[:, idx_s].reshape(N, 4, 4)

        pixN = pix.unsqueeze(0).expand(N, -1, -1)               # (N,3,H*W)
        cam_t = torch.bmm(invKt, pixN) * d_t                    # (N,3,H*W)
        cam_t_h = torch.cat([cam_t, torch.ones(N, 1, H * W, device=device, dtype=dtype)], dim=1)
        M = torch.bmm(Ts, torch.inverse(Tt))                    # cam_t -> cam_s
        cam_s = torch.bmm(M, cam_t_h)[:, :3]                    # (N,3,H*W)

        proj = torch.bmm(Ks, cam_s)                             # (N,3,H*W)
        z = proj[:, 2:3]                                        # projected depth in frame s
        z_safe = torch.where(z.abs() < eps, torch.full_like(z, eps), z)
        u_s = proj[:, 0:1] / z_safe
        v_s = proj[:, 1:2] / z_safe
        gx = 2.0 * u_s / (W - 1) - 1.0
        gy = 2.0 * v_s / (H - 1) - 1.0
        grid = torch.cat([gx, gy], dim=1).permute(0, 2, 1).reshape(N, H, W, 2)

        d_s_at = F.grid_sample(d_s, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        z_map = z.reshape(N, 1, H, W)

        in_bounds = (gx.abs() <= 1.0) & (gy.abs() <= 1.0) & (z > eps)
        valid = (in_bounds.reshape(N, 1, H, W) &
                 (d_t.reshape(N, 1, H, W) > 0) & (d_s_at > 0)).to(dtype)

        # scale-robust: compare in inverse-depth (disparity) space so far/near are balanced
        resid = (1.0 / z_map.clamp(min=1e-3) - 1.0 / d_s_at.clamp(min=1e-3)).abs() * valid
        total = total + resid.sum() / valid.sum().clamp(min=1.0)
        cnt += 1

    return total / max(cnt, 1)
