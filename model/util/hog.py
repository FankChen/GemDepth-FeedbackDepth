"""Differentiable-friendly HOG (Histogram of Oriented Gradients) feature maps.

Produces a dense per-pixel orientation-histogram descriptor that can be used as a
*signal* to warp/compare across temporal neighbours (see ``signal_error_map``).

The HOG descriptor itself is extracted purely from the input image (it does not depend
on depth), so it is returned detached: the gradient that matters for self-supervision
flows through the depth-driven warp, not through the descriptor extraction.
"""
import torch
import torch.nn.functional as F


def _gray(images):
    # images: (N,3,H,W) in any range -> (N,1,H,W) luminance.
    if images.shape[1] == 3:
        r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b
    return images[:, 0:1]


def hog_feature_map(images, nbins=9, cell=8, eps=1e-6):
    """Dense soft-binned HOG descriptor.

    Args:
        images: (B,T,3,H,W) frames at the desired (stage) resolution.
        nbins:  number of unsigned orientation bins over [0, pi).
        cell:   spatial cell size for local aggregation (smoothing window).
    Returns:
        hog: (B,T,nbins,H,W) L2-normalised orientation histogram, detached.
    """
    B, T, C, H, W = images.shape
    x = images.reshape(B * T, C, H, W).float()
    gray = _gray(x)  # (N,1,H,W)

    # Sobel gradients.
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)

    mag = torch.sqrt(gx * gx + gy * gy + eps)                 # (N,1,H,W)
    ori = torch.atan2(gy, gx)                                 # [-pi, pi]
    ori = torch.remainder(ori, torch.pi)                     # unsigned [0, pi)

    # Soft (linear) assignment to neighbouring orientation bins.
    bin_width = torch.pi / nbins
    pos = ori / bin_width                                     # [0, nbins)
    lo = torch.floor(pos)
    frac = pos - lo
    lo_idx = lo.long() % nbins
    hi_idx = (lo_idx + 1) % nbins

    hist = x.new_zeros(B * T, nbins, H, W)
    hist.scatter_add_(1, lo_idx, mag * (1.0 - frac))
    hist.scatter_add_(1, hi_idx, mag * frac)

    # Local cell aggregation (smoothing) then L2 block normalisation.
    if cell and cell > 1:
        pad = cell // 2
        hist = F.avg_pool2d(hist, kernel_size=cell, stride=1, padding=pad)
        if hist.shape[-2:] != (H, W):
            hist = F.interpolate(hist, size=(H, W), mode='bilinear', align_corners=False)
    norm = torch.sqrt((hist * hist).sum(dim=1, keepdim=True) + eps)
    hist = hist / norm

    return hist.reshape(B, T, nbins, H, W).detach()
