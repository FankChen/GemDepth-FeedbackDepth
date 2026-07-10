# Dense (per-pixel) Histogram of Oriented Gradients, used as an optional warp signal for
# the per-layer DPT head. The output is a soft, differentiable orientation histogram at every
# pixel so it can be inverse-warped and compared across frames just like the RGB signal.

import math

import torch
import torch.nn.functional as F


def hog_feature_map(images, nbins=9, eps=1e-6):
    """Compute a dense soft-HOG feature map.

    Args:
        images: (B,T,C,H,W) frames (RGB or grayscale). Multi-channel inputs are converted to
            luminance before computing gradients.
        nbins: number of unsigned orientation bins spanning [0, pi).
        eps: numerical stabiliser.
    Returns:
        (B,T,nbins,H,W) per-pixel orientation histogram weighted by gradient magnitude.
    """
    B, T, C, H, W = images.shape
    x = images.reshape(B * T, C, H, W).float()

    if C == 3:
        weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        gray = (x * weights).sum(dim=1, keepdim=True)
    elif C == 1:
        gray = x
    else:
        gray = x.mean(dim=1, keepdim=True)

    kx = gray.new_tensor([[-1.0, 0.0, 1.0]]).view(1, 1, 1, 3)
    ky = gray.new_tensor([[-1.0], [0.0], [1.0]]).view(1, 1, 3, 1)
    gx = F.conv2d(gray, kx, padding=(0, 1))
    gy = F.conv2d(gray, ky, padding=(1, 0))

    mag = torch.sqrt(gx * gx + gy * gy + eps)
    ang = torch.atan2(gy, gx)                       # (-pi, pi]
    ang = torch.remainder(ang, math.pi)             # unsigned orientation in [0, pi)

    bin_width = math.pi / nbins
    # Soft (linear) assignment to the two nearest bins for a differentiable histogram.
    pos = ang / bin_width                            # (BT,1,H,W) in [0, nbins)
    lo = torch.floor(pos)
    frac = pos - lo
    lo_idx = lo.long() % nbins
    hi_idx = (lo_idx + 1) % nbins

    hist = gray.new_zeros(B * T, nbins, H, W)
    hist.scatter_add_(1, lo_idx, mag * (1.0 - frac))
    hist.scatter_add_(1, hi_idx, mag * frac)

    return hist.reshape(B, T, nbins, H, W)
