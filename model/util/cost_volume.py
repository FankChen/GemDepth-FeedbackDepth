"""Cost-volume building blocks, ported from IGEV-MVS (Xu et al., CVPR 2023).

Reference: "Iterative Geometry Encoding Volume for Stereo Matching", arXiv:2303.06615,
code at https://github.com/gangweiX/IGEV (the ``IGEV-MVS`` variant).

The MVS variant is the one that matters here: unlike the stereo version it does not
assume a rectified pair, it sweeps a set of depth hypotheses using the relative camera
pose. That makes it directly applicable to consecutive video frames, which is exactly
the setting of this repository -- the "left/right" pair becomes "current/neighbour".

Depths are handled in *normalised inverse depth*: hypothesis index ``i`` maps to
``1/d = 1/d_max + i/(D-1) * (1/d_min - 1/d_max)``, so index 0 is the far plane and index
D-1 is the near plane. Dividing the index by ``D-1`` therefore yields a disparity-like
quantity in [0, 1], which the repository's scale-and-shift-invariant losses can consume
without conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def depth_hypotheses(depth_min, depth_max, num_sample, height, width, device, dtype):
    """Uniform samples in inverse-depth space -- (1,D,H,W) metric depth.

    Uniform in inverse depth rather than depth: matching resolution is what matters, and
    it degrades with distance exactly like inverse depth does.
    """
    inv_min, inv_max = 1.0 / float(depth_min), 1.0 / float(depth_max)
    index = torch.arange(num_sample, device=device, dtype=dtype) / max(num_sample - 1, 1)
    inverse_depth = inv_max + index * (inv_min - inv_max)
    return (1.0 / inverse_depth).view(1, num_sample, 1, 1).expand(1, num_sample, height, width)


def groupwise_correlation(feat_a, feat_b, num_groups):
    """(N,C,D,H,W) x (N,C,1,H,W) -> (N,G,D,H,W) correlation, averaged within each group."""
    channels = feat_a.shape[1]
    assert channels % num_groups == 0, f"{channels} channels do not split into {num_groups} groups"
    per_group = channels // num_groups
    a = feat_a.reshape(feat_a.shape[0], num_groups, per_group, *feat_a.shape[2:])
    b = feat_b.reshape(feat_b.shape[0], num_groups, per_group, *feat_b.shape[2:])
    return (a * b).mean(dim=2)


class PixelViewWeight(nn.Module):
    """Per-pixel confidence for one source view, from its correlation volume.

    A neighbouring frame is useless where the scene is occluded, out of frame, or moving,
    and in video all three are common. IGEV-MVS weights each source view per pixel before
    averaging; the weight is driven by how peaked that view's matching distribution is.
    """

    def __init__(self, num_groups):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_groups, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, correlation):
        batch, groups, depth, height, width = correlation.shape
        scores = self.conv(
            correlation.transpose(1, 2).reshape(batch * depth, groups, height, width))
        scores = scores.view(batch, depth, height, width)
        # A confident view has one clear winner among the hypotheses.
        return torch.softmax(scores, dim=1).max(dim=1).values.unsqueeze(1)


class FeatureAtt(nn.Module):
    """Excite cost-volume channels with weights computed from the 2D feature (CoEx)."""

    def __init__(self, volume_channels, feature_channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, 1),
            nn.BatchNorm2d(feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, volume_channels, 1),
        )

    def forward(self, volume, feature):
        return torch.sigmoid(self.att(feature).unsqueeze(2)) * volume


def _conv3d(in_ch, out_ch, stride=1, transposed=False):
    if transposed:
        layer = nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
    else:
        layer = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
    return nn.Sequential(layer, nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))


class Hourglass3D(nn.Module):
    """Lightweight 3D UNet that regularises the cost volume into a geometry encoding volume.

    This is what separates IGEV from RAFT-style methods: raw correlations only see local
    evidence, so they cannot resolve textureless or occluded regions. Aggregating across
    the volume propagates confident matches into ambiguous neighbourhoods.

    ``guide_channels`` are the 2D backbone features at each downsampled level; the number
    of levels follows from where the volume was built, so a volume at stride 8 aggregates
    over fewer levels than one at stride 4.
    """

    def __init__(self, channels, guide_channels):
        super().__init__()
        self.down = nn.ModuleList()
        self.down_att = nn.ModuleList()
        self.up = nn.ModuleList()
        self.up_att = nn.ModuleList()
        self.agg = nn.ModuleList()

        widths = [channels * (2 ** (i + 1)) for i in range(len(guide_channels))]
        prev = channels
        for width, guide in zip(widths, guide_channels):
            self.down.append(nn.Sequential(_conv3d(prev, width, stride=2), _conv3d(width, width)))
            self.down_att.append(FeatureAtt(width, guide))
            prev = width

        for level in reversed(range(len(widths))):
            target = channels if level == 0 else widths[level - 1]
            self.up.append(_conv3d(widths[level], target, transposed=True))
            if level == 0:
                self.agg.append(nn.Identity())
                self.up_att.append(nn.Identity())
            else:
                self.agg.append(nn.Sequential(_conv3d(target * 2, target), _conv3d(target, target)))
                self.up_att.append(FeatureAtt(target, guide_channels[level - 1]))

    def forward(self, volume, guides):
        skips = []
        x = volume
        for block, att, guide in zip(self.down, self.down_att, guides):
            x = att(block(x), guide)
            skips.append(x)

        for index, (block, agg, att) in enumerate(zip(self.up, self.agg, self.up_att)):
            x = block(x)
            level = len(skips) - 2 - index
            if level >= 0:
                skip = skips[level]
                x = agg(torch.cat([_match(x, skip), skip], dim=1))
                x = att(x, guides[level])
        return x


def _match(x, reference):
    """Trilinear-resize ``x`` onto ``reference``'s spatial shape (odd sizes break stride-2)."""
    if x.shape[2:] == reference.shape[2:]:
        return x
    return F.interpolate(x, size=reference.shape[2:], mode='trilinear', align_corners=True)


def _bilinear_sampler(volume, coords):
    """Sample (N,1,1,D) volumes at (N,1,K,1) hypothesis coordinates, in pixel units."""
    depth = volume.shape[-1]
    xgrid = 2.0 * coords[..., :1] / max(depth - 1, 1) - 1.0
    grid = torch.cat([xgrid, torch.zeros_like(xgrid)], dim=-1)
    return F.grid_sample(volume, grid, align_corners=True)


def indexed_volume_dim(num_levels, radius):
    """Channel count produced by :class:`VolumeIndexer` -- both volumes, every level."""
    return 2 * num_levels * (2 * radius + 1)


class VolumeIndexer:
    """Look up a neighbourhood of the volume around the current hypothesis index.

    Each GRU iteration reads the volume at ``d_k +/- r`` rather than re-running the sweep,
    which is what makes the iterations cheap. Two pooled levels widen the receptive field
    over the hypothesis axis.
    """

    def __init__(self, geometry_volume, correlation_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.pyramids = []
        for volume in (geometry_volume, correlation_volume):
            batch, _, depth, height, width = volume.shape
            flat = volume.permute(0, 3, 4, 1, 2).reshape(batch * height * width, 1, 1, depth)
            levels = [flat]
            for _ in range(num_levels - 1):
                levels.append(F.avg_pool2d(levels[-1], [1, 2], stride=[1, 2]))
            self.pyramids.append(levels)
        self.shape = (batch, height, width)

    def __call__(self, index_map):
        batch, height, width = self.shape
        offsets = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1,
                                 device=index_map.device, dtype=index_map.dtype)
        offsets = offsets.view(1, 1, -1, 1)
        outputs = []
        for level in range(self.num_levels):
            coords = offsets + index_map.reshape(batch * height * width, 1, 1, 1) / (2 ** level)
            for pyramid in self.pyramids:
                sampled = _bilinear_sampler(pyramid[level], coords)
                outputs.append(sampled.view(batch, height, width, -1))
        return torch.cat(outputs, dim=-1).permute(0, 3, 1, 2).contiguous().float()


def convex_upsample(low_res, weights, scale):
    """Upsample by a learned convex combination of each 3x3 low-resolution neighbourhood."""
    batch, _, height, width = low_res.shape
    neighbourhood = F.unfold(low_res, 3, padding=1).reshape(batch, 9, height, width)
    neighbourhood = F.interpolate(
        neighbourhood, (height * scale, width * scale), mode='nearest')
    return (neighbourhood * weights).sum(dim=1, keepdim=True)
