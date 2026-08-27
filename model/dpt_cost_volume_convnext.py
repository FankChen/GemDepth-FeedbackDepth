"""IGEV-MVS ported to consecutive video frames, as a drop-in decoder.

Reference: "Iterative Geometry Encoding Volume for Stereo Matching" (arXiv:2303.06615),
``IGEV-MVS`` variant. This is a direct port, not a re-design: the pipeline is still
sweep -> correlate -> 3D-regularise -> softargmin -> ConvGRU iterations -> convex upsample.
The only substitution is which views get matched. IGEV-MVS matches a reference view
against other views of a static scene using their known relative poses; here the "other
views" are the neighbouring frames of a video and the pose comes from GEM.

Unlike the DPT heads in this repository, this one does not fuse the backbone pyramid into
a depth: the pyramid feeds the cost volume (level ``volume_level`` supplies the matching
features, the deeper levels guide the 3D aggregation) and the depth comes out of the
volume. It therefore *replaces* the DPT stack rather than extending it.

Three consequences of the video setting that stereo does not have, and that this port
does not solve -- they are properties of the experiment, not bugs:
  * a static camera means a zero baseline, and every hypothesis then warps identically,
    so the volume carries no information at all;
  * moving objects violate the static-scene assumption the sweep is built on;
  * the pose is estimated rather than calibrated, so its scale error transfers directly
    into depth. ``PixelViewWeight`` down-weights unreliable pixels, which softens the
    first two but does not remove them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.decoder_registry import register
from model.util.cost_volume import (Hourglass3D, PixelViewWeight, VolumeIndexer,
                                    convex_upsample, depth_hypotheses,
                                    groupwise_correlation, indexed_volume_dim)
from model.util.warp import plane_sweep_warp, scale_intrinsics


class ConvGRU(nn.Module):
    """RAFT/IGEV convolutional GRU."""

    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, hidden, *inputs):
        x = torch.cat(inputs, dim=1)
        hx = torch.cat([hidden, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * hidden, x], dim=1)))
        return (1.0 - z) * hidden + z * q


class MotionEncoder(nn.Module):
    """Fuse the indexed volume values with the current hypothesis into a GRU input."""

    def __init__(self, volume_dim, output_dim=128):
        super().__init__()
        self.conv_volume = nn.Sequential(
            nn.Conv2d(volume_dim, 64, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.conv_index = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse = nn.Conv2d(128, output_dim - 1, 3, padding=1)

    def forward(self, index_map, volume_features):
        fused = F.relu(self.fuse(torch.cat([
            self.conv_volume(volume_features), self.conv_index(index_map)], dim=1)))
        return torch.cat([fused, index_map], dim=1)


@register
class DPTHeadCostVolumeConvNeXt(nn.Module):
    """Cost-volume decoder: matches each frame against its temporal neighbours."""

    def __init__(self, in_channels_list, num_frames=4, patch_size=4,
                 volume_level=1, num_sample=32, num_groups=8, volume_channels=8,
                 match_dim=96, iters=8, hidden_dim=128, corr_radius=4, corr_levels=2,
                 depth_min=0.5, depth_max=80.0, warp_offsets=(-1, 1), **_ignored):
        super().__init__()
        self.volume_level = int(volume_level)
        self.volume_stride = int(patch_size) * (2 ** self.volume_level)
        self.num_sample = int(num_sample)
        self.num_groups = int(num_groups)
        self.iters = int(iters)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.warp_offsets = tuple(warp_offsets)

        # Matching descriptors. L2-normalised so the correlation is a cosine similarity and
        # cannot be inflated by feature magnitude.
        self.matcher = nn.Sequential(
            nn.Conv2d(in_channels_list[self.volume_level], match_dim, 3, padding=1, bias=False),
            nn.InstanceNorm2d(match_dim), nn.ReLU(inplace=True),
            nn.Conv2d(match_dim, match_dim, 1))
        self.view_weight = PixelViewWeight(self.num_groups)

        guide_channels = in_channels_list[self.volume_level + 1:]
        self.volume_stem = nn.Conv3d(self.num_groups, volume_channels, 3, padding=1, bias=False)
        self.cost_agg = Hourglass3D(volume_channels, guide_channels)
        self.classifier = nn.Conv3d(volume_channels, 1, 3, padding=1, bias=False)

        # IGEV-MVS has no context network: the GRU hidden states are seeded from the
        # regularised volume itself, whose channel axis is the hypothesis axis.
        self.hidden_stems = nn.ModuleList([
            nn.Conv2d(self.num_sample, hidden_dim, 3, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=2)])

        self.corr_radius, self.corr_levels = int(corr_radius), int(corr_levels)
        self.encoder = MotionEncoder(
            indexed_volume_dim(self.corr_levels, self.corr_radius), output_dim=hidden_dim)
        self.gru_fine = ConvGRU(hidden_dim, hidden_dim + hidden_dim)
        self.gru_mid = ConvGRU(hidden_dim, hidden_dim + hidden_dim)
        self.gru_coarse = ConvGRU(hidden_dim, hidden_dim)
        self.index_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 3, padding=1))

        # Convex upsampling. IGEV derives the weights from an image stem, which a decoder
        # does not receive here, so this uses RAFT's original pixel-shuffle formulation --
        # same learned 3x3 convex combination, different place to read the features from.
        self.upsample_mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 9 * self.volume_stride ** 2, 1))

    def _build_volume(self, features, images, extrinsics, intrinsics, frame_length):
        """Sweep every frame against its neighbours -> (N,G,D,h,w) fused correlation."""
        descriptors = self.matcher(features)
        descriptors = descriptors / (descriptors.norm(dim=1, keepdim=True) + 1e-5)
        total, _, height, width = descriptors.shape
        batch = total // frame_length
        descriptors = descriptors.reshape(batch, frame_length, -1, height, width)

        K = scale_intrinsics(intrinsics.float(), images.shape[-2:], (height, width))
        extrinsics = extrinsics.float()
        samples = depth_hypotheses(self.depth_min, self.depth_max, self.num_sample,
                                   height, width, descriptors.device, torch.float32)

        correlation = descriptors.new_zeros(
            (batch, frame_length, self.num_groups, self.num_sample, height, width))
        weight = descriptors.new_zeros((batch, frame_length, 1, 1, height, width))

        for offset in self.warp_offsets:
            first, last = max(0, -offset), min(frame_length, frame_length - offset)
            if last <= first:
                continue
            ref = torch.arange(first, last, device=descriptors.device)
            src = ref + offset
            count = int(ref.numel())
            flat = batch * count

            warped, valid = plane_sweep_warp(
                descriptors[:, src].reshape(flat, -1, height, width).float(),
                samples.expand(flat, -1, -1, -1),
                K[:, ref].reshape(flat, 3, 3), K[:, src].reshape(flat, 3, 3),
                extrinsics[:, ref].reshape(flat, 4, 4), extrinsics[:, src].reshape(flat, 4, 4))

            pair = groupwise_correlation(
                warped, descriptors[:, ref].reshape(flat, -1, 1, height, width).float(),
                self.num_groups)
            pair = pair * valid
            confidence = (self.view_weight(pair)
                          * valid.mean(dim=2)).reshape(batch, count, 1, 1, height, width)
            correlation[:, ref] += pair.reshape(
                batch, count, self.num_groups, self.num_sample, height, width) * confidence
            weight[:, ref] += confidence

        correlation = correlation / weight.clamp_min(1e-5)
        return correlation.reshape(-1, self.num_groups, self.num_sample, height, width)

    def forward(self, out_features, patch_h, patch_w, frame_length,
                init_depth=None, images=None, extrinsics=None, intrinsics=None,
                layer_3_att=None, layer_4_att=None, mode=None):
        if images is None or extrinsics is None or intrinsics is None:
            raise ValueError(
                'DPTHeadCostVolumeConvNeXt needs camera poses; enable GEM '
                '(model.use_gem=true, model.encoder_decoder_only=false)')

        volume = self._build_volume(
            out_features[self.volume_level], images, extrinsics, intrinsics, frame_length)
        guides = [f.float() for f in out_features[self.volume_level + 1:]]

        volume = self.cost_agg(self.volume_stem(volume), guides)
        volume = self.classifier(volume).squeeze(1)                     # (N,D,h,w)

        probability = torch.softmax(volume, dim=1)
        index = torch.arange(self.num_sample, device=volume.device,
                             dtype=probability.dtype).view(1, -1, 1, 1)
        hypothesis = (probability * index).sum(dim=1, keepdim=True)     # softargmin

        indexer = VolumeIndexer(volume.unsqueeze(1), probability.unsqueeze(1),
                                num_levels=self.corr_levels, radius=self.corr_radius)

        hidden = []
        state = volume
        for stem in self.hidden_stems:
            state = stem(state)
            hidden.append(torch.tanh(state))

        predictions = [self._to_full_resolution(hypothesis, hidden[0], patch_h, patch_w)]
        for _ in range(self.iters):
            hypothesis = hypothesis.detach()
            indexed = indexer(hypothesis)

            hidden[2] = self.gru_coarse(hidden[2], F.avg_pool2d(hidden[1], 3, 2, 1))
            hidden[1] = self.gru_mid(hidden[1], F.avg_pool2d(hidden[0], 3, 2, 1),
                                     _resize(hidden[2], hidden[1]))
            hidden[0] = self.gru_fine(hidden[0], self.encoder(hypothesis, indexed),
                                      _resize(hidden[1], hidden[0]))

            hypothesis = hypothesis + self.index_head(hidden[0])
            predictions.append(
                self._to_full_resolution(hypothesis, hidden[0], patch_h, patch_w))

        if self.training:
            return predictions
        return predictions[-1]

    def _to_full_resolution(self, hypothesis, hidden, patch_h, patch_w):
        """Hypothesis index -> normalised disparity at input resolution."""
        weights = F.pixel_shuffle(self.upsample_mask(hidden), self.volume_stride)
        weights = torch.softmax(weights, dim=1)
        disparity = convex_upsample(hypothesis, weights, self.volume_stride)
        return disparity / max(self.num_sample - 1, 1)


def _resize(x, reference):
    return F.interpolate(x, size=reference.shape[-2:], mode='bilinear', align_corners=True)
