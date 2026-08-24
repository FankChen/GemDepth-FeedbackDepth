# Multi-scale refine head that re-measures geometry between steps (path A).
#
# Iterative stereo (RAFT / IGEV / FoundationStereo) does not win because the
# update operator is applied many times; it wins because every application reads
# a *new* measurement. The current disparity indexes the cost volume, so each
# step sees evidence about where it is still wrong. Without that, repeating an
# update is a closed system: the only thing that changes is the network's own
# previous answer, and the cheapest solution it can learn is the identity.
#
# Monocular video has no cost volume, but it does have the same kind of signal:
# warp the neighbouring frames with the running depth and the predicted camera
# pose, and the photometric residual tells you where the geometry disagrees with
# the video. Crucially that residual is a function of the depth, so it changes
# every round -- which is what turns "more rounds" from "a deeper head" into an
# estimate -> measure -> correct loop.
#
# The error encoders are zero-initialised, so at init this head is numerically
# identical to its parent and the comparison starts from the same point.
#
# Requires camera poses, i.e. GEM (``use_gem: true``). With ``extrinsics=None``
# every injection is skipped and the head degrades to the plain multi-scale one.

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.decoder_registry import register
from model.dpt_multiscale_convnext import DPTHeadMultiScaleRefineConvNeXt
from model.util.warp import photometric_error_map, scale_intrinsics

_EPS = 1e-3


@register
class DPTHeadMultiScaleErrMapConvNeXt(DPTHeadMultiScaleRefineConvNeXt):
    """Multi-scale refine head with a photometric-consistency feedback loop."""

    def __init__(self, *args, warp_offsets=(-1, 1),
                 predicts_inverse_depth=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.warp_offsets = tuple(warp_offsets)
        # Which quantity the head emits. Warping needs metric depth, and the
        # video/SSI losses train this head in disparity space, so the running
        # prediction has to be inverted before it can move pixels around.
        self.predicts_inverse_depth = bool(predicts_inverse_depth)

        features = self.output_conv1_heads[0].in_channels
        self.error_encoders = nn.ModuleList([
            self._make_error_encoder(features)
            for _ in range(len(self.delta_heads))
        ])

    @staticmethod
    def _make_error_encoder(features):
        """(photometric residual, validity) -> a feature-sized correction."""
        encoder = nn.Sequential(
            nn.Conv2d(2, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, features, kernel_size=3, stride=1, padding=1),
        )
        # Zero-init the last conv: the injection starts as a no-op, so training
        # begins from the parent head rather than from a random perturbation of it.
        nn.init.zeros_(encoder[-1].weight)
        nn.init.zeros_(encoder[-1].bias)
        return encoder

    def _to_metric_depth(self, prediction):
        prediction = F.relu(prediction).clamp_min(_EPS)
        if self.predicts_inverse_depth:
            return 1.0 / prediction
        return prediction

    def _adapt_feat(self, scale_index, feat, depth_prev, output_size, ctx):
        images = ctx['images']
        extrinsics = ctx['extrinsics']
        intrinsics = ctx['intrinsics']
        # Nothing to measure before the first estimate exists, and nothing to
        # measure with unless GEM supplied a pose.
        if depth_prev is None or images is None or extrinsics is None or intrinsics is None:
            return feat

        frame_length = int(ctx['frame_length'])
        bt, _, height, width = feat.shape
        batch = bt // frame_length

        depth = F.interpolate(
            self._carry(depth_prev), size=(height, width),
            mode="bilinear", align_corners=True)
        depth = self._to_metric_depth(depth.float()).reshape(
            batch, frame_length, 1, height, width)

        frames = F.interpolate(
            images.flatten(0, 1).float(), size=(height, width),
            mode="bilinear", align_corners=False)
        frames = frames.reshape(batch, frame_length, frames.shape[1], height, width)

        K = scale_intrinsics(
            intrinsics.detach().float(), images.shape[-2:], (height, width))
        error, valid = photometric_error_map(
            frames, depth, K, extrinsics.detach().float(),
            offsets=self.warp_offsets)

        # A pixel whose warp is not finite has no usable geometric evidence,
        # which is exactly what `valid` already expresses -- so fold it in there
        # rather than letting it through. This is not defensive padding: early in
        # training GEM's focal length can still overflow to inf, and because the
        # encoders are zero-initialised, 0 * nan = nan would silently poison the
        # feature map and destroy the very no-op guarantee the zero-init provides.
        finite = torch.isfinite(error) & torch.isfinite(valid)
        error = torch.where(finite, error, torch.zeros_like(error))
        valid = torch.where(finite, valid, torch.zeros_like(valid))

        error_input = torch.cat([error, valid], dim=2).reshape(bt, 2, height, width)
        return feat + self.error_encoders[scale_index](error_input.to(feat.dtype))
