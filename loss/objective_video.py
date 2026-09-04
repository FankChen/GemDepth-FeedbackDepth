"""Built-in depth objectives and the legacy-compatible prediction router."""

import torch
import torch.nn as nn

from loss.multiscale_video_l1_loss import MultiScaleVideoL1Loss
from loss.multiscale_videoloss import MultiScaleVideoDepthLoss
from loss.objective_registry import register
from loss.videoloss import VideoDepthLoss


class RoutedDepthObjective(nn.Module):
    """Route tensor/list predictions without exposing that detail to training."""

    def __init__(self, single, multiscale, description):
        super().__init__()
        self.single = single
        self.multiscale = multiscale
        self.description = description

    @staticmethod
    def _strip_depth_channel(value):
        if value.ndim == 5:
            if value.shape[2] != 1:
                raise ValueError(
                    f"Expected singleton depth channel, got {tuple(value.shape)}")
            return value.squeeze(2)
        if value.ndim != 4:
            raise ValueError(
                f"Expected depth tensor (B,T[,1],H,W), got {tuple(value.shape)}")
        return value

    def forward(self, prediction, target, mask, intrinsic_gt, extrinsic_gt,
                pose_enc_list, extrinsic_pred):
        target = self._strip_depth_channel(target)
        mask = self._strip_depth_channel(mask)
        if isinstance(prediction, (list, tuple)):
            predictions = [self._strip_depth_channel(item) for item in prediction]
            return self.multiscale(
                predictions, target, mask, intrinsic_gt, extrinsic_gt,
                pose_enc_list, extrinsic_pred)
        prediction = self._strip_depth_channel(prediction)
        return self.single(
            prediction, target, mask, intrinsic_gt, extrinsic_gt,
            pose_enc_list, extrinsic_pred)


def resolve_scale_weights(scale_weights=None, scale_gamma=None, num_scales=4):
    """Resolve explicit or IGEV-style weights; explicit weights take priority."""
    if scale_weights is not None:
        return [float(weight) for weight in scale_weights]
    if scale_gamma is None or float(scale_gamma) <= 0:
        return None
    gamma = float(scale_gamma)
    count = int(num_scales)
    return [gamma ** (count - 1 - index) for index in range(count)]


@register("video", "videoloss", "video_loss", "multiscale_video")
def build_video_objective(
        pose_flag, scale_weights=None, scale_gamma=None, num_scales=4,
        normalize_scale_weights=True, alpha=0.5, beta=0.2, scales=4,
        trim=0, stable_scale=10.0, camera_weight_focal=0.0):
    """Original objective, with every term exposed as a config argument."""
    weights = resolve_scale_weights(scale_weights, scale_gamma, num_scales)
    common = dict(
        alpha=float(alpha), beta=float(beta), scales=int(scales),
        trim=float(trim), stable_scale=float(stable_scale),
        pose_flag=bool(pose_flag),
        camera_weight_focal=float(camera_weight_focal))
    return RoutedDepthObjective(
        single=VideoDepthLoss(**common),
        multiscale=MultiScaleVideoDepthLoss(
            **common, scale_weights=weights,
            normalize_scale_weights=bool(normalize_scale_weights)),
        description=(
            f"video(alpha={float(alpha)}, stable_scale={float(stable_scale)}, "
            f"pose_flag={bool(pose_flag)}, scale_weights={weights}, "
            f"normalize={bool(normalize_scale_weights)})"),
    )


@register("l2")
def build_legacy_l2_objective(
        pose_flag, scale_weights=None, scale_gamma=None, num_scales=4,
        normalize_scale_weights=True, loss_type="l2"):
    """Historical contract: video objective for tensors, metric L1/L2 for lists."""
    del normalize_scale_weights  # MultiScaleVideoL1Loss always normalises weights.
    weights = resolve_scale_weights(scale_weights, scale_gamma, num_scales)
    return RoutedDepthObjective(
        single=VideoDepthLoss(pose_flag=bool(pose_flag)),
        multiscale=MultiScaleVideoL1Loss(
            loss_type=str(loss_type), scale_weights=weights),
        description=(
            f"legacy_l2(single=video, multiscale={loss_type}, "
            f"pose_flag={bool(pose_flag)}, scale_weights={weights})"),
    )
