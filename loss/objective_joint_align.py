"""CARVE-style joint per-frame and per-sequence depth supervision.

The project already has a per-frame affine-invariant term: predictions are
flattened over B*T, then robust-normalised independently for every frame.  What
is missing is a regression term after one affine fit shared by the full clip.
This module adds that term; it does not replace the existing frame objective.

CARVE reports that the pair works better than sequence-only supervision, while
local-region alignment hurts.  No local alignment is implemented here.
"""

import torch

from loss.multiscale_videoloss import MultiScaleVideoDepthLoss
from loss.objective_registry import register
from loss.objective_video import RoutedDepthObjective, resolve_scale_weights
from loss.videoloss import VideoDepthLoss, compute_scale_and_shift


def sequence_aligned_l1_inverse(prediction, target_inverse, mask):
    """L1 inverse-depth error after one scale/shift fit for the whole clip."""
    prediction = torch.clamp(prediction, min=5e-3, max=1500)
    scale, shift = compute_scale_and_shift(
        prediction.flatten(1, 2), target_inverse.flatten(1, 2),
        mask.flatten(1, 2))
    aligned = (scale.view(-1, 1, 1, 1) * prediction
               + shift.view(-1, 1, 1, 1))
    weighted = (aligned - target_inverse).abs() * mask
    return weighted.sum() / mask.sum().clamp(min=1.0)


def sequence_aligned_l1(prediction, target, mask):
    target_inverse = torch.zeros_like(target)
    valid = target > 0
    target_inverse[valid] = 1.0 / target[valid]
    return sequence_aligned_l1_inverse(prediction, target_inverse, mask)


class JointAlignVideoDepthLoss(VideoDepthLoss):
    def __init__(self, sequence_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.sequence_weight = float(sequence_weight)

    def forward(self, prediction, target, mask, intrinsic_gt, extrinsic_gt,
                pose_enc_list, extrinsic_pred):
        losses = super().forward(
            prediction, target, mask, intrinsic_gt, extrinsic_gt,
            pose_enc_list, extrinsic_pred)
        sequence = sequence_aligned_l1(prediction, target, mask)
        losses["sequence_loss"] = sequence
        losses["total_loss"] = (
            losses["total_loss"] + self.sequence_weight * sequence)
        return losses


class JointAlignMultiScaleVideoDepthLoss(MultiScaleVideoDepthLoss):
    def __init__(self, sequence_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.sequence_weight = float(sequence_weight)

    def _weights(self, count):
        if self.scale_weights is None:
            return [1.0 / count] * count
        weights = [float(weight) for weight in self.scale_weights]
        if self.normalize_scale_weights:
            weight_sum = float(sum(weights))
            if weight_sum <= 0:
                raise ValueError("scale_weights must have a positive sum")
            weights = [weight / weight_sum for weight in weights]
        return weights

    def forward(self, prediction, target, mask, intrinsic_gt, extrinsic_gt,
                pose_enc_list, extrinsic_pred):
        predictions = [prediction] if torch.is_tensor(prediction) else list(prediction)
        losses = super().forward(
            predictions, target, mask, intrinsic_gt, extrinsic_gt,
            pose_enc_list, extrinsic_pred)

        target_inverse = torch.zeros_like(target)
        valid = target > 0
        target_inverse[valid] = 1.0 / target[valid]
        aggregate = target.new_zeros(())
        for index, (weight, pred) in enumerate(
                zip(self._weights(len(predictions)), predictions)):
            if pred.shape[-2:] != target.shape[-2:]:
                resized_inverse, resized_mask = self._resize_gt(
                    target_inverse, mask, pred.shape[-2:])
            else:
                resized_inverse = target_inverse
                resized_mask = mask
            sequence = sequence_aligned_l1_inverse(
                pred, resized_inverse, resized_mask)
            losses[f"scale_{index}_sequence_loss"] = sequence
            aggregate = aggregate + float(weight) * sequence

        losses["sequence_loss"] = aggregate
        losses["total_loss"] = (
            losses["total_loss"] + self.sequence_weight * aggregate)
        return losses


@register("joint_align")
def build_joint_align_objective(
        pose_flag, scale_weights=None, scale_gamma=None, num_scales=4,
        normalize_scale_weights=True, alpha=0.5, beta=0.2, scales=4,
        trim=0, stable_scale=10.0, sequence_weight=1.0,
        camera_weight_focal=0.0):
    weights = resolve_scale_weights(scale_weights, scale_gamma, num_scales)
    common = dict(
        alpha=float(alpha), beta=float(beta), scales=int(scales),
        trim=float(trim), stable_scale=float(stable_scale),
        pose_flag=bool(pose_flag), sequence_weight=float(sequence_weight),
        camera_weight_focal=float(camera_weight_focal))
    return RoutedDepthObjective(
        single=JointAlignVideoDepthLoss(**common),
        multiscale=JointAlignMultiScaleVideoDepthLoss(
            **common, scale_weights=weights,
            normalize_scale_weights=bool(normalize_scale_weights)),
        description=(
            f"joint_align(frame_ssi=1, sequence={float(sequence_weight)}, "
            f"alpha={float(alpha)}, stable_scale={float(stable_scale)}, "
            f"pose_flag={bool(pose_flag)})"),
    )
