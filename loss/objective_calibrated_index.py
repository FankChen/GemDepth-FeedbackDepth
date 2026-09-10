"""Absolute index supervision for explicitly calibrated depth-volume controls.

Not SSI: an affine change of the prediction changes its physical lookup index.
Camera/depth/bounds MUST share units; the standalone oracle uses metric GT.
"""

import torch
import torch.nn as nn

from loss.objective_registry import register
from loss.objective_video import RoutedDepthObjective


class CalibratedIndexL1Objective(nn.Module):
    def __init__(self, depth_min, depth_max):
        super().__init__()
        self.depth_min, self.depth_max = float(depth_min), float(depth_max)
        if not 0 < self.depth_min < self.depth_max:
            raise ValueError("Require 0 < depth_min < depth_max")
        self.description = f"absolute_index_l1({self.depth_min},{self.depth_max}); no affine fit or prediction floor"

    def forward(self, prediction, target, mask, intrinsic_gt=None, extrinsic_gt=None,
                pose_enc_list=None, extrinsic_pred=None):
        if isinstance(prediction, (list, tuple)):
            raise TypeError("Volume-only baseline expects one prediction, not unregistered iteration weighting")
        strip = RoutedDepthObjective._strip_depth_channel
        prediction, target, mask = strip(prediction), strip(target), strip(mask)
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError("Prediction, metric target and mask must have identical shapes")
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite prediction; refusing to mask a model failure")
        valid = (mask.bool() & torch.isfinite(target)
                 & (target >= self.depth_min) & (target <= self.depth_max))
        if not valid.any():
            raise ValueError("No valid metric target inside the configured depth bounds")
        target_q = ((target[valid].reciprocal() - 1. / self.depth_max)
                    / (1. / self.depth_min - 1. / self.depth_max))
        value = (prediction[valid] - target_q).abs().mean()
        return {"total_loss": value, "index_l1": value,
                "valid_pixels": valid.sum().detach()}


@register("calibrated_index_l1")
def build_calibrated_index_objective(depth_min, depth_max, pose_flag=False):
    if pose_flag:
        raise ValueError("Calibrated oracle uses GT cameras, not an auxiliary predicted-camera loss")
    return CalibratedIndexL1Objective(depth_min, depth_max)