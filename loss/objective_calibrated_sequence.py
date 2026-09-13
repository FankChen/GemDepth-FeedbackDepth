"""Absolute physical-index L1 with explicit, sum-normalised sequence weights.

IGEV-MVS relative weights: initial=1; iteration i (0-based) =
gamma**(15 * (N-i-1)/(N-1)). Unlike the official unnormalised SUM, this
controlled experiment divides by the total weight. No implicit 9x loss scale,
SSI, affine fit, hidden truncation, or treating one final tensor as nine outputs.
"""

import math

import torch
import torch.nn as nn

from loss.objective_calibrated_index import CalibratedIndexL1Objective
from loss.objective_registry import register


def igev_relative_weights(iterations, gamma=.9):
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 0 <= iterations <= 8:
        raise ValueError("Sequence control requires an integer 0..8 iterations")
    if not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("Gamma must be finite and in (0,1]")
    if iterations == 0:
        return (1.,)
    if iterations == 1:
        return (1., 1.)
    return (1.,) + tuple(gamma ** (15. * (iterations - i - 1) / (iterations - 1))
                        for i in range(iterations))


class CalibratedSequenceIndexL1Objective(nn.Module):
    def __init__(self, depth_min, depth_max, prediction_weights):
        super().__init__()
        self.base = CalibratedIndexL1Objective(depth_min, depth_max)
        self.depth_min, self.depth_max = self.base.depth_min, self.base.depth_max
        self.raw_weights = tuple(float(value) for value in prediction_weights)
        if (not self.raw_weights or not all(math.isfinite(w) and w >= 0 for w in self.raw_weights)
                or not 0 < math.fsum(self.raw_weights) < math.inf):
            raise ValueError("Sequence weights must be nonnegative finite with a positive finite sum")
        self.weights = tuple(w / math.fsum(self.raw_weights) for w in self.raw_weights)
        self.description = f"sum_normalized_absolute_index_l1 weights={self.weights}; no affine fit"

    def forward(self, predictions, target, mask, **_unused):
        if not isinstance(predictions, (tuple, list)) or len(predictions) != len(self.weights):
            raise ValueError(f"Expected exactly {len(self.weights)} initial/iteration predictions in a sequence")
        records = [self.base(q, target, mask) for q in predictions]
        losses = tuple(record["index_l1"] for record in records)
        # Exclude zero-weight graphs entirely: F1 must reproduce legacy final-only
        # gradients, not backpropagate through eight artificial zero-loss paths.
        total = sum(loss * weight for loss, weight in zip(losses, self.weights) if weight > 0)
        return {"total_loss": total, "index_l1": losses[-1],
                "per_prediction_index_l1": torch.stack(losses),
                "valid_pixels": records[-1]["valid_pixels"]}


@register("calibrated_igev_sequence_index_l1")
def build_igev_sequence_objective(depth_min, depth_max, iterations, gamma=.9, pose_flag=False):
    if pose_flag:
        raise ValueError("Calibrated sequence experiment uses GT cameras, not predicted-camera supervision")
    return CalibratedSequenceIndexL1Objective(depth_min, depth_max, igev_relative_weights(iterations, gamma))


@register("calibrated_final_sequence_index_l1")
def build_final_sequence_objective(depth_min, depth_max, iterations, pose_flag=False):
    if pose_flag:
        raise ValueError("Calibrated sequence experiment uses GT cameras")
    igev_relative_weights(iterations)  # Validate the count with the same contract.
    return CalibratedSequenceIndexL1Objective(depth_min, depth_max, (0.,) * iterations + (1.,))