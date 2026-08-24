# Multi-scale refine head with a differentiable coarse-to-fine chain (path D).
#
# The base head hands each scale ``depth_prev.detach()``, so the four steps are
# four independent predictors that happen to read a running depth: no gradient
# ever crosses a scale boundary. That makes "coarse-to-fine" a data path only,
# not an optimisation path -- a coarse step is never told that its output made
# the next step's job harder.
#
# This variant removes the detach, so the chain trains end-to-end and the ladder
# becomes a genuine update operator. It adds no parameters, which makes it the
# cheapest way to separate "the structure helps" from "the extra capacity helps".

from model.decoder_registry import register
from model.dpt_multiscale_convnext import DPTHeadMultiScaleRefineConvNeXt


@register
class DPTHeadMultiScaleGradConvNeXt(DPTHeadMultiScaleRefineConvNeXt):
    """Multi-scale refine head whose coarse-to-fine chain carries gradients.

    ``carry_scale`` interpolates between the two regimes: 1.0 is fully
    differentiable, 0.0 reproduces the detached base head, and values in between
    scale the gradient that reaches the coarser steps (a straight-through style
    mix that keeps the forward pass identical either way).
    """

    def __init__(self, *args, carry_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.carry_scale = float(carry_scale)

    def _carry(self, depth):
        if self.carry_scale >= 1.0:
            return depth
        if self.carry_scale <= 0.0:
            return depth.detach()
        # Same value, fraction of the gradient: detached part contributes none.
        return self.carry_scale * depth + (1.0 - self.carry_scale) * depth.detach()
