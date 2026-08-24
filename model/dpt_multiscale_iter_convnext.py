# Iterative multi-scale refine head with weight sharing across rounds (path C).
#
# RAFT/IGEV-style stereo does not get its accuracy from a deep head; it walks a
# *shared* update operator many times, carrying a hidden state. Two properties
# matter and they are separable:
#
#   1. more update steps,
#   2. the same weights at every step, so step count costs no parameters.
#
# The base head has neither: four scales, four separate delta heads, one pass.
# This variant keeps a ConvGRU per scale and re-enters the ladder
# ``refine_rounds`` times, reusing those cells every round. Parameter count is
# therefore independent of ``refine_rounds``, which is exactly what is needed to
# tell "iterating helps" apart from "a bigger head helps" -- the confound that
# currently explains the in-domain gain of the base head.
#
# Note what this variant deliberately does NOT add: new evidence. The pyramid
# features are identical in every round, so the only thing that changes is the
# running depth and the hidden state. That makes it an open loop, and a useful
# null hypothesis: if extra rounds help here, it is the recurrence itself; if
# they do not, the missing ingredient is a depth-dependent measurement (see the
# error-map head).

import torch
import torch.nn as nn

from model.decoder_registry import register
from model.dpt_multiscale_convnext import DPTHeadMultiScaleRefineConvNeXt


class ConvGRUCell(nn.Module):
    """Standard RAFT convolutional GRU: update gate, reset gate, candidate."""

    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1.0 - z) * h + z * q


@register
class DPTHeadMultiScaleIterConvNeXt(DPTHeadMultiScaleRefineConvNeXt):
    """Multi-scale refine head that iterates a shared ConvGRU update operator."""

    def __init__(self, *args, refine_rounds=4, hidden_dim=64, **kwargs):
        super().__init__(*args, refine_rounds=refine_rounds, **kwargs)

        # The GRU consumes exactly what the base delta head consumed: the
        # projected path feature concatenated with the encoded running depth.
        input_dim = self.output_conv1_heads[0].out_channels + self.depth_encoder[0].out_channels
        self.hidden_dim = int(hidden_dim)
        self.gru_cells = nn.ModuleList([
            ConvGRUCell(self.hidden_dim, input_dim) for _ in range(len(self.delta_heads))
        ])
        # Predict the residual from the hidden state instead of from the raw
        # concatenation; replacing the module list keeps the base head's
        # zero-bias initialisation of the last layer.
        self.delta_heads = nn.ModuleList([
            self._make_delta_head(self.hidden_dim) for _ in range(len(self.gru_cells))
        ])
        for delta_head in self.delta_heads:
            nn.init.zeros_(delta_head[-1].bias)

    def _refine_context(self, images, extrinsics, intrinsics, frame_length):
        ctx = super()._refine_context(images, extrinsics, intrinsics, frame_length)
        # One hidden state per scale, carried across rounds (not across samples).
        ctx['hidden'] = [None] * len(self.gru_cells)
        return ctx

    def _predict_delta(self, scale_index, feat, depth_prev, output_size, ctx=None):
        update = self._delta_input(scale_index, feat, depth_prev, output_size)
        hidden = None if ctx is None else ctx['hidden'][scale_index]
        if hidden is None:
            hidden = update.new_zeros(
                update.shape[0], self.hidden_dim, *update.shape[-2:])
        hidden = self.gru_cells[scale_index](hidden, update)
        if ctx is not None:
            ctx['hidden'][scale_index] = hidden
        return self.delta_heads[scale_index](hidden)
