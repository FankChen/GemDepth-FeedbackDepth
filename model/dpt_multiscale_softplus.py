"""Softplus-output variant of the raw-delta multiscale refine heads.

Fixes the from-scratch catastrophic collapse (typical frame all-zero + a few
exploding pixels) WITHOUT changing the multiscale design: the signed per-scale
delta accumulation, the cross-scale ``.detach()`` gradient truncation, and the
per-scale auxiliary supervision are ALL IDENTICAL to the parent heads. The only
change is that the final inverse-depth output is squashed through ``softplus``.

Why this stops the collapse
---------------------------
The parent heads return a raw, signed accumulated depth that ``GemDepth.forward``
then passes through a hard ``F.relu`` (inverse depth must be >= 0). ``relu`` has a
DEAD ZONE: once a pixel's raw value drifts negative, ``relu`` zeros both the value
AND its gradient, so it is stuck at 0 forever. Because the main + aux losses are
scale-shift invariant (SSI), nothing anchors the raw output's absolute level, so
over training it drifts into that dead zone and collapses.

``softplus(x)`` is strictly positive AND has a non-zero gradient everywhere (its
derivative is ``sigmoid(x) > 0``), so there is NO dead zone: even if the raw value
drifts negative, gradient still flows and the loss can pull it back. Because
``softplus(...) > 0`` always, the downstream ``F.relu`` in ``GemDepth.forward``
becomes a harmless identity, so nothing else has to change.

Note
----
Only the returned final depth is squashed. ``self.aux_depths`` (set by the parent
forward) stay raw, exactly as before, so the auxiliary supervision is unchanged.
"""
import torch.nn.functional as F

from model.dpt_multiscale import DPTHeadMultiScaleRefine
from model.dpt_multiscale_convnext import DPTHeadMultiScaleRefineConvNeXt


class DPTHeadMultiScaleRefineSoftplus(DPTHeadMultiScaleRefine):
    """ViT raw-delta multiscale head with a strictly-positive (softplus) output."""

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if isinstance(out, (list, tuple)):
            return [F.softplus(d) for d in out]
        return F.softplus(out)


class DPTHeadMultiScaleRefineConvNeXtSoftplus(DPTHeadMultiScaleRefineConvNeXt):
    """ConvNeXt raw-delta multiscale head with a strictly-positive (softplus) output."""

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if isinstance(out, (list, tuple)):
            return [F.softplus(d) for d in out]
        return F.softplus(out)
