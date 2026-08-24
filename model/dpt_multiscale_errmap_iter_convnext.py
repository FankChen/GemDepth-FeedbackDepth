# Both halves of the RAFT analogy at once.
#
# The iterative head supplies a weight-shared update operator applied over
# several rounds; the error-map head supplies a measurement that changes when the
# depth changes. Neither is expected to carry the method alone:
#
#   * iteration without new evidence is an open loop -- the cheapest fixed point
#     it can learn is the identity;
#   * evidence without iteration is measured once and never re-read, which is
#     the one thing RAFT's cost-volume lookup is for.
#
# Python's cooperative ``super()`` does the composition: the MRO puts the
# error-map feature injection in front of the GRU update, both delegating to the
# shared base loop, so this file adds behaviour without restating either half.

from model.decoder_registry import register
from model.dpt_multiscale_errmap_convnext import DPTHeadMultiScaleErrMapConvNeXt
from model.dpt_multiscale_iter_convnext import DPTHeadMultiScaleIterConvNeXt


@register
class DPTHeadMultiScaleErrMapIterConvNeXt(DPTHeadMultiScaleErrMapConvNeXt,
                                          DPTHeadMultiScaleIterConvNeXt):
    """Iterative multi-scale refinement driven by photometric re-measurement."""
