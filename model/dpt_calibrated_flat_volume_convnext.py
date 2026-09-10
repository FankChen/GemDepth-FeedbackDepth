"""Matched C0 control: remove depth-axis variation during BOTH train and eval.

No new parameters or changed initialisation. Keep the complete calibrated sweep,
matcher, view weights and image guides; only average the fused raw depth axis.
This is not a geometry-free or compute-saving baseline. The full-volume C1 is
the unchanged DPTHeadCalibratedVolumeOnlyConvNeXt.
"""

from model.decoder_registry import register
from model.dpt_calibrated_volume_only_convnext import DPTHeadCalibratedVolumeOnlyConvNeXt


@register
class DPTHeadCalibratedFlatVolumeConvNeXt(DPTHeadCalibratedVolumeOnlyConvNeXt):
    def _build_volume(self, features, images, extrinsics, intrinsics, frame_length):
        raw = super()._build_volume(features, images, extrinsics, intrinsics, frame_length)
        return raw.mean(dim=2, keepdim=True).expand_as(raw).contiguous()