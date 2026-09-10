"""Calibrated, volume-only implementation control; NOT an RGB-only model.

Keep the legacy decoder/checkpoints unchanged. This registered control retains
its matching, aggregation and learned upsampling but has no recurrent updates
or volume lookup. A separate runner supplies calibrated metric cameras directly,
bypassing GEM and GemDepth's relative-depth postprocessing. The explicit gauge
acknowledgement prevents accidental use through the old RGB-only training path.
"""

import torch
import torch.nn as nn

from model.decoder_registry import register
from model.dpt_cost_volume_convnext import DPTHeadCostVolumeConvNeXt


def native_convnext_camera_input(intrinsics, stride):
    """Account for ConvNeXt's non-padded 4x4/2x2 stride-convolution centres.

    A native stride-s feature cell is centred at s*u + (s-1)/2 image pixels.
    The inherited builder applies ratio-only K scaling, so subtract that origin
    BEFORE passing it to the legacy scaler. Do not modify the stored GT camera.
    """
    result = intrinsics.clone()
    result[..., 0, 2] -= (stride - 1) / 2
    result[..., 1, 2] -= (stride - 1) / 2
    return result


def validate_metric_cameras(images, extrinsics, intrinsics, frame_length):
    """Validate numerical/shape contracts, not infer physical units from tensors.

    Off-image principal points are valid. Zero motion is also valid geometry,
    although it supplies no triangulation evidence. Neither should be rejected
    by a camera-validity check.
    """
    if images is None or extrinsics is None or intrinsics is None:
        raise ValueError("Calibrated control requires explicit images and GT K/T")
    if images.ndim != 5 or images.shape[1] != frame_length or frame_length < 2:
        raise ValueError("Expected images (B,T,3,H,W) with T>=2")
    batch, frames = images.shape[:2]
    if tuple(intrinsics.shape) != (batch, frames, 3, 3):
        raise ValueError("Expected per-frame intrinsics (B,T,3,3)")
    if tuple(extrinsics.shape) != (batch, frames, 4, 4):
        raise ValueError("Expected world-to-camera extrinsics (B,T,4,4)")
    if not torch.isfinite(intrinsics).all() or not torch.isfinite(extrinsics).all():
        raise ValueError("Nonfinite K/T: refusing to turn invalid geometry into a zero volume")
    focal = intrinsics[..., (0, 1), (0, 1)]
    if (focal <= 0).any() or (torch.linalg.det(intrinsics.float()).abs() < 1e-8).any():
        raise ValueError("Intrinsics require positive focal lengths and an invertible K")
    if not torch.allclose(intrinsics[..., 2, :], intrinsics.new_tensor([0., 0., 1.]).expand(batch, frames, -1)):
        raise ValueError("Invalid intrinsic homogeneous row")
    if not torch.allclose(extrinsics[..., 3, :], extrinsics.new_tensor([0., 0., 0., 1.]).expand(batch, frames, -1)):
        raise ValueError("Invalid extrinsic homogeneous row")
    rotation = extrinsics[..., :3, :3].float()
    identity = torch.eye(3, device=rotation.device).expand_as(rotation)
    if (not torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=1e-3, rtol=0.)
            or not torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]), atol=1e-3, rtol=0.)):
        raise ValueError("Extrinsic rotations must be proper rotations")


@register
class DPTHeadCalibratedVolumeOnlyConvNeXt(DPTHeadCostVolumeConvNeXt):
    output_space = "normalized_inverse_depth_index"

    def __init__(self, *args, iters=0, **kwargs):
        if iters != 0:
            raise ValueError("This baseline has no GRU; register a separate method after baseline validation")
        super().__init__(*args, iters=0, **kwargs)
        if not (0 < self.depth_min < self.depth_max) or self.num_sample < 2:
            raise ValueError("Require 0 < depth_min < depth_max and at least two bins")
        # Construct common components in the same order as the legacy head, then
        # remove unused recurrent parameters rather than claiming nominal capacity.
        del self.encoder, self.gru_fine, self.gru_mid, self.gru_coarse, self.index_head
        self.hidden_stems = nn.ModuleList([self.hidden_stems[0]])
        self.last_diagnostics = {}

    def forward(self, out_features, patch_h, patch_w, frame_length, *, images=None,
                extrinsics=None, intrinsics=None, geometry_gauge=None):
        if geometry_gauge != "metric":
            raise ValueError("Explicit geometry_gauge='metric' is required; GEM-normalized T is not metric")
        validate_metric_cameras(images, extrinsics, intrinsics, frame_length)
        if any(size % 32 for size in images.shape[-2:]):
            raise ValueError("Calibration control requires image dimensions divisible by 32")
        feature_hw = out_features[self.volume_level].shape[-2:]
        if tuple(feature_hw) != tuple(size // self.volume_stride for size in images.shape[-2:]):
            raise ValueError("Expected native ConvNeXt feature grid at the declared stride")
        sweep_intrinsics = native_convnext_camera_input(intrinsics, self.volume_stride)
        raw = self._build_volume(out_features[self.volume_level], images,
                                 extrinsics, sweep_intrinsics, frame_length)
        guides = [feature.float() for feature in out_features[self.volume_level + 1:]]
        logits = self.classifier(self.cost_agg(self.volume_stem(raw), guides)).squeeze(1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite calibrated volume logits")
        probability = logits.softmax(dim=1)
        bins = torch.arange(self.num_sample, device=logits.device, dtype=logits.dtype)
        index = (probability * bins.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        hidden = torch.tanh(self.hidden_stems[0](logits))
        prediction = self._to_full_resolution(index, hidden, patch_h, patch_w)
        # q=0 is a valid far plane. No ReLU/.005 floor/affine fit is applied here.
        self.last_diagnostics = {
            "raw_zero_fraction": float((raw.detach().abs().sum(dim=(1, 2)) == 0).float().mean()),
            "raw_depth_std": float(raw.detach().std(dim=2, unbiased=False).mean()),
            "logit_depth_std": float(logits.detach().std(dim=1, unbiased=False).mean()),
        }
        return prediction