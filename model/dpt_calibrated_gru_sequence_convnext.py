"""Sequence-output controls; historical C1/G1 heads remain byte-for-byte intact.

The recurrent computation and state names match calibrated G1. Only readouts
are added: q0 BEFORE coordinate detach, then q1..q8 BEFORE the next detach.
Training returns all readouts; ordinary evaluation returns the last tensor.
Explicit return_sequence=True is an evaluation diagnostic, not a different
recurrence. No graph-bearing tensors are retained on the module after forward.
"""

import torch
import torch.nn.functional as F

from model.decoder_registry import register
from model.dpt_calibrated_gru_convnext import (
    DPTHeadCalibratedGRUConvNeXt, bounded_index_update,
)
from model.dpt_calibrated_volume_only_convnext import (
    DPTHeadCalibratedVolumeOnlyConvNeXt,
    native_convnext_camera_input, validate_metric_cameras,
)
from model.dpt_cost_volume_convnext import _resize
from model.util.cost_volume import VolumeIndexer


@register
class DPTHeadCalibratedVolumeSequenceConvNeXt(DPTHeadCalibratedVolumeOnlyConvNeXt):
    """No-GRU baseline using the SAME sequence/loss interface, with one q0."""

    recurrent_state_prefixes = ()
    required_gradient_modules = ("matcher", "classifier", "hidden_stems", "upsample_mask")

    def load_c1_initial(self, shared_state):
        self.load_state_dict(shared_state, strict=True)

    def forward(self, out_features, patch_h, patch_w, frame_length, *, images=None,
                extrinsics=None, intrinsics=None, geometry_gauge=None, return_sequence=False):
        q = super().forward(out_features, patch_h, patch_w, frame_length, images=images,
                            extrinsics=extrinsics, intrinsics=intrinsics, geometry_gauge=geometry_gauge)
        return (q,) if self.training or return_sequence else q


@register
class DPTHeadCalibratedGRUSequenceConvNeXt(DPTHeadCalibratedGRUConvNeXt):
    """Same bounded raw+GEV GRU as G1, with differentiable initial/every-step q."""

    required_gradient_modules = (
        "matcher", "classifier", "hidden_stems", "upsample_mask", "encoder",
        "gru_coarse", "gru_mid", "gru_fine", "index_head",
    )

    def _checked_readout(self, index, hidden, patch_h, patch_w):
        q = self._to_full_resolution(index, hidden, patch_h, patch_w)
        if not torch.isfinite(q).all() or (q < -1e-6).any() or (q > 1. + 1e-6).any():
            raise FloatingPointError("Sequence GRU produced invalid physical q; no clamp/floor")
        return q

    def forward(self, out_features, patch_h, patch_w, frame_length, *, images=None,
                extrinsics=None, intrinsics=None, geometry_gauge=None, return_sequence=False):
        if geometry_gauge != "metric":
            raise ValueError("Explicit geometry_gauge='metric' required")
        validate_metric_cameras(images, extrinsics, intrinsics, frame_length)
        if any(size % 32 for size in images.shape[-2:]):
            raise ValueError("Calibrated GRU requires image dimensions divisible by 32")
        expected_hw = tuple(size // self.volume_stride for size in images.shape[-2:])
        if tuple(out_features[self.volume_level].shape[-2:]) != expected_hw:
            raise ValueError("Expected native ConvNeXt feature centres at the declared stride")

        raw = self._build_volume(out_features[self.volume_level], images, extrinsics,
                                 native_convnext_camera_input(intrinsics, self.volume_stride), frame_length)
        guides = [feature.float() for feature in out_features[self.volume_level + 1:]]
        logits = self.classifier(self.cost_agg(self.volume_stem(raw), guides)).squeeze(1)
        if not torch.isfinite(raw).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite calibrated raw/GEV volume")
        probability = logits.softmax(dim=1)
        bins = torch.arange(self.num_sample, device=logits.device, dtype=logits.dtype).view(1, -1, 1, 1)
        index = (probability * bins).sum(dim=1, keepdim=True)
        lookup = VolumeIndexer(logits.unsqueeze(1), raw.mean(dim=1, keepdim=True),
                               num_levels=self.corr_levels, radius=self.corr_radius)
        hidden, state = [], logits
        for stem in self.hidden_stems:
            state = stem(state)
            hidden.append(torch.tanh(state))

        emit_sequence = self.training or return_sequence
        predictions = []
        if emit_sequence:
            # Never detach q0 or its index: this is the direct softargmin loss.
            # Use the INITIAL fine hidden state, not the final recurrent state.
            predictions.append(self._checked_readout(index, hidden[0], patch_h, patch_w))
        initial_index = index.detach()
        self.last_iteration_diagnostics = []
        maximum = self.num_sample - 1
        lookup_width = 2 * self.corr_radius + 1
        for iteration in range(self.iters):
            index = index.detach()  # Lookup coordinates only; volumes/hidden stay differentiable.
            sampled = lookup(index)
            if not torch.isfinite(sampled).all():
                raise FloatingPointError("Nonfinite indexed volume evidence")
            hidden[2] = self.gru_coarse(hidden[2], F.avg_pool2d(hidden[1], 3, 2, 1))
            hidden[1] = self.gru_mid(hidden[1], F.avg_pool2d(hidden[0], 3, 2, 1),
                                    _resize(hidden[2], hidden[1]))
            hidden[0] = self.gru_fine(hidden[0], self.encoder(index, sampled),
                                     _resize(hidden[1], hidden[0]))
            delta = self.index_head(hidden[0])
            proposal = index + delta
            updated = bounded_index_update(index, delta, maximum)
            with torch.no_grad():
                raw_samples = torch.cat([
                    sampled[:, (2 * level + 1) * lookup_width:(2 * level + 2) * lookup_width]
                    for level in range(self.corr_levels)], dim=1)
                self.last_iteration_diagnostics.append({
                    "iteration": iteration + 1, "lookup_index_min": float(index.min()),
                    "lookup_index_max": float(index.max()), "raw_lookup_mean_abs": float(raw_samples.abs().mean()),
                    "delta_mean": float(delta.mean()), "delta_abs_max": float(delta.abs().max()),
                    "additive_proposal_outside_fraction": float(((proposal < 0) | (proposal > maximum)).float().mean()),
                    "accepted_delta_mean": float((updated - index).mean()),
                    "updated_index_min": float(updated.min()), "updated_index_max": float(updated.max()),
                })
            index = updated
            if emit_sequence:
                # Read out BEFORE the next iteration detaches its coordinates.
                predictions.append(self._checked_readout(index, hidden[0], patch_h, patch_w))

        prediction = predictions[-1] if emit_sequence else self._checked_readout(index, hidden[0], patch_h, patch_w)
        self.last_diagnostics = {
            "raw_zero_fraction": float((raw.detach().abs().sum(dim=(1, 2)) == 0).float().mean()),
            "raw_depth_std": float(raw.detach().std(dim=2, unbiased=False).mean()),
            "logit_depth_std": float(logits.detach().std(dim=1, unbiased=False).mean()),
            "initial_index_mean": float(initial_index.mean()), "final_index_mean": float(index.detach().mean()),
            "max_additive_proposal_outside_fraction": max(
                (row["additive_proposal_outside_fraction"] for row in self.last_iteration_diagnostics), default=0.),
            "final_index_boundary_fraction": float(((index.detach() <= 0) | (index.detach() >= maximum)).float().mean()),
        }
        return tuple(predictions) if emit_sequence else prediction