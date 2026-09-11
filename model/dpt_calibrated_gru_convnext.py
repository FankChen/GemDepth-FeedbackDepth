"""Calibrated raw+GEV recurrent control, isolated from every historical head.

Matches IGEV-MVS's two independent lookup volumes, hidden initialisation,
coarse->mid->fine GRU updates and per-iteration coordinate detach. This is NOT
an exact official reproduction: C1's backbone/hourglass/upsampler and final-only
absolute-index objective are retained for the matched control, and the index
update is explicitly bounded to the calibrated volume (see bounded_index_update).
No predicted GEM camera, SSI gauge, softmax-as-raw substitution or output floor.
"""

import torch
import torch.nn.functional as F

from model.decoder_registry import register
from model.dpt_calibrated_volume_only_convnext import (
    native_convnext_camera_input, validate_metric_cameras,
)
from model.dpt_cost_volume_convnext import DPTHeadCostVolumeConvNeXt, _resize
from model.util.cost_volume import VolumeIndexer


def bounded_index_update(index, delta_bins, max_index):
    """Remaining-range tanh update; identity at zero, no hard clamp or logit eps.

    Let u=tanh(2*delta/M), M=D-1. A positive u moves a fraction u of the
    remaining distance to M; a negative u moves a fraction -u towards 0.
    Both branches are bounded convex moves, with a unit local slope w.r.t.
    delta at the midpoint. The operation is differentiable almost everywhere
    (like ReLU at delta=0). It is a declared variation, NOT official IGEV's
    unconstrained additive update. Log the additive proposal's violations too;
    bounding a coordinate must not conceal an unstable update network.
    """
    if max_index <= 0 or index.shape != delta_bins.shape:
        raise ValueError("Index update requires matching shapes and at least two bins")
    if not torch.isfinite(index).all() or not torch.isfinite(delta_bins).all():
        raise FloatingPointError("Nonfinite recurrent index/update; no numeric replacement")
    if (index < 0).any() or (index > max_index).any():
        raise ValueError("Current index is outside the physical volume")
    fraction = torch.tanh(delta_bins * (2. / max_index))
    return torch.where(delta_bins >= 0,
                       index + (max_index - index) * fraction,
                       index + index * fraction)


@register
class DPTHeadCalibratedGRUConvNeXt(DPTHeadCostVolumeConvNeXt):
    output_space = "normalized_inverse_depth_index"
    index_update_contract = "remaining_range_tanh_v1"
    # C1 supplies every other key. Missing shared tensors must never be hidden
    # by strict=False or replaced with a trained C1 final checkpoint.
    recurrent_state_prefixes = (
        "hidden_stems.1.", "hidden_stems.2.", "encoder.", "gru_fine.",
        "gru_mid.", "gru_coarse.", "index_head.",
    )

    def __init__(self, *args, iters=8, **kwargs):
        super().__init__(*args, iters=iters, **kwargs)
        if self.iters < 0 or self.iters > 8 or not 0 < self.depth_min < self.depth_max:
            raise ValueError("Calibrated GRU control supports 0..8 iterations and positive ordered bounds")
        if self.num_sample < 2 ** self.corr_levels or self.corr_levels < 1 or self.corr_radius < 0:
            raise ValueError("Invalid depth-volume lookup pyramid")
        self.last_diagnostics = {}
        self.last_iteration_diagnostics = []

    def load_c1_initial(self, shared_state):
        """Strictly map C1's common INITIAL state; leave only declared new modules."""
        current = self.state_dict()
        shared_keys = {key for key in current if not key.startswith(self.recurrent_state_prefixes)}
        if set(shared_state) != shared_keys:
            raise ValueError(
                f"C1 common initial keys differ: missing={sorted(shared_keys - set(shared_state))}, "
                f"unexpected={sorted(set(shared_state) - shared_keys)}")
        for key, value in shared_state.items():
            if value.shape != current[key].shape or value.dtype != current[key].dtype:
                raise ValueError(f"C1 initial tensor shape/dtype differs: {key}")
            current[key] = value.detach().clone()
        self.load_state_dict(current, strict=True)

    def forward(self, out_features, patch_h, patch_w, frame_length, *, images=None,
                extrinsics=None, intrinsics=None, geometry_gauge=None):
        if geometry_gauge != "metric":
            raise ValueError("Explicit geometry_gauge='metric' required; predicted normalized T is not metric")
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

        # Official IGEV-MVS uses a scalar raw correlation (G=1). C1's G=8
        # aggregation is unchanged; its per-group raw scores are averaged ONLY
        # for the scalar direct-lookup branch. Never substitute probability here.
        lookup = VolumeIndexer(logits.unsqueeze(1), raw.mean(dim=1, keepdim=True),
                               num_levels=self.corr_levels, radius=self.corr_radius)
        hidden, state = [], logits
        for stem in self.hidden_stems:
            state = stem(state)
            hidden.append(torch.tanh(state))

        initial_index = index.detach()
        self.last_iteration_diagnostics = []
        maximum = self.num_sample - 1
        lookup_width = 2 * self.corr_radius + 1
        for iteration in range(self.iters):
            index = index.detach()  # Official coordinate detach; volumes/hidden remain differentiable.
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
                raw_samples = torch.cat([sampled[:, (2 * level + 1) * lookup_width:(2 * level + 2) * lookup_width]
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

        prediction = self._to_full_resolution(index, hidden[0], patch_h, patch_w)
        if not torch.isfinite(prediction).all() or (prediction < -1e-6).any() or (prediction > 1. + 1e-6).any():
            raise FloatingPointError("GRU produced invalid normalized physical index")
        self.last_diagnostics = {
            "raw_zero_fraction": float((raw.detach().abs().sum(dim=(1, 2)) == 0).float().mean()),
            "raw_depth_std": float(raw.detach().std(dim=2, unbiased=False).mean()),
            "logit_depth_std": float(logits.detach().std(dim=1, unbiased=False).mean()),
            "initial_index_mean": float(initial_index.mean()), "final_index_mean": float(index.detach().mean()),
            "max_additive_proposal_outside_fraction": max(
                (row["additive_proposal_outside_fraction"] for row in self.last_iteration_diagnostics), default=0.),
            "final_index_boundary_fraction": float(((index.detach() <= 0) | (index.detach() >= maximum)).float().mean()),
        }
        return prediction  # Same single-final-output contract in train/eval; no new loss routing.