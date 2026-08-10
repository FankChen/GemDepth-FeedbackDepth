import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.videoloss import (
    TrimmedProcrustesLoss,
    TemporalGradientMatchingLoss,
    TrimmedMAELoss,
    Cameraloss,
    compute_scale_and_shift,
)


class MultiScaleVideoDepthLoss(nn.Module):
    """Multi-scale counterpart of :class:`~loss.videoloss.VideoDepthLoss`.

    The coarse-to-fine refinement head (``DPTHeadMultiScaleRefine``) emits a
    *list* of per-scale depth predictions instead of a single map. This loss
    applies the same spatial (SSI + multi-scale gradient) and temporal-stability
    supervision to every scale and averages them with per-scale weights, while
    the camera/pose loss is computed only once (it is resolution independent).

    ``forward`` accepts either a single tensor (``(B, T, H, W)``) or a list of
    such tensors (coarse -> fine). Each element may be at a different spatial
    resolution; the ground-truth inverse depth and mask are nearest-neighbour
    resized to match before the loss is computed, so depth discontinuities are
    never bilinearly interpolated.
    """

    def __init__(self, alpha=0.5, beta=0.2, scales=4, trim=0, stable_scale=10,
                 reduction="batch-based", pose_flag=True, scale_weights=None,
                 normalize_scale_weights=True):
        super().__init__()
        self.beta = beta
        self.spatial_loss = TrimmedProcrustesLoss(alpha=alpha, scales=scales, trim=trim, reduction=reduction)
        self.stable_loss = TemporalGradientMatchingLoss(trim=trim, reduction=reduction, temp_grad_decay=0.5, temp_grad_scales=1)
        self.camera_loss = Cameraloss()
        self.stable_scale = stable_scale
        self.data_loss = TrimmedMAELoss(trim=trim, reduction=reduction)
        self.pose_flag = pose_flag
        # Optional per-scale weights (coarse -> fine). None => uniform average.
        self.scale_weights = scale_weights
        self.normalize_scale_weights = normalize_scale_weights

    def _resize_gt(self, target_inverse, mask, size):
        """Nearest-resize GT inverse depth and mask to ``size`` = (h, w)."""
        B, T, _, _ = target_inverse.shape
        tinv = target_inverse.flatten(0, 1).unsqueeze(1)
        m = mask.flatten(0, 1).unsqueeze(1)
        tinv = F.interpolate(tinv, size=size, mode='nearest').squeeze(1).unflatten(0, (B, T))
        m = F.interpolate(m, size=size, mode='nearest').squeeze(1).unflatten(0, (B, T))
        return tinv, m

    def forward(self, prediction, target, mask, intrinsic_gt, extrinsic_gt, pose_enc_list, extrinsic_pred):
        # Accept a single tensor or a list/tuple of per-scale predictions.
        if torch.is_tensor(prediction):
            predictions = [prediction]
        else:
            predictions = list(prediction)
        n_scales = len(predictions)
        if n_scales == 0:
            raise ValueError("MultiScaleVideoDepthLoss received an empty prediction list")

        if self.scale_weights is None:
            weights = [1.0 / n_scales] * n_scales
        else:
            if len(self.scale_weights) != n_scales:
                raise ValueError(
                    f"scale_weights length {len(self.scale_weights)} != number of scales {n_scales}")
            weights = [float(w) for w in self.scale_weights]
            if self.normalize_scale_weights:
                weight_sum = float(sum(weights))
                if weight_sum <= 0:
                    raise ValueError("scale_weights must have a positive sum")
                weights = [w / weight_sum for w in weights]

        target = target.clone()
        target_inverse = torch.zeros_like(target)
        valid_mask = target > 0
        target_inverse[valid_mask] = 1.0 / target[valid_mask]

        loss_dict = {}
        total = 0
        agg_spatial = agg_ssi = agg_gm = agg_stable = 0

        for i, pred in enumerate(predictions):
            w = weights[i]
            pred = torch.clamp(pred, min=5e-3, max=1500)

            # Match GT resolution to this scale (nearest, to preserve depth edges).
            if pred.shape[-2:] != target.shape[-2:]:
                tinv, m = self._resize_gt(target_inverse, mask, pred.shape[-2:])
            else:
                tinv, m = target_inverse, mask

            # Spatial (SSI + multi-scale gradient) loss in inverse-depth space.
            spatial, ssi, gm = self.spatial_loss(
                prediction=pred.flatten(0, 1), target=tinv.flatten(0, 1), mask=m.flatten(0, 1).float())

            # Temporal-stability loss after a per-sample scale+shift alignment.
            scale, shift = compute_scale_and_shift(pred.flatten(1, 2), tinv.flatten(1, 2), m.flatten(1, 2))
            pred_aligned = scale.view(-1, 1, 1, 1) * pred + shift.view(-1, 1, 1, 1)
            stable = self.stable_loss(prediction=pred_aligned, target=tinv, mask=m) * self.stable_scale
            scale_total = spatial + stable

            # Keep the unweighted loss of every prediction level visible for
            # diagnostics. Weighting is applied only to the aggregate below.
            loss_dict[f'scale_{i}_spatial_loss'] = spatial
            loss_dict[f'scale_{i}_ssi'] = ssi
            loss_dict[f'scale_{i}_gm'] = gm
            loss_dict[f'scale_{i}_stable_loss'] = stable
            loss_dict[f'scale_{i}_total_loss'] = scale_total

            total = total + w * scale_total
            agg_spatial = agg_spatial + w * spatial
            agg_ssi = agg_ssi + w * ssi
            agg_gm = agg_gm + w * gm
            agg_stable = agg_stable + w * stable

        loss_dict['spatial_loss'] = agg_spatial
        loss_dict['ssi'] = agg_ssi
        loss_dict['gm'] = agg_gm
        loss_dict['stable_loss'] = agg_stable

        # Camera/pose loss is resolution independent -> compute once on the
        # finest (full-resolution) prediction against the full-resolution GT.
        if self.pose_flag:
            extrinsic_gt = torch.stack(extrinsic_gt, dim=1)
            finest = torch.clamp(predictions[-1], min=5e-3, max=1500)
            loss_dict['pose_loss'], loss_dict['trans'], loss_dict['quat'] = self.camera_loss(
                finest, target, mask, extrinsic_gt, intrinsic_gt, extrinsic_pred, pose_enc_list)
            total = total + loss_dict['pose_loss'] * self.beta

        loss_dict['total_loss'] = total
        return loss_dict
