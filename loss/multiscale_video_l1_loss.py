import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleVideoL1Loss(nn.Module):
    """Dead-simple multi-scale depth supervision.

    Directly constrains the predicted depth against the ground-truth depth with
    a masked per-pixel loss (L2 / MSE by default, L1 optional) at every scale.
    No SSI alignment, no gradient/temporal/pose terms — just ``pred`` vs ``gt``.

    ``forward`` accepts either a single tensor ``(B, T, H, W)`` or a list of such
    tensors (coarse -> fine). Each scale may be at a different spatial
    resolution; the GT depth and mask are nearest-neighbour resized to match so
    depth discontinuities are never bilinearly interpolated. Per-scale losses are
    combined with optional weights (uniform average by default).
    """

    def __init__(self, loss_type="l2", scale_weights=None):
        super().__init__()
        if loss_type not in ("l1", "l2"):
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type}")
        self.loss_type = loss_type
        # Optional per-scale weights (coarse -> fine). None => uniform average.
        self.scale_weights = scale_weights

    def _resize_gt(self, target, mask, size):
        """Nearest-resize GT depth and mask to ``size`` = (h, w)."""
        B, T, _, _ = target.shape
        t = F.interpolate(target.flatten(0, 1).unsqueeze(1), size=size, mode='nearest')
        m = F.interpolate(mask.flatten(0, 1).unsqueeze(1), size=size, mode='nearest')
        return t.squeeze(1).unflatten(0, (B, T)), m.squeeze(1).unflatten(0, (B, T))

    def forward(self, prediction, target, mask, intrinsic_gt=None, extrinsic_gt=None, pose_enc_list=None, extrinsic_pred=None):
        # Accept a single tensor or a list/tuple of per-scale predictions.
        if torch.is_tensor(prediction):
            predictions = [prediction]
        else:
            predictions = list(prediction)
        n_scales = len(predictions)
        if n_scales == 0:
            raise ValueError("MultiScaleVideoL1Loss received an empty prediction list")

        if self.scale_weights is None:
            weights = [1.0 / n_scales] * n_scales
        else:
            if len(self.scale_weights) != n_scales:
                raise ValueError(
                    f"scale_weights length {len(self.scale_weights)} != number of scales {n_scales}")
            weight_sum = float(sum(self.scale_weights))
            weights = [float(w) / weight_sum for w in self.scale_weights]

        target = target.float()
        mask = mask.float()

        loss_dict = {}
        total = 0
        for i, pred in enumerate(predictions):
            pred = pred.float()

            # Match GT resolution to this scale (nearest, to preserve depth edges).
            if pred.shape[-2:] != target.shape[-2:]:
                t, m = self._resize_gt(target, mask, pred.shape[-2:])
            else:
                t, m = target, mask

            diff = pred - t
            if self.loss_type == "l2":
                per_pixel = (diff * diff) * m
            else:
                per_pixel = diff.abs() * m
            scale_loss = per_pixel.sum() / m.sum().clamp(min=1.0)

            total = total + weights[i] * scale_loss
            loss_dict[f'scale_{i}'] = scale_loss

        loss_dict['total_loss'] = total
        
        return loss_dict
