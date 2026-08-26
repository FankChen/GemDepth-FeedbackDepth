"""Numerically stable scale/shift alignment for inverse-depth evaluation."""

import numpy as np


def require_finite(values, label, source=''):
    """Return float64 values or raise with the exact offending source."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f'{label} is empty' + (f': {source}' if source else ''))
    finite = np.isfinite(array)
    if not finite.all():
        bad = int((~finite).sum())
        suffix = f': {source}' if source else ''
        raise FloatingPointError(
            f'{label} contains NaN/Inf: bad={bad}/{array.size} '
            f'finite_fraction={finite.mean():.8f}{suffix}')
    return array


def stable_scale_and_shift(prediction, target, variance_eps=1e-12):
    """Fit ``scale * prediction + shift`` with centered finite arithmetic.

    This is equivalent to two-column least squares on well-conditioned inputs,
    but avoids LAPACK/SVD failures. A constant prediction has a defined best
    affine solution: zero scale and target mean. Non-finite inputs are rejected,
    never silently dropped.
    """
    pred = require_finite(prediction, 'SSI prediction').reshape(-1)
    tgt = require_finite(target, 'SSI target').reshape(-1)
    if pred.shape != tgt.shape:
        raise ValueError(
            f'SSI prediction/target size mismatch: {pred.shape} vs {tgt.shape}')

    pred_mean = float(pred.mean())
    target_mean = float(tgt.mean())
    pred_centered = pred - pred_mean
    target_centered = tgt - target_mean

    # Normalize before the covariance calculation so very large yet finite
    # predictions cannot overflow a squared norm.
    centered_scale = float(np.max(np.abs(pred_centered)))
    reference_scale = max(float(np.max(np.abs(pred))), 1.0)
    if centered_scale <= variance_eps * reference_scale:
        scale = 0.0
        shift = target_mean
    else:
        normalized = pred_centered / centered_scale
        variance = float(np.mean(normalized * normalized))
        covariance = float(np.mean(normalized * target_centered))
        if not np.isfinite(variance) or variance <= variance_eps:
            scale = 0.0
            shift = target_mean
        else:
            scale = covariance / variance / centered_scale
            shift = target_mean - scale * pred_mean

    if not np.isfinite(scale) or not np.isfinite(shift):
        raise FloatingPointError(
            f'Non-finite SSI solution: scale={scale}, shift={shift}')
    return float(scale), float(shift)
