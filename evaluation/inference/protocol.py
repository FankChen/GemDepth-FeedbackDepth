"""Inference protocol selection and fail-fast output validation."""

import numpy as np
from omegaconf import OmegaConf


def resolve_inference_clip_len(cfg):
    """Return the training clip length for scratch encoder+decoder models.

    Original GemDepth checkpoints keep the historical overlapping 32-frame
    protocol (``None``). Scratch models must be evaluated with exactly the T used
    in training; otherwise temporal modules see a distribution they never
    trained on and cross-window affine alignment changes the experiment.
    """
    encoder_decoder_only = bool(OmegaConf.select(
        cfg, 'model.encoder_decoder_only', default=False))
    if not encoder_decoder_only:
        return None
    clip_len = int(OmegaConf.select(cfg, 'dataset.train.seq_len', default=0))
    if clip_len <= 0:
        raise ValueError(
            f'Scratch encoder+decoder inference requires a positive '
            f'dataset.train.seq_len, got {clip_len}')
    return clip_len


def validate_inverse_depth_output(depths, dataset, sequence):
    """Reject malformed/non-finite predictions before writing any NPY files."""
    values = np.asarray(depths)
    context = f'dataset={dataset} sequence={sequence}'
    if values.ndim != 3:
        raise ValueError(
            f'Expected inverse-depth video (T,H,W), got shape={values.shape}; '
            f'{context}')
    if values.size == 0:
        raise ValueError(f'Empty inverse-depth output; {context}')
    finite = np.isfinite(values)
    if not finite.all():
        count = int((~finite).sum())
        raise FloatingPointError(
            f'Non-finite inverse-depth output before save: bad={count}/'
            f'{values.size} finite_fraction={finite.mean():.8f}; {context}')
    return values


def infer_video_with_protocol(model, videos, target_fps, input_size, device,
                              fp32, clip_len, dataset, sequence):
    """Call model inference with the resolved protocol and validate its output."""
    depths, fps = model.infer_video_depth(
        videos, target_fps, input_size=input_size, device=device,
        fp32=fp32, clip_len=clip_len)
    depths = validate_inverse_depth_output(depths, dataset, sequence)
    return depths, fps
