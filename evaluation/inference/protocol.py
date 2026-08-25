"""Inference protocol selection and fail-fast output validation."""

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / 'config'


def load_experiment_config(path):
    """Load a config exactly the way training composes it.

    Training goes through hydra, which resolves ``defaults:``; a plain
    ``OmegaConf.load`` does not. For a config that inherits its backbone and
    decoder from a base file that difference is silent and total: evaluation
    rebuilds a *different* model, the checkpoint's tensors all land in
    ``missing``, and the run reports scores for randomly initialised weights
    instead of failing. Compose whenever the file declares ``defaults``.
    """
    path = Path(path).resolve()
    raw = OmegaConf.load(path)
    if 'defaults' not in raw:
        return raw

    name = path.relative_to(_CONFIG_ROOT).with_suffix('').as_posix()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_ROOT)):
        return compose(config_name=name)


def resolve_inference_clip_len(cfg):
    """Return the training clip length for scratch encoder+decoder models.

    Original GemDepth checkpoints keep the historical overlapping 32-frame
    protocol (``None``). Scratch models must be evaluated with exactly the T used
    in training; otherwise temporal modules see a distribution they never
    trained on and cross-window affine alignment changes the experiment.

    ``encoder_decoder_only`` is only a proxy for "we trained this from scratch",
    and it stops being one as soon as an experiment needs GEM: train.py forbids
    the two together, so a scratch GEM run has to set it false and would silently
    fall back to the 32-frame protocol despite being trained on T=4. Such a
    config states its clip length under ``inference.clip_len`` instead. The
    proxy is left in place for everything else because switching the whole rule
    over to ``seq_len`` would change the protocol -- and therefore the published
    numbers -- of the single_a100 and vkitti_baseline experiments.
    """
    explicit = OmegaConf.select(cfg, 'inference.clip_len', default=None)
    if explicit is not None:
        clip_len = int(explicit)
        return clip_len if clip_len > 0 else None

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
