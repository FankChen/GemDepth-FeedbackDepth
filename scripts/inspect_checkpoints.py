"""Verify that every trained checkpoint matches the config it claims to come from.

Rebuilds each model from ``config/vkitti/<name>.yaml`` and loads
``checkpoint/<name>/final_model.pth`` into it. A clean run means the weights on disk
really are the architecture the config describes -- no silently mismatched decoder, no
GEM that was configured but never built, no head that changed after the run finished.

Any missing or unexpected key is a genuine discrepancy: these checkpoints are saved from
the same factory that is used to rebuild here, so an exact match is the expectation, not
an aspiration.

Usage (from repo root, on the machine holding the checkpoints):
    /usr/local/bin/python scripts/inspect_checkpoints.py
    /usr/local/bin/python scripts/inspect_checkpoints.py vkitti_costvol vkitti_ms_gem
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'evaluation', 'inference'))

import torch
from omegaconf import OmegaConf

from model.factory import build_gemdepth_from_config
from protocol import load_experiment_config

CHECKPOINT_ROOT = 'checkpoint'
CONFIG_ROOT = os.path.join('config', 'vkitti')

# Modules that only exist for one decoder family: their presence is a fingerprint of
# which head was actually trained, independent of what the config file says today.
FINGERPRINTS = {
    'cost volume': ('head.cost_agg.', 'head.matcher.', 'head.gru_fine.'),
    'errmap': ('head.error_encoders.',),
    'iterative GRU': ('head.gru_cells.',),
    'multiscale delta': ('head.delta_heads.',),
    'GEM': ('camera_head.', 'camera_token', 'global_blocks.'),
    'ASTT': ('spatial_blocks.', 'time_blocks.'),
}


def _load_state_dict(path):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(blob, dict) and 'model_state_dict' in blob:
        blob = blob['model_state_dict']
    return {key[len('module.'):] if key.startswith('module.') else key: value
            for key, value in blob.items()}


def inspect(name):
    config_path = os.path.join(CONFIG_ROOT, f'{name}.yaml')
    checkpoint_path = os.path.join(CHECKPOINT_ROOT, name, 'final_model.pth')
    if not os.path.exists(config_path):
        return f'{name:34s} SKIP  no config at {config_path}'
    if not os.path.exists(checkpoint_path):
        return f'{name:34s} SKIP  no final_model.pth'

    state = _load_state_dict(checkpoint_path)
    present = [label for label, prefixes in FINGERPRINTS.items()
               if any(key.startswith(prefixes) for key in state)]

    cfg = load_experiment_config(config_path)
    decoder = OmegaConf.select(cfg, 'model.decoder', default='?')
    model = build_gemdepth_from_config(cfg, load_backbone_pretrained=False)
    missing, unexpected = model.load_state_dict(state, strict=False)

    total = sum(v.numel() for v in state.values()) / 1e6
    head = sum(v.numel() for k, v in state.items() if k.startswith('head.')) / 1e6
    verdict = 'OK  ' if not missing and not unexpected else 'DIFF'
    line = (f'{name:34s} {verdict}  {decoder:36s} {total:7.1f}M (head {head:5.1f}M)  '
            f'[{", ".join(present)}]')
    if missing:
        line += f'\n{"":36s} missing({len(missing)}): {list(missing)[:4]}'
    if unexpected:
        line += f'\n{"":36s} unexpected({len(unexpected)}): {list(unexpected)[:4]}'
    return line


def main():
    names = sys.argv[1:]
    if not names:
        names = sorted(entry for entry in os.listdir(CHECKPOINT_ROOT)
                       if os.path.isdir(os.path.join(CHECKPOINT_ROOT, entry))
                       and entry.startswith('vkitti'))
    for name in names:
        try:
            print(inspect(name), flush=True)
        except Exception as exc:  # a broken arm should not hide the healthy ones
            print(f'{name:34s} FAIL  {type(exc).__name__}: {exc}', flush=True)


if __name__ == '__main__':
    main()
