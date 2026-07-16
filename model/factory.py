"""Shared GemDepth construction from an OmegaConf experiment config.

Training, smoke tests, and inference must use this single mapping so checkpoint
structure cannot drift between entry points.
"""

from omegaconf import OmegaConf

from model.gemdepth import GemDepth


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


def gemdepth_kwargs_from_config(cfg, load_backbone_pretrained=True):
    encoder = str(cfg.encoder)
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder={encoder}; options={list(MODEL_CONFIGS)}")
    return {
        **MODEL_CONFIGS[encoder],
        'head_type': str(OmegaConf.select(cfg, 'model.head_type', default='temporal')),
        'use_gem': bool(OmegaConf.select(cfg, 'model.use_gem', default=True)),
        'use_astt': bool(OmegaConf.select(cfg, 'model.use_astt', default=True)),
        'use_temporal': bool(OmegaConf.select(cfg, 'model.use_temporal', default=True)),
        'lora': bool(OmegaConf.select(cfg, 'model.lora', default=False)),
        'lora_r': int(OmegaConf.select(cfg, 'model.lora_r', default=8)),
        'lora_alpha': int(OmegaConf.select(cfg, 'model.lora_alpha', default=16)),
        'lora_dropout': float(OmegaConf.select(cfg, 'model.lora_dropout', default=0.0)),
        'dinov2_weights': OmegaConf.select(cfg, 'model.dinov2_weights', default=None),
        'backbone': str(OmegaConf.select(cfg, 'model.backbone', default='dinov2')),
        'backbone_weights': OmegaConf.select(cfg, 'model.backbone_weights', default=None),
        'load_backbone_pretrained': bool(load_backbone_pretrained),
        'error_signal': str(OmegaConf.select(cfg, 'model.error_signal', default='rgb')),
        'warp_offsets': tuple(OmegaConf.select(cfg, 'model.warp_offsets', default=[-1, 1])),
        'metric_depth_mode': str(OmegaConf.select(
            cfg, 'model.metric_depth_mode', default='softplus')),
        'metric_init_depth': float(OmegaConf.select(
            cfg, 'model.metric_init_depth', default=20.0)),
        'metric_min_depth': float(OmegaConf.select(
            cfg, 'model.metric_min_depth', default=0.1)),
        'metric_max_depth': float(OmegaConf.select(
            cfg, 'model.metric_max_depth', default=200.0)),
    }


def build_gemdepth_from_config(cfg, load_backbone_pretrained=True):
    return GemDepth(**gemdepth_kwargs_from_config(
        cfg, load_backbone_pretrained=load_backbone_pretrained))
