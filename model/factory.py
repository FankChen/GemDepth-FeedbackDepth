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
        # 【多尺度头三个消融开关】读取自 config 的 model.* 字段，透传给
        # DPTHeadMultiScaleRefineConvNeXt（见 model/dpt_multiscale_convnext.py 文件头的实验对照表）：
        #   multiscale_fullres_mode:   'none'/'last'/'all'/'all_native' -> C/E2a, C_fullres, E1/E2b, E3/E4
        #   multiscale_depth_feedback: 是否让 delta_head 看到当前累积深度 -> E2a/E2b/E4 用 true
        #   multiscale_fp32_head:      delta_head 卷积是否强制 FP32（对齐 baseline） -> 仅 fp32 对照组用 true
        'multiscale_native_res': bool(OmegaConf.select(cfg, 'model.multiscale_native_res', default=True)),
        'multiscale_fullres_mode': str(OmegaConf.select(cfg, 'model.multiscale_fullres_mode', default='none')),
        'multiscale_depth_feedback': bool(OmegaConf.select(cfg, 'model.multiscale_depth_feedback', default=False)),
        'multiscale_fp32_head': bool(OmegaConf.select(cfg, 'model.multiscale_fp32_head', default=False)),
    }


def build_gemdepth_from_config(cfg, load_backbone_pretrained=True):
    return GemDepth(**gemdepth_kwargs_from_config(
        cfg, load_backbone_pretrained=load_backbone_pretrained))
