# Pluggable vision backbones for the depth model.
#
# Wraps timm DINOv3 backbones behind a uniform interface so the DPT head can consume
# multi-scale NCHW feature maps regardless of whether the backbone is an (isotropic) ViT
# or a (hierarchical) ConvNeXt. The frozen backbone can be LoRA-finetuned.
#
# Each backbone exposes:
#   .forward(x) -> list[Tensor]   # 4 feature maps, NCHW, coarse->fine order matching DPT
#   .embed_dims                   # list[int], channels of the 4 feature maps
#   .is_hierarchical              # True for ConvNeXt (4 different strides), False for ViT
#   .patch_size / .feat_strides   # spatial reduction info
#
# Verified with timm 1.0.27:
#   ViT-S+   : vit_small_plus_patch16_dinov3.lvd1689m  -> 4x (B,384,H/16,W/16)
#   ConvNeXt : convnext_small.dinov3_lvd1689m          -> [(96,H/4),(192,H/8),(384,H/16),(768,H/32)]

import torch
import torch.nn as nn

import timm

from .util.lora import inject_lora


# name -> (timm_model, kind, lora_targets)
_REGISTRY = {
    'dinov3_vitsplus':   ('vit_small_plus_patch16_dinov3.lvd1689m', 'vit',      ('qkv', 'proj')),
    'dinov3_vits':       ('vit_small_patch16_dinov3.lvd1689m',      'vit',      ('qkv', 'proj')),
    'dinov3_convnext_s': ('convnext_small.dinov3_lvd1689m',         'convnext', ('fc1', 'fc2')),
    'dinov3_convnext_t': ('convnext_tiny.dinov3_lvd1689m',          'convnext', ('fc1', 'fc2')),
}


class _ViTBackbone(nn.Module):
    """Isotropic ViT: 4 evenly-spaced blocks, all at stride = patch_size."""

    is_hierarchical = False

    def __init__(self, model, patch_size=16):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        depth = len(model.blocks)
        # DAv2-style spread of 4 taps across depth (e.g. depth 12 -> [2,5,8,11]).
        self.indices = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        dim = model.embed_dim
        self.embed_dims = [dim, dim, dim, dim]
        self.feat_strides = [patch_size] * 4

    def forward(self, x):
        feats = self.model.forward_intermediates(
            x, indices=self.indices, norm=False,
            output_fmt='NCHW', intermediates_only=True)
        return list(feats)


class _ConvNeXtBackbone(nn.Module):
    """Hierarchical ConvNeXt: 4 stage outputs at strides 4/8/16/32."""

    is_hierarchical = True

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.patch_size = 4
        self.indices = [0, 1, 2, 3]
        # ConvNeXt tiny/small both use these stage widths.
        self.embed_dims = [96, 192, 384, 768]
        self.feat_strides = [4, 8, 16, 32]

    def forward(self, x):
        feats = self.model.forward_intermediates(
            x, indices=self.indices, norm=False,
            output_fmt='NCHW', intermediates_only=True)
        return list(feats)


def _load_vit(timm_name, weights):
    # No explicit path means: use timm's official Hugging Face pretrained weights.
    # A local path overrides the source while keeping timm's checkpoint handling.
    kw = dict(pretrained=True, num_classes=0)
    if weights:
        kw['pretrained_cfg_overlay'] = dict(file=weights)
    model = timm.create_model(timm_name, **kw)
    return _ViTBackbone(model, patch_size=16)


def _load_convnext(timm_name, weights):
    # Raw DINOv3 ConvNeXt ckpt carries extra per-stage norms (norms.3) not in timm's model,
    # so a local raw checkpoint is loaded non-strict through timm's remap filter.
    # Without a path, let timm download and load its converted official HF weights.
    from timm.models.convnext import checkpoint_filter_fn
    if not weights:
        model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        return _ConvNeXtBackbone(model)

    model = timm.create_model(timm_name, pretrained=False, num_classes=0)
    raw = torch.load(weights, map_location='cpu', weights_only=False)
    raw = raw.get('model', raw) if isinstance(raw, dict) else raw
    filt = checkpoint_filter_fn(raw, model)
    missing, unexpected = model.load_state_dict(filt, strict=False)
    print(f"[backbone] convnext loaded: missing={len(missing)} unexpected={len(unexpected)}")
    return _ConvNeXtBackbone(model)


def build_backbone(name, weights=None, lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.0):
    """Build a backbone by registry name, optionally loading weights and injecting LoRA.

    Returns the backbone module (frozen base if lora=True; caller enables lora params).
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Options: {list(_REGISTRY)}")
    timm_name, kind, lora_targets = _REGISTRY[name]

    if kind == 'vit':
        bb = _load_vit(timm_name, weights)
    elif kind == 'convnext':
        bb = _load_convnext(timm_name, weights)
    else:
        raise ValueError(kind)

    if lora:
        n = inject_lora(bb.model, r=lora_r, alpha=lora_alpha,
                        dropout=lora_dropout, targets=tuple(lora_targets))
        print(f"[backbone] {name}: injected LoRA into {n} layers, targets={lora_targets}")
    return bb
