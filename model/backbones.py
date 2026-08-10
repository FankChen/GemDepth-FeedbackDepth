import torch
import torch.nn as nn
import timm

from .backbone_registry import register
from .dinov2 import DINOv2, DinoVisionTransformer
from .util.lora import inject_lora


def _inject_lora(module, enabled, r, alpha, dropout, targets, name):
    if not enabled:
        return
    n = inject_lora(
        module, r=r, alpha=alpha, dropout=dropout, targets=tuple(targets))
    print(f"[backbone] {name}: injected LoRA into {n} layers, targets={targets}")


@register
class DINOv2Backbone(DinoVisionTransformer):
    feature_format = "dinov2_tokens"
    is_hierarchical = False
    patch_size = 14

    def __init__(
        self,
        encoder="vitl",
        weights=None,
        pretrained=True,
        lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_targets=("qkv", "proj"),
    ):
        base = DINOv2(model_name=encoder)
        self.__dict__ = base.__dict__.copy()
        self.embed_dims = [self.embed_dim] * 4
        self.feat_strides = [self.patch_size] * 4

        if weights and pretrained:
            if str(weights).startswith("timm://"):
                timm_name = str(weights)[len("timm://"):]
                timm_model = timm.create_model(
                    timm_name, pretrained=True, num_classes=0)
                state = timm_model.state_dict()
                missing, unexpected = self.load_state_dict(state, strict=False)
                if missing != ["mask_token"] or unexpected:
                    raise RuntimeError(
                        f"Unexpected timm DINOv2 mismatch: "
                        f"missing={missing} unexpected={unexpected}")
                del timm_model
                print(
                    f"[backbone] loaded {len(state)} DINOv2 tensors from "
                    f"timm/HF {timm_name}")
            else:
                state = torch.load(
                    weights, map_location="cpu", weights_only=False)
                state = state.get("model", state) if isinstance(state, dict) else state
                self.load_state_dict(state, strict=True)
                print(f"[backbone] loaded DINOv2 weights from {weights}")

        _inject_lora(
            self, lora, lora_r, lora_alpha, lora_dropout,
            lora_targets, self.__class__.__name__)


class _DINOv3ViTBackbone(nn.Module):
    feature_format = "nchw_tokens"
    is_hierarchical = False
    patch_size = 16

    def __init__(
        self,
        timm_name,
        weights=None,
        pretrained=True,
        lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_targets=("qkv", "proj"),
    ):
        super().__init__()
        kwargs = dict(pretrained=pretrained or bool(weights), num_classes=0)
        if weights:
            kwargs["pretrained_cfg_overlay"] = dict(file=weights)
        self.model = timm.create_model(timm_name, **kwargs)
        depth = len(self.model.blocks)
        self.indices = [
            depth // 4 - 1,
            depth // 2 - 1,
            3 * depth // 4 - 1,
            depth - 1,
        ]
        dim = self.model.embed_dim
        self.embed_dims = [dim] * 4
        self.feat_strides = [self.patch_size] * 4
        _inject_lora(
            self.model, lora, lora_r, lora_alpha, lora_dropout,
            lora_targets, self.__class__.__name__)

    def forward(self, x):
        return list(self.model.forward_intermediates(
            x,
            indices=self.indices,
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        ))


@register
class DINOv3ViTSPlusBackbone(_DINOv3ViTBackbone):
    def __init__(
        self,
        weights=None,
        pretrained=True,
        lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_targets=("qkv", "proj"),
    ):
        super().__init__(
            "vit_small_plus_patch16_dinov3.lvd1689m",
            weights=weights,
            pretrained=pretrained,
            lora=lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
        )


@register
class DINOv3ViTSmallBackbone(_DINOv3ViTBackbone):
    def __init__(
        self,
        weights=None,
        pretrained=True,
        lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_targets=("qkv", "proj"),
    ):
        super().__init__(
            "vit_small_patch16_dinov3.lvd1689m",
            weights=weights,
            pretrained=pretrained,
            lora=lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
        )


class _DINOv3ConvNeXtBackbone(nn.Module):
    feature_format = "pyramid"
    is_hierarchical = True
    patch_size = 4

    def __init__(
        self,
        timm_name,
        weights=None,
        pretrained=True,
        lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_targets=("fc1", "fc2"),
    ):
        super().__init__()
        if not weights:
            self.model = timm.create_model(
                timm_name, pretrained=pretrained, num_classes=0)
        else:
            from timm.models.convnext import checkpoint_filter_fn

            self.model = timm.create_model(
                timm_name, pretrained=False, num_classes=0)
            raw = torch.load(weights, map_location="cpu", weights_only=False)
            raw = raw.get("model", raw) if isinstance(raw, dict) else raw
            state = checkpoint_filter_fn(raw, self.model)
            missing, unexpected = self.model.load_state_dict(
                state, strict=False)
            print(
                f"[backbone] ConvNeXt loaded: missing={len(missing)} "
                f"unexpected={len(unexpected)}")

        self.indices = [0, 1, 2, 3]
        self.embed_dims = [96, 192, 384, 768]
        self.feat_strides = [4, 8, 16, 32]
        _inject_lora(
            self.model, lora, lora_r, lora_alpha, lora_dropout,
            lora_targets, self.__class__.__name__)

    def forward(self, x):
        return list(self.model.forward_intermediates(
            x,
            indices=self.indices,
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True,
        ))


@register
class DINOv3ConvNeXtSmallBackbone(_DINOv3ConvNeXtBackbone):
    def __init__(
        self,
        weights=None,
        pretrained=True,
        lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_targets=("fc1", "fc2"),
    ):
        super().__init__(
            "convnext_small.dinov3_lvd1689m",
            weights=weights,
            pretrained=pretrained,
            lora=lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
        )


@register
class DINOv3ConvNeXtTinyBackbone(_DINOv3ConvNeXtBackbone):
    def __init__(
        self,
        weights=None,
        pretrained=True,
        lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_targets=("fc1", "fc2"),
    ):
        super().__init__(
            "convnext_tiny.dinov3_lvd1689m",
            weights=weights,
            pretrained=pretrained,
            lora=lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
        )
