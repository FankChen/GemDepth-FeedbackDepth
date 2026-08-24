"""Backbone-feature adapters.

GEM and ASTT both operate on a single token map -- ``feats[3]``, of shape
``(B*T, L, C)``. Backbone families disagree on what they emit: DINOv2 returns
``(token, cls)`` pairs, DINOv3 ViT returns uniform-stride NCHW maps, and
ConvNeXt returns a hierarchical NCHW pyramid whose levels differ in both stride
and channel count.

An adapter owns that view change in both directions, which is what lets
``GemDepth.forward`` stay free of ``feature_format`` branches. Adapters register
themselves under the ``feature_format`` string their backbone declares, mirroring
the backbone and decoder registries: supporting a new backbone family means
adding an adapter here, not editing the forward pass.

Two grids matter and they are not the same thing:

* the **head patch grid** ``H // backbone.patch_size`` -- the resolution the DPT
  head upsamples from;
* the **geometry grid** ``H // backbone.feat_strides[3]`` -- the resolution of the
  level GEM/ASTT actually run on.

They coincide for uniform-stride ViTs, which is why the distinction stayed
invisible until ConvNeXt (patch_size 4, deepest stride 32) joined.
"""

ADAPTER_REGISTRY = {}


def register_adapter(feature_format):
    """Register an adapter class under the ``feature_format`` it handles."""

    def decorate(cls):
        existing = ADAPTER_REGISTRY.get(feature_format)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"feature_format {feature_format!r} is already handled by "
                f"{existing.__module__}.{existing.__qualname__}")
        ADAPTER_REGISTRY[feature_format] = cls
        return cls

    return decorate


def available_feature_formats():
    return tuple(sorted(ADAPTER_REGISTRY))


def get_adapter(feature_format):
    adapter_cls = ADAPTER_REGISTRY.get(str(feature_format))
    if adapter_cls is None:
        raise ValueError(
            f"Unknown feature_format={feature_format!r}; "
            f"options={available_feature_formats()}")
    return adapter_cls()


class FeatureAdapter:
    """Two-way view change between a backbone and the GEM/ASTT token space."""

    def geometry_grid(self, backbone, height, width):
        """Spatial grid of the level GEM/ASTT run on."""
        stride = backbone.feat_strides[3]
        return height // stride, width // stride

    def encode(self, backbone, x):
        """``(B*T, 3, H, W)`` -> ``(feats, cls_tokens, geometry_hw)``.

        ``feats[3]`` must be ``(B*T, L, C)`` so GEM/ASTT can consume it, with
        ``L == geometry_hw[0] * geometry_hw[1]``.
        """
        raise NotImplementedError

    def decode(self, feats, cls_tokens, geometry_hw):
        """Inverse view change, producing what the DPT head expects."""
        raise NotImplementedError


class _TokenAdapter(FeatureAdapter):
    """Shared by the backbones whose levels are already uniform token maps."""

    def decode(self, feats, cls_tokens, geometry_hw):
        return tuple((feat.float(), cls.float())
                     for feat, cls in zip(feats, cls_tokens))


@register_adapter("dinov2_tokens")
class DINOv2TokenAdapter(_TokenAdapter):
    """DINOv2 already hands back ``(token, cls)`` pairs per selected layer."""

    def encode(self, backbone, x):
        raw = backbone.get_intermediate_layers(
            x, backbone.indices, return_class_token=True)
        feats = [token for token, _ in raw]
        cls_tokens = [cls for _, cls in raw]
        return feats, cls_tokens, self.geometry_grid(backbone, *x.shape[-2:])


@register_adapter("nchw_tokens")
class NCHWTokenAdapter(_TokenAdapter):
    """DINOv3 ViT: uniform NCHW maps flattened to tokens.

    The DPT head's ``(token, cls)`` API is satisfied with a zero class token,
    which these backbones do not expose and the head does not read.
    """

    def encode(self, backbone, x):
        feats, cls_tokens = [], []
        for feature in backbone(x):
            token = feature.flatten(2).permute(0, 2, 1).contiguous()
            feats.append(token)
            cls_tokens.append(token.new_zeros((token.shape[0], token.shape[2])))
        return feats, cls_tokens, self.geometry_grid(backbone, *x.shape[-2:])


@register_adapter("pyramid")
class PyramidAdapter(FeatureAdapter):
    """ConvNeXt: keep the hierarchical NCHW pyramid, expose only its top level.

    Level 3 is the one GEM/ASTT touch, and the only one small enough for full
    spatial attention (14x14 at crop 448, against 112x112 for level 0), so it is
    the only level that changes view. ``decode`` folds it back to NCHW because
    the hierarchical DPT head consumes native feature maps.
    """

    def encode(self, backbone, x):
        pyramid = [feature.float() for feature in backbone(x)]
        top = pyramid[3]
        tokens = top.flatten(2).permute(0, 2, 1).contiguous()
        # ConvNeXt has no class token; the hierarchical head never reads one.
        cls_tokens = [None] * len(pyramid)
        return pyramid[:3] + [tokens], cls_tokens, top.shape[-2:]

    def decode(self, feats, cls_tokens, geometry_hw):
        height, width = geometry_hw
        top = feats[3]
        top = top.permute(0, 2, 1).reshape(
            top.shape[0], -1, height, width).contiguous()
        return [feature.float() for feature in feats[:3]] + [top.float()]
