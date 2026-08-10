import sys
import unittest
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.backbone_registry import (
    available_backbone_names,
    build_backbone,
    get_backbone_class,
    register,
)


@register
class RegistryTestBackbone(nn.Module):
    feature_format = "test"

    def __init__(self, channels, stages=4):
        super().__init__()
        self.channels = channels
        self.stages = stages


class BackboneRegistryTest(unittest.TestCase):
    def test_registered_class_is_resolved_by_class_name(self):
        self.assertIs(
            get_backbone_class("RegistryTestBackbone"), RegistryTestBackbone)
        self.assertIn(
            "DINOv3ConvNeXtSmallBackbone", available_backbone_names())

    def test_constructor_arguments_are_injected_by_signature(self):
        backbone = build_backbone(
            "RegistryTestBackbone",
            backbone_kwargs={"stages": 3},
            channels=192,
            ignored_context="not forwarded",
        )
        self.assertEqual(backbone.channels, 192)
        self.assertEqual(backbone.stages, 3)

    def test_unknown_backbone_kwarg_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported backbone_kwargs"):
            build_backbone(
                "RegistryTestBackbone",
                backbone_kwargs={"unknown": True},
                channels=192,
            )

    def test_missing_required_argument_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing constructor arguments"):
            build_backbone("RegistryTestBackbone")

    def test_duplicate_class_name_is_rejected(self):
        first = type("RegistryDuplicateBackbone", (nn.Module,), {})
        second = type("RegistryDuplicateBackbone", (nn.Module,), {})
        register(first)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register(second)

    def test_unknown_backbone_lists_options(self):
        with self.assertRaisesRegex(ValueError, "Unknown backbone"):
            get_backbone_class("MissingBackbone")


if __name__ == "__main__":
    unittest.main()
