import sys
import unittest
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.freeze_registry import (
    apply_freeze,
    available_freeze_names,
    get_freeze_policy,
    register,
)
from model.module_groups import GEM_PREFIXES, NEW_MODULE_PREFIXES


@register("registry_test_policy")
def _registry_test_policy(model, use_gem=False):
    return f"saw use_gem={use_gem}"


class _TinyGemDepth(nn.Module):
    """Minimal stand-in exposing the attribute names the policies rely on."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 4)
        self.camera_head = nn.Linear(4, 4)
        self.global_blocks = nn.Linear(4, 4)
        self.spatial_blocks = nn.Linear(4, 4)


class FreezeRegistryTest(unittest.TestCase):
    def setUp(self):
        self.model = _TinyGemDepth()

    def test_builtin_policies_are_discovered(self):
        names = available_freeze_names()
        for expected in ("default", "head_only", "gem_frozen"):
            self.assertIn(expected, names)
        self.assertIsNotNone(get_freeze_policy("default"))

    def test_context_is_injected_by_parameter_name(self):
        # The policy declares only model/use_gem, so extra context is dropped
        # rather than raising a TypeError.
        summary = apply_freeze(
            "registry_test_policy",
            model=self.model,
            use_gem=True,
            unrelated="ignored",
        )
        self.assertEqual(summary, "saw use_gem=True")

    def test_default_policy_leaves_everything_trainable(self):
        self.assertIsNone(apply_freeze("default", model=self.model, use_gem=False))
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

    def test_explicit_null_freeze_mode_falls_back_to_default(self):
        self.assertIsNone(apply_freeze(None, model=self.model, use_gem=False))
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

    def test_head_only_leaves_just_the_head_trainable(self):
        apply_freeze("head_only", model=self.model, use_gem=False)
        self.assertTrue(all(p.requires_grad for p in self.model.head.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.model.camera_head.parameters()))

    def test_gem_frozen_freezes_gem_but_not_astt(self):
        apply_freeze("gem_frozen", model=self.model, use_gem=True)
        frozen = {
            name for name, param in self.model.named_parameters()
            if not param.requires_grad
        }
        self.assertTrue(any(name.startswith(GEM_PREFIXES) for name in frozen))
        self.assertTrue(all(p.requires_grad for p in self.model.spatial_blocks.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.head.parameters()))

    def test_gem_frozen_requires_gem(self):
        with self.assertRaisesRegex(ValueError, "requires model.use_gem"):
            apply_freeze("gem_frozen", model=self.model, use_gem=False)

    def test_unknown_policy_lists_the_options(self):
        with self.assertRaisesRegex(ValueError, "options="):
            apply_freeze("does_not_exist", model=self.model, use_gem=False)

    def test_module_groups_stay_consistent(self):
        for prefix in GEM_PREFIXES:
            self.assertIn(prefix, NEW_MODULE_PREFIXES)


if __name__ == "__main__":
    unittest.main()
