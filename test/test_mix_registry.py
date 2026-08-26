import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.mix_registry import (
    apply_mix_policy,
    available_mix_names,
    get_mix_policy,
    register,
)

# Measured from a stage1_lite launch on the five training sets of
# config/stages/stage1_repro.yaml. TartanAir prints no count, so it is the
# remainder of the 834172-entry list the dataloader reports.
_COUNTS = {
    'TartanAir': 295196,
    'pointodyssey': 301594,
    'mvs_synth': 8280,
    'vkitti': 11342,
    'dynamic_replica': 10760,
}
# What the code did before the policy existed: hardcoded 1 for vkitti and
# tartanair, the loader's class attribute for the rest.
_NATIVE = {
    'TartanAir': 1,
    'pointodyssey': 1,
    'mvs_synth': 26,
    'vkitti': 1,
    'dynamic_replica': 1,
}


def _shares(counts, ratios):
    entries = {label: counts[label] * ratios[label] for label in counts}
    total = sum(entries.values())
    return {label: n / total for label, n in entries.items()}


@register("registry_test_policy")
def _registry_test_policy(counts):
    return {label: 1 for label in counts}


class MixRegistryTest(unittest.TestCase):
    def test_builtin_policies_are_discovered(self):
        for expected in ("native", "uniform"):
            self.assertIn(expected, available_mix_names())
        self.assertIsNotNone(get_mix_policy("native"))

    def test_context_is_injected_by_parameter_name(self):
        # The policy declares only `counts`; native_ratios must be dropped
        # rather than raise.
        ratios = apply_mix_policy(
            "registry_test_policy", counts=_COUNTS, native_ratios=_NATIVE)
        self.assertEqual(set(ratios), set(_COUNTS))

    def test_native_reproduces_the_historical_mix(self):
        ratios = apply_mix_policy("native", counts=_COUNTS, native_ratios=_NATIVE)
        self.assertEqual(ratios, _NATIVE)

        shares = _shares(_COUNTS, ratios)
        # The imbalance this policy preserves, and the reason uniform exists:
        # VKITTI2 has more clips than MVS-Synth yet a far smaller share.
        self.assertLess(shares['vkitti'], 0.02)
        self.assertGreater(shares['mvs_synth'], 0.2)
        self.assertGreater(_COUNTS['vkitti'], _COUNTS['mvs_synth'])

        # Pins the split the recorded runs were trained under, so a change to
        # the default shows up here rather than as an unexplained result shift.
        self.assertAlmostEqual(shares['pointodyssey'], 0.362, delta=0.003)
        self.assertAlmostEqual(shares['TartanAir'], 0.354, delta=0.003)
        self.assertAlmostEqual(shares['mvs_synth'], 0.258, delta=0.003)
        self.assertAlmostEqual(shares['vkitti'], 0.0136, delta=0.001)
        self.assertAlmostEqual(shares['dynamic_replica'], 0.0129, delta=0.001)

    def test_explicit_null_falls_back_to_native(self):
        self.assertEqual(
            apply_mix_policy(None, counts=_COUNTS, native_ratios=_NATIVE), _NATIVE)

    def test_uniform_equalises_every_source(self):
        ratios = apply_mix_policy("uniform", counts=_COUNTS, native_ratios=_NATIVE)
        shares = _shares(_COUNTS, ratios)
        for label, share in shares.items():
            # Integer repetition cannot hit 20% exactly for every source.
            self.assertAlmostEqual(share, 1 / len(_COUNTS), delta=0.03, msg=label)

    def test_uniform_lifts_the_underweighted_driving_source(self):
        before = _shares(_COUNTS, apply_mix_policy(
            "native", counts=_COUNTS, native_ratios=_NATIVE))
        after = _shares(_COUNTS, apply_mix_policy(
            "uniform", counts=_COUNTS, native_ratios=_NATIVE))
        self.assertGreater(after['vkitti'] / before['vkitti'], 10)

    def test_ratios_are_at_least_one(self):
        # A source larger than the target must still appear, not vanish.
        counts = dict(_COUNTS, huge=10 ** 7)
        ratios = apply_mix_policy("uniform", counts=counts, native_ratios=_NATIVE)
        self.assertTrue(all(r >= 1 for r in ratios.values()))

    def test_unknown_policy_lists_the_options(self):
        with self.assertRaisesRegex(ValueError, "options="):
            apply_mix_policy("does_not_exist", counts=_COUNTS, native_ratios=_NATIVE)


if __name__ == "__main__":
    unittest.main()
