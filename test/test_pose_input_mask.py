import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.gemdepth import GemDepth


class _ModeOnly:
    """Stand-in for a built GemDepth: the mask only reads ``self.training``."""

    def __init__(self, training):
        self.training = training


def _mask(training, B=16, T=4):
    return GemDepth._pose_input_mask(_ModeOnly(training), B, T, torch.device("cpu"))


class PoseInputMaskTest(unittest.TestCase):
    B, T = 16, 4

    def test_eval_injects_pose_into_every_frame(self):
        mask = _mask(training=False)
        self.assertEqual(mask.shape, (self.B * self.T,))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(bool(mask.all()))

    def test_eval_is_deterministic_and_does_not_consume_rng(self):
        torch.manual_seed(0)
        before = torch.rand(1)

        torch.manual_seed(0)
        first = _mask(training=False)
        second = _mask(training=False)
        after = torch.rand(1)

        self.assertTrue(bool((first == second).all()))
        # Inference must not perturb the global RNG stream either, otherwise
        # anything else drawing from it becomes order-dependent.
        self.assertTrue(torch.equal(before, after))

    def test_training_still_drops_pose(self):
        torch.manual_seed(0)
        keep_rate = torch.stack(
            [_mask(training=True).float().mean() for _ in range(200)]
        ).mean()
        # 0.9 (clip geometry) * 0.95 (frame geometry) * 0.5 (clip pose) = 0.4275
        self.assertAlmostEqual(float(keep_rate), 0.4275, delta=0.02)

    def test_training_masks_vary_between_calls(self):
        torch.manual_seed(0)
        self.assertFalse(bool((_mask(training=True) == _mask(training=True)).all()))


if __name__ == "__main__":
    unittest.main()
