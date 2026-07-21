import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / 'evaluation' / 'eval'
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from alignment import require_finite, stable_scale_and_shift


class StableEvaluationAlignmentTest(unittest.TestCase):
    def test_matches_known_affine_relation(self):
        prediction = np.linspace(0.01, 2.0, 10000, dtype=np.float64)
        target = 2.75 * prediction - 0.13
        scale, shift = stable_scale_and_shift(prediction, target)
        self.assertAlmostEqual(scale, 2.75, places=12)
        self.assertAlmostEqual(shift, -0.13, places=12)

    def test_matches_lstsq_on_well_conditioned_input(self):
        rng = np.random.default_rng(7)
        prediction = rng.normal(size=20000)
        target = rng.normal(size=20000)
        expected = np.linalg.lstsq(
            np.stack((prediction, np.ones_like(prediction)), axis=1),
            target, rcond=None)[0]
        actual = stable_scale_and_shift(prediction, target)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_constant_prediction_has_defined_solution(self):
        prediction = np.full(1000, 0.5)
        target = np.linspace(0.01, 0.3, 1000)
        scale, shift = stable_scale_and_shift(prediction, target)
        self.assertEqual(scale, 0.0)
        self.assertAlmostEqual(shift, float(target.mean()), places=12)

    def test_large_finite_prediction_does_not_overflow(self):
        prediction = np.array([1e250, 2e250, 3e250], dtype=np.float64)
        target = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        scale, shift = stable_scale_and_shift(prediction, target)
        self.assertTrue(np.isfinite(scale))
        self.assertTrue(np.isfinite(shift))

    def test_nonfinite_input_fails_with_source(self):
        values = np.array([1.0, np.inf])
        with self.assertRaisesRegex(FloatingPointError, 'bad.npy'):
            require_finite(values, 'Predicted inverse depth', 'bad.npy')


if __name__ == '__main__':
    unittest.main()
