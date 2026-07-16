import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.util.warp_v2 import (
    scale_intrinsics_v2,
    temporal_depth_error_v2,
    temporal_signal_error_v2,
)


class WarpV2Test(unittest.TestCase):
    @staticmethod
    def geometry(frames=2, height=8, width=10):
        K = torch.tensor([
            [20.0, 0.0, (width - 1) / 2],
            [0.0, 20.0, (height - 1) / 2],
            [0.0, 0.0, 1.0],
        ]).reshape(1, 1, 3, 3).repeat(1, frames, 1, 1)
        ext = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, frames, 1, 1)
        depth = torch.full((1, frames, 1, height, width), 20.0)
        return K, ext, depth

    def test_pixel_center_intrinsic_scaling(self):
        K, _, _ = self.geometry(height=8, width=10)
        scaled = scale_intrinsics_v2(K, (8, 10), (4, 5))
        self.assertAlmostEqual(float(scaled[0, 0, 0, 0]), 10.0)
        self.assertAlmostEqual(float(scaled[0, 0, 1, 1]), 10.0)
        self.assertAlmostEqual(float(scaled[0, 0, 0, 2]), 2.0)
        self.assertAlmostEqual(float(scaled[0, 0, 1, 2]), 1.5)

    def test_identity_rgb_warp_matches_target_and_captures_stage_image(self):
        torch.manual_seed(0)
        base = torch.rand(1, 1, 3, 8, 10)
        signal = base.repeat(1, 2, 1, 1, 1)
        K, ext, depth = self.geometry()
        error, valid, diagnostic = temporal_signal_error_v2(
            signal, depth, K, ext, offsets=(1,), distance='rgb_l1',
            border_margin=1.0, return_diagnostics=True)
        self.assertGreater(float(valid[:, 0].mean()), 0.4)
        self.assertLess(float(error[:, 0].max()), 1e-6)
        self.assertEqual(tuple(diagnostic['target'].shape), tuple(signal.shape))
        selected = valid.bool().expand_as(signal)
        torch.testing.assert_close(
            diagnostic['warped'][selected], signal[selected], atol=1e-6, rtol=0)

    def test_identity_feature_cosine_error_is_zero(self):
        torch.manual_seed(1)
        base = torch.randn(1, 1, 32, 8, 10)
        feature = base.repeat(1, 2, 1, 1, 1)
        K, ext, depth = self.geometry()
        error, valid = temporal_signal_error_v2(
            feature, depth, K, ext, offsets=(1,), distance='feature_cosine')
        self.assertGreater(float(valid[:, 0].mean()), 0.4)
        self.assertLess(float(error[:, 0].max()), 1e-6)

    def test_identity_metric_depth_error_is_zero(self):
        K, ext, depth = self.geometry()
        error, valid = temporal_depth_error_v2(depth, K, ext, offsets=(1,))
        self.assertGreater(float(valid[:, 0].mean()), 0.4)
        self.assertLess(float(error[:, 0].max()), 1e-6)

    def test_border_margin_removes_feature_padding_band(self):
        signal = torch.ones(1, 2, 3, 8, 10)
        K, ext, depth = self.geometry()
        _, valid = temporal_signal_error_v2(
            signal, depth, K, ext, offsets=(1,), border_margin=1.0)
        self.assertEqual(float(valid[0, 0, 0, 0].sum()), 0.0)
        self.assertEqual(float(valid[0, 0, 0, -1].sum()), 0.0)
        self.assertEqual(float(valid[0, 0, 0, :, 0].sum()), 0.0)
        self.assertEqual(float(valid[0, 0, 0, :, -1].sum()), 0.0)
        self.assertTrue(bool(valid[0, 0, 0, 2:-2, 2:-2].all()))


if __name__ == '__main__':
    unittest.main()
