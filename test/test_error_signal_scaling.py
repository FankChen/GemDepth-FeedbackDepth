import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.util.gt_error import normalize_error_map
from model.util.warp import cosine_error_map


class ErrorSignalScalingTest(unittest.TestCase):
    def test_fixed_mode_does_not_amplify_tiny_feature_noise(self):
        error = torch.full((1, 4, 1, 8, 8), 3.5e-5)
        valid = torch.ones_like(error)
        fixed = normalize_error_map(error, valid, mode='fixed')
        legacy = normalize_error_map(error, valid, mode='mean')
        torch.testing.assert_close(fixed, error)
        torch.testing.assert_close(legacy, torch.full_like(legacy, 0.2))

    def test_fixed_mode_preserves_rgb_and_geometry_unit_range(self):
        error = torch.tensor([[[[[-0.2, 0.25, 1.5]]]]])
        valid = torch.tensor([[[[[1.0, 1.0, 0.0]]]]])
        result = normalize_error_map(error, valid, mode='fixed')
        expected = torch.tensor([[[[[0.0, 0.25, 0.0]]]]])
        torch.testing.assert_close(result, expected)

    @staticmethod
    def geometry(frames, height, width):
        K = torch.tensor([
            [20.0, 0.0, (width - 1) / 2],
            [0.0, 20.0, (height - 1) / 2],
            [0.0, 0.0, 1.0],
        ]).view(1, 1, 3, 3).repeat(1, frames, 1, 1)
        ext = torch.eye(4).view(1, 1, 4, 4).repeat(1, frames, 1, 1)
        depth = torch.full((1, frames, 1, height, width), 20.0)
        return depth, K, ext

    def test_cosine_error_is_zero_for_identical_warped_features(self):
        torch.manual_seed(0)
        base = torch.randn(1, 1, 16, 6, 7)
        features = base.repeat(1, 2, 1, 1, 1)
        depth, K, ext = self.geometry(2, 6, 7)
        error, valid = cosine_error_map(features, depth, K, ext, offsets=(1,))
        self.assertGreater(float(valid[:, 0].mean()), 0.99)
        self.assertLess(float(error[:, 0].max()), 1e-6)

    def test_cosine_error_has_fixed_unit_scale(self):
        torch.manual_seed(1)
        base = torch.randn(1, 1, 16, 6, 7)
        features = torch.cat((base, -base), dim=1)
        depth, K, ext = self.geometry(2, 6, 7)
        error, valid = cosine_error_map(features, depth, K, ext, offsets=(1,))
        selected = error[:, 0][valid[:, 0].bool()]
        torch.testing.assert_close(selected, torch.ones_like(selected), atol=1e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()