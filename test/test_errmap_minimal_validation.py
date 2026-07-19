import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    'validate_errmap_minimal',
    ROOT / 'evaluation/visualization/validate_errmap_minimal.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MinimalErrmapValidationTest(unittest.TestCase):
    def test_d_to_z_calibration_recovers_metric_depth(self):
        torch.manual_seed(0)
        gt = torch.rand(1, 2, 1, 8, 8) * 30.0 + 2.0
        raw = 2.5 / gt + 0.17
        mask = torch.ones_like(gt)
        result = MODULE.calibrate_stage_depth(raw, gt, mask, (8, 8))
        torch.testing.assert_close(result['metric'], gt, atol=1e-3, rtol=1e-4)
        self.assertAlmostEqual(float(result['scale'][0]), 0.4, places=4)
        self.assertAlmostEqual(float(result['shift'][0]), -0.068, places=4)

    def test_local_depth_perturbation_changes_error_in_roi(self):
        torch.manual_seed(1)
        height, width, frames = 12, 14, 2
        rgb_one = torch.rand(1, 1, 3, height, width)
        rgb = rgb_one.repeat(1, frames, 1, 1, 1)
        feature_one = torch.randn(1, 1, 16, height, width)
        feature = feature_one.repeat(1, frames, 1, 1, 1)
        metric = torch.full((1, frames, 1, height, width), 20.0)
        K = torch.tensor([
            [24.0, 0.0, (width - 1) / 2],
            [0.0, 24.0, (height - 1) / 2],
            [0.0, 0.0, 1.0],
        ]).reshape(1, 1, 3, 3).repeat(1, frames, 1, 1)
        poses = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, frames, 1, 1)
        poses[:, 1, 0, 3] = 0.5
        predicted = MODULE.compute_errors(
            rgb, feature, metric, K, poses, offsets=(1,),
            border_margin=1.0, occlusion_rel=0.05, occlusion_abs=0.1,
            diagnostics=False)
        perturbed_metric, region = MODULE.make_perturbation(metric, frame_index=0)
        perturbed = MODULE.compute_errors(
            rgb, feature, perturbed_metric, K, poses, offsets=(1,),
            border_margin=1.0, occlusion_rel=0.05, occlusion_abs=0.1,
            diagnostics=False)
        roi = region[:, 0].float()
        difference = (
            (perturbed['rgb_error'][:, 0] - predicted['rgb_error'][:, 0]).abs()
            + (perturbed['feature_error'][:, 0] - predicted['feature_error'][:, 0]).abs())
        self.assertGreater(float((difference * roi).sum()), 0.0)

    def test_error_gated_corrector_has_no_zero_error_bypass(self):
        torch.manual_seed(2)
        corrector = MODULE.ErrorGatedCorrector(8)
        feature = torch.randn(2, 8, 10, 10)
        zero_error = torch.zeros(2, 3, 10, 10)
        output = corrector(feature, zero_error)
        torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()