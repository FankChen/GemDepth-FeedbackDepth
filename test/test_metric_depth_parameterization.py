import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.dpt_multiscale_gt_error import DPTHeadMultiScaleGTError
from train import compute_metric_depth_loss


class MetricDepthParameterizationTest(unittest.TestCase):
    def make_head(self, mode):
        return DPTHeadMultiScaleGTError(
            in_channels=64,
            features=32,
            out_channels=(32, 64, 128, 128),
            num_frames=4,
            use_temporal=False,
            error_signal='rgbfeat',
            metric_depth_mode=mode,
            metric_init_depth=20.0,
            metric_min_depth=0.1,
            metric_max_depth=200.0,
        )

    def test_legacy_and_log_modes_keep_identical_state_schema(self):
        legacy = self.make_head('softplus')
        fixed = self.make_head('log_depth')
        legacy_state = legacy.state_dict()
        fixed_state = fixed.state_dict()
        self.assertEqual(list(legacy_state), list(fixed_state))
        for name in legacy_state:
            self.assertEqual(legacy_state[name].shape, fixed_state[name].shape, name)

    def test_log_depth_initializes_to_requested_metric_depth(self):
        head = self.make_head('log_depth')
        feature = torch.randn(8, 32, 12, 12)
        head.metric_depths = []
        head.metric_log_depths = []
        metric = head._predict_metric_depth('p1', feature, b=2, t=4)
        torch.testing.assert_close(metric, torch.full_like(metric, 20.0))
        torch.testing.assert_close(
            head.metric_log_depths[0],
            torch.full_like(head.metric_log_depths[0], torch.tensor(20.0).log().item()))

    def test_collapsed_warp_value_retains_direct_log_loss_gradient(self):
        head = self.make_head('log_depth')
        final_conv = head.metric_depth_heads['p1'][-2]
        with torch.no_grad():
            final_conv.weight.zero_()
            final_conv.bias.fill_(-100.0)

        feature = torch.randn(8, 32, 8, 8)
        head.metric_depths = []
        head.metric_log_depths = []
        metric = head._predict_metric_depth('p1', feature, b=2, t=4)
        self.assertTrue(torch.isfinite(metric).all())
        torch.testing.assert_close(metric, torch.full_like(metric, 0.1))

        gt = torch.full((2, 4, 1, 16, 16), 20.0)
        mask = torch.ones_like(gt)
        loss = compute_metric_depth_loss(
            head.metric_depths, gt, mask,
            metric_log_depths=head.metric_log_depths)
        loss.backward()
        self.assertIsNotNone(final_conv.bias.grad)
        self.assertTrue(torch.isfinite(final_conv.bias.grad).all())
        self.assertGreater(float(final_conv.bias.grad.abs().max()), 0.0)

    def test_warp_depth_is_bounded_but_loss_logit_is_not_clamped(self):
        head = self.make_head('log_depth')
        final_conv = head.metric_depth_heads['p2'][-2]
        feature = torch.randn(4, 32, 6, 6)
        with torch.no_grad():
            final_conv.weight.zero_()
            final_conv.bias.fill_(100.0)
        head.metric_depths = []
        head.metric_log_depths = []
        metric = head._predict_metric_depth('p2', feature, b=1, t=4)
        torch.testing.assert_close(metric, torch.full_like(metric, 200.0))
        self.assertEqual(float(head.metric_log_depths[0].min()), 100.0)


if __name__ == '__main__':
    unittest.main()
