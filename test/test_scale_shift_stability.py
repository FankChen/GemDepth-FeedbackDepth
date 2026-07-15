import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loss.videoloss import VideoDepthLoss, compute_scale_and_shift


class ScaleShiftStabilityTest(unittest.TestCase):
    def test_matches_centered_reference_when_well_conditioned(self):
        torch.manual_seed(0)
        prediction = torch.randn(8, 65, 65)
        target = torch.randn_like(prediction) * 0.2
        mask = (torch.rand_like(prediction) > 0.1).float()

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        p = prediction.double()
        t = target.double()
        w = mask.double()
        count = w.sum(dim=(1, 2))
        mean_p = (w * p).sum(dim=(1, 2)) / count
        mean_t = (w * t).sum(dim=(1, 2)) / count
        centered_p = p - mean_p[:, None, None]
        centered_t = t - mean_t[:, None, None]
        reference_scale = (w * centered_p * centered_t).sum(dim=(1, 2)) / (
            w * centered_p.square()).sum(dim=(1, 2))
        reference_shift = mean_t - reference_scale * mean_p

        torch.testing.assert_close(scale.double(), reference_scale, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(shift.double(), reference_shift, atol=1e-7, rtol=1e-6)

    def test_constant_and_near_constant_predictions_stay_finite(self):
        torch.manual_seed(1)
        for dtype in (torch.float32, torch.bfloat16):
            for base in (0.01, 1.0, 100.0, 1e4):
                for noise in (0.0, 1e-8, 1e-6, 1e-4):
                    prediction = torch.full((4, 65, 65), base, dtype=dtype)
                    prediction = (
                        prediction + torch.randn_like(prediction) * noise
                    ).requires_grad_()
                    target = torch.rand((4, 65, 65), dtype=torch.float32) * 0.2
                    mask = torch.ones_like(target)

                    scale, shift = compute_scale_and_shift(prediction, target, mask)
                    aligned = scale[:, None, None] * prediction + shift[:, None, None]
                    loss = aligned.float().abs().mean()
                    loss.backward()

                    self.assertTrue(torch.isfinite(scale).all())
                    self.assertTrue(torch.isfinite(shift).all())
                    self.assertLessEqual(float(scale.abs().max()), 100.0)
                    self.assertIsNotNone(prediction.grad)
                    self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_empty_mask_uses_identity_alignment(self):
        prediction = torch.randn(2, 8, 8)
        target = torch.randn_like(prediction)
        mask = torch.zeros_like(prediction)
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        torch.testing.assert_close(scale, torch.ones_like(scale))
        torch.testing.assert_close(shift, torch.zeros_like(shift))

    def test_video_depth_loss_constant_prediction_backward(self):
        prediction = torch.full((2, 4, 32, 32), 0.01, requires_grad=True)
        target = torch.rand(2, 4, 32, 32) * 79.0 + 1.0
        mask = torch.ones_like(target)
        loss_dict = VideoDepthLoss(pose_flag=False)(
            prediction, target, mask, None, None, None, None)
        loss = loss_dict['total_loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == '__main__':
    unittest.main()
