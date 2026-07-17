import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.dpt_gt_error_v2 import DPTHeadGTErrorV2
from model.dpt_temporal import DPTHeadTemporal
from train import compute_aux_depth_loss_disp_clip


class DPTGTErrorV2Test(unittest.TestCase):
    def make_heads(self):
        kwargs = dict(
            in_channels=64,
            features=32,
            out_channels=[32, 64, 128, 128],
            num_frames=2,
            use_temporal=False,
            patch_size=14,
        )
        torch.manual_seed(4)
        baseline = DPTHeadTemporal(**kwargs)
        torch.manual_seed(4)
        v2 = DPTHeadGTErrorV2(
            **kwargs,
            error_signal='rgbfeat',
            metric_init_depth=20.0,
            metric_min_depth=1e-3,
            metric_max_depth=100.0,
        )
        common = {
            name: value for name, value in v2.state_dict().items()
            if name in baseline.state_dict()
        }
        baseline.load_state_dict(common, strict=True)
        return baseline.eval(), v2.eval()

    @staticmethod
    def inputs():
        batch, frames, patch_h, patch_w, channels = 1, 2, 8, 8, 64
        torch.manual_seed(5)
        one_frame_tokens = [
            torch.randn(1, patch_h * patch_w, channels) for _ in range(4)
        ]
        features = []
        for token in one_frame_tokens:
            repeated = token.repeat(batch * frames, 1, 1)
            cls = torch.zeros(batch * frames, channels)
            features.append((repeated, cls))
        image = torch.randn(batch, 1, 3, 112, 112).repeat(1, frames, 1, 1, 1)
        K = torch.tensor([
            [90.0, 0.0, 55.5],
            [0.0, 90.0, 55.5],
            [0.0, 0.0, 1.0],
        ]).unsqueeze(0)
        poses = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, frames, 1, 1)
        return features, image, K, poses, patch_h, patch_w, frames

    def test_zero_initialized_v2_is_exact_temporal_dpt_anchor(self):
        baseline, v2 = self.make_heads()
        features, image, K, poses, patch_h, patch_w, frames = self.inputs()
        with torch.no_grad():
            expected = baseline(features, patch_h, patch_w, frames)
            actual = v2(
                features, patch_h, patch_w, frames,
                images=image, gt_intrinsics=K, gt_extrinsics=poses)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_all_four_levels_have_depth_loss_and_matching_warp_images(self):
        _, v2 = self.make_heads()
        v2.capture_warps = True
        features, image, K, poses, patch_h, patch_w, frames = self.inputs()
        with torch.no_grad():
            output = v2(
                features, patch_h, patch_w, frames,
                images=image, gt_intrinsics=K, gt_extrinsics=poses)
        self.assertEqual(tuple(v2.stage_depths), ('p4', 'p3', 'p2', 'p1'))
        self.assertEqual(len(v2.aux_depths), 4)
        self.assertEqual(len(v2.metric_depths), 4)
        self.assertEqual(tuple(v2.error_maps), ('p4', 'p3', 'p2', 'p1'))
        self.assertEqual(tuple(v2.warp_visuals), ('p4', 'p3', 'p2', 'p1'))
        self.assertEqual(tuple(output.shape), (2, 1, 112, 112))
        expected_stage_sizes = {'p4': 8, 'p3': 16, 'p2': 32, 'p1': 64}
        for stage, native_size in expected_stage_sizes.items():
            self.assertEqual(
                tuple(v2.stage_depths[stage].shape), (1, 2, 1, 112, 112))
            self.assertEqual(
                tuple(v2.warp_visuals[stage]['target'].shape),
                (1, 2, 3, native_size, native_size))
            self.assertEqual(
                tuple(v2.warp_visuals[stage]['warped'].shape),
                (1, 2, 3, native_size, native_size))
            self.assertGreater(float(v2.valid_maps[stage][:, 0].mean()), 0.0)

        gt = torch.rand(1, 2, 1, 112, 112) * 40.0 + 2.0
        mask = torch.ones_like(gt)
        loss = compute_aux_depth_loss_disp_clip(v2.aux_depths, gt, mask)
        self.assertTrue(torch.isfinite(loss))

    def test_zero_error_cannot_bypass_into_feature_only_correction(self):
        _, v2 = self.make_heads()
        zero = torch.zeros(2, 3, 64, 64)
        correction = v2.final_error_correction(zero)
        torch.testing.assert_close(correction, torch.zeros_like(correction), atol=0, rtol=0)
        for encoder in v2.error_encoders.values():
            feedback = encoder(torch.zeros(2, 3, 16, 16))
            torch.testing.assert_close(feedback, torch.zeros_like(feedback), atol=0, rtol=0)

    def test_four_levels_share_the_final_clip_affine_gauge(self):
        torch.manual_seed(7)
        gt = torch.rand(1, 2, 1, 16, 16) * 20.0 + 2.0
        inverse = 1.0 / gt
        common = 2.0 * inverse + 0.3
        matching = [common.clone() for _ in range(4)]
        mask = torch.ones_like(gt)
        matched_loss = compute_aux_depth_loss_disp_clip(matching, gt, mask)
        self.assertLess(float(matched_loss), 1e-6)

        mismatched = [common.clone() for _ in range(4)]
        mismatched[0] = 4.0 * inverse + 1.0
        mismatch_loss = compute_aux_depth_loss_disp_clip(mismatched, gt, mask)
        self.assertGreater(float(mismatch_loss), 1e-3)

    def test_inactive_default_stays_inert_and_loads_like_v2(self):
        _, v2 = self.make_heads()
        # Default build (feedback_gate_init=0) must have no gate params and keep
        # the error pathway exactly zero, so old v2 checkpoints load unchanged.
        self.assertFalse(v2.errmap_active)
        self.assertFalse(hasattr(v2, 'feedback_gates'))
        zero = torch.zeros(2, 3, 16, 16)
        feedback = v2._apply_feedback('p2', zero)
        torch.testing.assert_close(feedback, torch.zeros_like(feedback), atol=0, rtol=0)

    def test_active_feedback_breaks_the_zero_init_deadlock(self):
        kwargs = dict(
            in_channels=64, features=32, out_channels=[32, 64, 128, 128],
            num_frames=2, use_temporal=False, patch_size=14)
        torch.manual_seed(4)
        active = DPTHeadGTErrorV2(
            **kwargs, error_signal='rgbfeat', metric_init_depth=20.0,
            metric_min_depth=1e-3, metric_max_depth=100.0, feedback_gate_init=0.1)
        self.assertTrue(active.errmap_active)
        self.assertEqual(set(active.feedback_gates), {'p4', 'p3', 'p2'})
        # A non-zero error now produces non-zero feedback / correction (the v2
        # deadlock was zero-init last conv -> both gate and encoder gradients 0).
        error = torch.randn(2, 3, 16, 16)
        feedback = active._apply_feedback('p2', error)
        self.assertGreater(float(feedback.abs().max()), 0.0)
        correction = active.correction_gate * active.final_error_correction(error)
        self.assertGreater(float(correction.abs().max()), 0.0)


if __name__ == '__main__':
    unittest.main()
