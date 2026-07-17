import sys
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = ROOT / 'evaluation' / 'inference'
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from protocol import (
    infer_video_with_protocol,
    resolve_inference_clip_len,
    validate_inverse_depth_output,
)


class InferenceProtocolTest(unittest.TestCase):
    class FakeModel:
        def __init__(self):
            self.kwargs = None

        def infer_video_depth(self, videos, target_fps, **kwargs):
            self.kwargs = kwargs
            return np.ones((len(videos), 8, 12), dtype=np.float32), target_fps

    def test_scratch_temporal_uses_training_clip_length(self):
        cfg = OmegaConf.create({
            'model': {'encoder_decoder_only': True},
            'dataset': {'train': {'seq_len': 4}},
        })
        self.assertEqual(resolve_inference_clip_len(cfg), 4)

    def test_scratch_static_uses_one_frame_clips(self):
        cfg = OmegaConf.create({
            'model': {'encoder_decoder_only': True},
            'dataset': {'train': {'seq_len': 1}},
        })
        self.assertEqual(resolve_inference_clip_len(cfg), 1)

    def test_original_gemdepth_keeps_overlapping_protocol(self):
        cfg = OmegaConf.create({
            'model': {'encoder_decoder_only': False},
            'dataset': {'train': {'seq_len': 4}},
        })
        self.assertIsNone(resolve_inference_clip_len(cfg))

    def test_invalid_scratch_sequence_length_fails_closed(self):
        cfg = OmegaConf.create({
            'model': {'encoder_decoder_only': True},
            'dataset': {'train': {'seq_len': 0}},
        })
        with self.assertRaisesRegex(ValueError, 'positive'):
            resolve_inference_clip_len(cfg)

    def test_finite_video_passes(self):
        depth = np.ones((4, 12, 16), dtype=np.float32)
        result = validate_inverse_depth_output(depth, 'kitti', 'seq0')
        self.assertIs(result, depth)

    def test_nonfinite_video_reports_sequence_before_save(self):
        depth = np.ones((4, 12, 16), dtype=np.float32)
        depth[2, 3, 4] = np.nan
        with self.assertRaisesRegex(
                FloatingPointError, 'dataset=kitti sequence=seq_bad'):
            validate_inverse_depth_output(depth, 'kitti', 'seq_bad')

    def test_inference_call_receives_resolved_clip_length(self):
        model = self.FakeModel()
        videos = np.zeros((7, 8, 12, 3), dtype=np.uint8)
        depths, fps = infer_video_with_protocol(
            model, videos, target_fps=1, input_size=518, device='cuda',
            fp32=True, clip_len=4, dataset='kitti', sequence='seq0')
        self.assertEqual(model.kwargs['clip_len'], 4)
        self.assertEqual(model.kwargs['input_size'], 518)
        self.assertTrue(model.kwargs['fp32'])
        self.assertEqual(depths.shape[0], len(videos))
        self.assertEqual(fps, 1)


if __name__ == '__main__':
    unittest.main()
