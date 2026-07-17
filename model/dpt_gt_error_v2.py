"""Baseline-anchored multi-level DPT with pixel-consistent temporal warp.

The historical multiscale head replaced GemDepth's validated output readout with
an additive raw-depth recursion. This v2 keeps the original temporal-DPT pyramid
and *shared* ``output_conv1/output_conv2`` readout as the anchor at every level.
Error feedback is zero-initialized and injected before the next DPT fusion block,
so initialization is exactly the original GemDepth temporal DPT.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_temporal import DPTHeadTemporal
from .util.gt_error import imagenet_denormalize
from .util.warp_v2 import (
    scale_intrinsics_v2,
    temporal_depth_error_v2,
    temporal_signal_error_v2,
)


_STAGE_NAMES = ('p4', 'p3', 'p2', 'p1')
_ERROR_SIGNALS = ('rgb', 'feat', 'rgbfeat', 'geom')


class DPTHeadGTErrorV2(DPTHeadTemporal):
    """Original GemDepth DPT anchor + four-level supervised error feedback."""

    def __init__(self, in_channels, features=256, use_bn=False,
                 out_channels=(256, 512, 1024, 1024), use_clstoken=False,
                 num_frames=32, pe='ape', use_temporal=True, patch_size=14,
                 error_signal='rgbfeat', warp_offsets=(-1, 1),
                 metric_init_depth=20.0, metric_min_depth=1e-3,
                 metric_max_depth=100.0, warp_border_margin=1.0,
                 warp_occlusion_rel=0.05, warp_occlusion_abs=0.10,
                 feedback_gate_init=0.0):
        super().__init__(
            in_channels=in_channels,
            features=features,
            use_bn=use_bn,
            out_channels=list(out_channels),
            use_clstoken=use_clstoken,
            num_frames=num_frames,
            pe=pe,
            use_temporal=use_temporal,
            patch_size=patch_size,
        )
        if error_signal not in _ERROR_SIGNALS:
            raise ValueError(f'error_signal must be one of {_ERROR_SIGNALS}, got {error_signal!r}')
        if not warp_offsets or any(int(offset) == 0 for offset in warp_offsets):
            raise ValueError(f'warp_offsets must be non-empty and non-zero, got {warp_offsets}')
        if not 0 < metric_min_depth < metric_init_depth < metric_max_depth:
            raise ValueError(
                'Expected 0 < metric_min_depth < metric_init_depth < metric_max_depth')

        self.stage_names = _STAGE_NAMES
        self.error_signal = error_signal
        self.warp_offsets = tuple(int(offset) for offset in warp_offsets)
        self.metric_min_depth = float(metric_min_depth)
        self.metric_max_depth = float(metric_max_depth)
        self.metric_log_min = math.log(self.metric_min_depth)
        self.metric_log_max = math.log(self.metric_max_depth)
        self.warp_border_margin = float(warp_border_margin)
        self.warp_occlusion_rel = float(warp_occlusion_rel)
        self.warp_occlusion_abs = float(warp_occlusion_abs)

        # Every DPT level predicts metric log-depth for its same-resolution warp.
        self.metric_depth_heads = nn.ModuleDict({
            stage: self._make_metric_log_head(features, metric_init_depth)
            for stage in self.stage_names
        })

        # p4/p3/p2 feedback is inserted BEFORE the next refinenet fusion. Biases
        # are disabled, so an all-zero [errors, validity] tensor gives exact zero.
        self.error_encoders = nn.ModuleDict({
            stage: self._make_error_encoder(features)
            for stage in self.stage_names[:-1]
        })
        self.final_error_correction = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.final_error_correction[-1].weight)

        # Error-feedback activation. feedback_gate_init == 0 (default) keeps the
        # feedback/correction EXACTLY zero (inert == anchored Temporal DPT): the
        # last conv is zero-init and no gate params exist, so pre-existing v2
        # checkpoints load unchanged. feedback_gate_init > 0 gives the last conv a
        # small non-zero init AND a learnable per-stage gate, which breaks the
        # zero-init gradient deadlock (W2=0 killed dL/dW1 while a near-zero error
        # killed dL/dW2) so the error pathway can actually be trained.
        self.errmap_active = float(feedback_gate_init) != 0.0
        if self.errmap_active:
            for encoder in self.error_encoders.values():
                nn.init.normal_(encoder[-1].weight, std=1e-2)
            nn.init.normal_(self.final_error_correction[-1].weight, std=1e-2)
            self.feedback_gates = nn.ParameterDict({
                stage: nn.Parameter(torch.tensor(float(feedback_gate_init)))
                for stage in self.stage_names[:-1]
            })
            self.correction_gate = nn.Parameter(torch.tensor(float(feedback_gate_init)))

        self.capture_warps = False
        self.aux_depths = []
        self.stage_depths = {}
        self.metric_depths = []
        self.metric_log_depths = []
        self.error_maps = {}
        self.raw_error_maps = {}
        self.valid_maps = {}
        self.warp_visuals = {}
        self.feedback_maps = {}
        self.stage_features = {}
        self.final_correction = None

    @staticmethod
    def _make_metric_log_head(features, init_depth):
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.constant_(head[-1].bias, math.log(float(init_depth)))
        return head

    @staticmethod
    def _make_error_encoder(features):
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, features, kernel_size=3, padding=1, bias=False),
        )
        nn.init.zeros_(encoder[-1].weight)
        return encoder

    def _prepare_dpt_inputs(self, out_features, patch_h, patch_w, frames,
                            layer_3_att=None, layer_4_att=None):
        projected = []
        for index, value in enumerate(out_features):
            if self.use_clstoken:
                tokens, cls_token = value
                readout = cls_token.unsqueeze(1).expand_as(tokens)
                tokens = self.readout_projects[index](torch.cat((tokens, readout), -1))
            else:
                tokens = value[0]
            tokens = tokens.permute(0, 2, 1).reshape(
                tokens.shape[0], tokens.shape[-1], patch_h, patch_w).contiguous()
            projected.append(self.resize_layers[index](self.projects[index](tokens)))

        layer_1, layer_2, layer_3, layer_4 = projected
        batch = layer_1.shape[0] // frames
        if self.use_temporal:
            layer_3 = self.motion_modules[0](
                layer_3.unflatten(0, (batch, frames)).permute(0, 2, 1, 3, 4),
                None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
            layer_4 = self.motion_modules[1](
                layer_4.unflatten(0, (batch, frames)).permute(0, 2, 1, 3, 4),
                None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        if layer_3_att is not None:
            layer_3 = layer_3 + layer_3_att
        if layer_4_att is not None:
            layer_4 = layer_4 + layer_4_att
        return batch, (
            self.scratch.layer1_rn(layer_1),
            self.scratch.layer2_rn(layer_2),
            self.scratch.layer3_rn(layer_3),
            self.scratch.layer4_rn(layer_4),
        )

    def _baseline_readout(self, feature, output_size):
        """Exact original GemDepth temporal-DPT output readout."""
        value = self.scratch.output_conv1(feature)
        value = F.interpolate(
            value, size=output_size, mode='bilinear', align_corners=True)
        original_dtype = value.dtype
        with torch.autocast(device_type=value.device.type, enabled=False):
            value = self.scratch.output_conv2(value.float())
        return value.to(original_dtype)

    def _metric_depth(self, stage, feature, batch, frames):
        log_depth = self.metric_depth_heads[stage](feature.detach().float())
        expected = (batch * frames, 1, *feature.shape[-2:])
        if log_depth.shape != expected:
            raise ValueError(
                f'Unexpected {stage} metric log-depth {tuple(log_depth.shape)}, expected {expected}')
        metric = torch.exp(log_depth.clamp(self.metric_log_min, self.metric_log_max))
        self.metric_log_depths.append(log_depth.reshape(
            batch, frames, 1, *feature.shape[-2:]))
        self.metric_depths.append(metric.reshape(
            batch, frames, 1, *feature.shape[-2:]))
        return metric

    @staticmethod
    def _camera_tensors(intrinsics, extrinsics, batch, frames, device):
        if intrinsics is None or extrinsics is None:
            raise ValueError('DPTHeadGTErrorV2 requires GT camera intrinsics/extrinsics')
        K = intrinsics.to(device=device, dtype=torch.float32)
        poses = extrinsics.to(device=device, dtype=torch.float32)
        if K.ndim == 3:
            K = K.unsqueeze(1).expand(-1, frames, -1, -1)
        if K.shape != (batch, frames, 3, 3):
            raise ValueError(f'Expected K {(batch, frames, 3, 3)}, got {tuple(K.shape)}')
        if poses.shape != (batch, frames, 4, 4):
            raise ValueError(
                f'Expected extrinsics {(batch, frames, 4, 4)}, got {tuple(poses.shape)}')
        return K, poses

    def _stage_error(self, stage, feature, images, metric, K, poses,
                     batch, frames):
        height, width = feature.shape[-2:]
        metric_bt = metric.detach().reshape(batch, frames, 1, height, width)
        K_stage = scale_intrinsics_v2(K, images.shape[-2:], (height, width))
        rgb = imagenet_denormalize(images.detach().float())
        rgb = F.interpolate(
            rgb.flatten(0, 1), size=(height, width),
            mode='bilinear', align_corners=False)
        rgb = rgb.reshape(batch, frames, 3, height, width)
        diagnostics = None

        with torch.no_grad():
            if self.error_signal in ('rgb', 'rgbfeat') or self.capture_warps:
                rgb_result = temporal_signal_error_v2(
                    rgb, metric_bt, K_stage, poses,
                    offsets=self.warp_offsets,
                    distance='rgb_l1',
                    border_margin=self.warp_border_margin,
                    occlusion_rel=self.warp_occlusion_rel,
                    occlusion_abs=self.warp_occlusion_abs,
                    return_diagnostics=self.capture_warps,
                )
                if self.capture_warps:
                    rgb_error, rgb_valid, diagnostics = rgb_result
                else:
                    rgb_error, rgb_valid = rgb_result

            if self.error_signal in ('feat', 'rgbfeat'):
                feature_signal = feature.detach().reshape(
                    batch, frames, feature.shape[1], height, width).float()
                feat_error, feat_valid = temporal_signal_error_v2(
                    feature_signal, metric_bt, K_stage, poses,
                    offsets=self.warp_offsets,
                    distance='feature_cosine',
                    border_margin=self.warp_border_margin,
                    occlusion_rel=self.warp_occlusion_rel,
                    occlusion_abs=self.warp_occlusion_abs,
                )

            if self.error_signal == 'geom':
                geom_error, geom_valid = temporal_depth_error_v2(
                    metric_bt, K_stage, poses,
                    offsets=self.warp_offsets,
                    border_margin=self.warp_border_margin,
                    occlusion_rel=self.warp_occlusion_rel,
                    occlusion_abs=self.warp_occlusion_abs,
                )

            if self.error_signal == 'rgb':
                channels = torch.cat((rgb_error, torch.zeros_like(rgb_error), rgb_valid), dim=2)
                valid = rgb_valid
                raw = {'rgb': rgb_error, 'rgb_valid': rgb_valid}
            elif self.error_signal == 'feat':
                channels = torch.cat((torch.zeros_like(feat_error), feat_error, feat_valid), dim=2)
                valid = feat_valid
                raw = {'feat': feat_error, 'feat_valid': feat_valid}
            elif self.error_signal == 'rgbfeat':
                valid = torch.minimum(rgb_valid, feat_valid)
                channels = torch.cat((rgb_error * valid, feat_error * valid, valid), dim=2)
                raw = {
                    'rgb': rgb_error * valid, 'rgb_valid': rgb_valid,
                    'feat': feat_error * valid, 'feat_valid': feat_valid,
                }
            else:
                channels = torch.cat((geom_error, torch.zeros_like(geom_error), geom_valid), dim=2)
                valid = geom_valid
                raw = {'geom': geom_error, 'geom_valid': geom_valid}

        self.error_maps[stage] = channels.detach()
        self.raw_error_maps[stage] = {name: value.detach() for name, value in raw.items()}
        self.valid_maps[stage] = valid.detach()
        if diagnostics is not None:
            self.warp_visuals[stage] = {
                name: value.detach() for name, value in diagnostics.items()
            }
        return channels.reshape(batch * frames, 3, height, width).to(feature.dtype)

    def _apply_feedback(self, stage, error):
        feedback = self.error_encoders[stage](error)
        if self.errmap_active:
            feedback = self.feedback_gates[stage] * feedback
        return feedback

    def _record_stage(self, stage, feature, images, K, poses, batch, frames,
                      output_size):
        if self.capture_warps:
            # Diagnostic-only reference (no clone): the minimal validator needs
            # the exact native feature that corresponds to this stage's d_s.
            self.stage_features[stage] = feature.detach()
        inverse_depth = self._baseline_readout(feature, output_size)
        self.stage_depths[stage] = inverse_depth.reshape(
            batch, frames, 1, *output_size)
        metric = self._metric_depth(stage, feature, batch, frames)
        error = self._stage_error(
            stage, feature, images, metric, K, poses, batch, frames)
        return inverse_depth, error

    def forward(self, out_features, patch_h, patch_w, frame_length,
                images=None, gt_intrinsics=None, gt_extrinsics=None,
                layer_3_att=None, layer_4_att=None, mode=None):
        del mode
        batch, layers = self._prepare_dpt_inputs(
            out_features, patch_h, patch_w, frame_length,
            layer_3_att=layer_3_att, layer_4_att=layer_4_att)
        if images is None or images.shape[:2] != (batch, frame_length):
            raise ValueError(
                f'Expected images (B,T,3,H,W) with {(batch, frame_length)}, '
                f'got {None if images is None else tuple(images.shape)}')
        K, poses = self._camera_tensors(
            gt_intrinsics, gt_extrinsics, batch, frame_length, layers[0].device)
        output_size = (int(patch_h * self.patch_size), int(patch_w * self.patch_size))
        layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn = layers

        self.aux_depths = []
        self.stage_depths = {}
        self.metric_depths = []
        self.metric_log_depths = []
        self.error_maps = {}
        self.raw_error_maps = {}
        self.valid_maps = {}
        self.warp_visuals = {}
        self.feedback_maps = {}
        self.stage_features = {}
        self.final_correction = None

        path_4 = self.scratch.refinenet4(
            layer_4_rn, False, size=layer_3_rn.shape[-2:])
        if self.use_temporal:
            path_4 = self.motion_modules[2](
                path_4.unflatten(0, (batch, frame_length)).permute(0, 2, 1, 3, 4),
                None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        depth_4, error_4 = self._record_stage(
            'p4', path_4, images, K, poses, batch, frame_length, output_size)
        feedback_4 = self._apply_feedback('p4', error_4)
        self.feedback_maps['p4'] = feedback_4.detach()

        path_3 = self.scratch.refinenet3(
            path_4 + feedback_4, layer_3_rn, False, size=layer_2_rn.shape[-2:])
        if self.use_temporal:
            path_3 = self.motion_modules[3](
                path_3.unflatten(0, (batch, frame_length)).permute(0, 2, 1, 3, 4),
                None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        depth_3, error_3 = self._record_stage(
            'p3', path_3, images, K, poses, batch, frame_length, output_size)
        feedback_3 = self._apply_feedback('p3', error_3)
        self.feedback_maps['p3'] = feedback_3.detach()

        path_2 = self.scratch.refinenet2(
            path_3 + feedback_3, layer_2_rn, False, size=layer_1_rn.shape[-2:])
        depth_2, error_2 = self._record_stage(
            'p2', path_2, images, K, poses, batch, frame_length, output_size)
        feedback_2 = self._apply_feedback('p2', error_2)
        self.feedback_maps['p2'] = feedback_2.detach()

        path_1 = self.scratch.refinenet1(
            path_2 + feedback_2, layer_1_rn, False)
        depth_1, error_1 = self._record_stage(
            'p1', path_1, images, K, poses, batch, frame_length, output_size)
        correction = self.final_error_correction(error_1)
        if self.errmap_active:
            correction = self.correction_gate * correction
        correction = F.interpolate(
            correction, size=output_size, mode='bilinear', align_corners=False)
        self.final_correction = correction.detach()
        # Match the original GemDepth output semantics before both auxiliary and
        # main supervision (GemDepth.forward applies the same ReLU once more).
        final_depth = F.relu(depth_1 + correction)

        # All four DPT levels are explicitly supervised. p1 stores the actual
        # corrected final prediction; p4/p3/p2 use the shared GemDepth readout.
        self.aux_depths = [
            self.stage_depths['p4'],
            self.stage_depths['p3'],
            self.stage_depths['p2'],
            final_depth.reshape(batch, frame_length, 1, *output_size),
        ]
        self.stage_depths['p1'] = self.aux_depths[-1]
        return final_depth
