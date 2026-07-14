"""Multiscale DPT refinement with GT-camera oracle error feedback.

This head is an additive research variant of ``DPTHeadMultiScaleRefine``.  The
original head is not modified.  It keeps the exact multiscale inverse-depth path
and adds two high-resolution (p2/p1) components:

* a positive metric-depth branch, supervised with GT metric depth, used only for
  camera warping;
* a two-channel error signal derived using GT intrinsics/extrinsics, followed by
  zero-initialized p2->p1 feedback and a zero-initialized p1 correction.

The zero initialization makes the initial main output exactly equal to the
multiscale baseline.  GT depth is never an input to this module; it is used only
by the separate metric-depth training loss.  This stage deliberately uses GT
cameras to isolate the depth/error design before a learned pose CNN is added.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_multiscale import DPTHeadMultiScaleRefine
from .util.gt_error import (
    geometric_depth_error_map,
    imagenet_denormalize,
    normalize_error_map,
    photometric_signal_error,
)
from .util.warp import scale_intrinsics


_ERROR_SIGNALS = ('rgb', 'feat', 'rgbfeat', 'geom')


class DPTHeadMultiScaleGTError(DPTHeadMultiScaleRefine):
    """GT-camera multiscale error feedback with a unified three-channel interface.

    Channels are ``[residual_slot_1, residual_slot_2, validity]``. Single-stream
    arms zero-pad slot 2; RGB+feature uses both residual slots. Thus every arm
    receives an explicit validity mask and has exactly equal capacity.
    """

    def __init__(self, in_channels, features=256, use_bn=False,
                 out_channels=(256, 512, 1024, 1024), use_clstoken=False,
                 num_frames=32, pe='ape', use_temporal=True, patch_size=14,
                 error_signal='rgb', warp_offsets=(-1, 1),
                 metric_init_depth=20.0):
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
            raise ValueError(f"error_signal must be one of {_ERROR_SIGNALS}, got {error_signal!r}")
        self.error_signal = error_signal
        self.warp_offsets = tuple(int(o) for o in warp_offsets)
        if not self.warp_offsets or any(o == 0 for o in self.warp_offsets):
            raise ValueError(f"warp_offsets must contain non-zero temporal offsets, got {warp_offsets}")

        self.metric_depth_heads = nn.ModuleDict({
            stage: self._make_metric_depth_head(features, metric_init_depth)
            for stage in ('p2', 'p1')
        })
        # p2 error is encoded and upsampled into the already-built p1 feature.
        self.error_encoders = nn.ModuleDict({
            'p2': nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(64, features, kernel_size=3, padding=1),
            )
        })
        nn.init.zeros_(self.error_encoders['p2'][-1].weight)
        nn.init.zeros_(self.error_encoders['p2'][-1].bias)

        # All four channel arms share this exact capacity.  Only the two error
        # channels differ.  The final layer is zero-init for baseline equivalence.
        self.p1_correction = nn.Sequential(
            nn.Conv2d(features + 3, features // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        nn.init.zeros_(self.p1_correction[-1].weight)
        nn.init.zeros_(self.p1_correction[-1].bias)

        # Populated on every forward for losses and diagnostics.
        self.metric_depths = []
        self.error_maps = {}
        self.valid_maps = {}

    @staticmethod
    def _make_metric_depth_head(features, init_depth):
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus(),
        )
        # Start from a stable positive constant. The branch remains fully
        # trainable; sigmoid(20) ~= 1 so the bias does not saturate gradients.
        nn.init.zeros_(head[-2].weight)
        nn.init.constant_(head[-2].bias, float(init_depth))
        return head

    @staticmethod
    def _prepare_geometry(gt_intrinsics, gt_extrinsics, b, t, device):
        if gt_intrinsics is None or gt_extrinsics is None:
            raise ValueError(
                "DPTHeadMultiScaleGTError requires GT intrinsics/extrinsics. "
                "This is an oracle-camera experiment; pass them explicitly.")
        K = gt_intrinsics.to(device=device, dtype=torch.float32)
        ext = gt_extrinsics.to(device=device, dtype=torch.float32)
        if K.ndim == 3:
            K = K.unsqueeze(1).expand(-1, t, -1, -1)
        if K.shape != (b, t, 3, 3):
            raise ValueError(f"Expected intrinsics {(b, t, 3, 3)}, got {tuple(K.shape)}")
        if ext.shape != (b, t, 4, 4):
            raise ValueError(f"Expected extrinsics {(b, t, 4, 4)}, got {tuple(ext.shape)}")
        return K, ext

    def _make_error_channels(self, feat, images, metric_depth, K, extrinsics,
                             b, t, src_hw, stage):
        bt, channels, h, w = feat.shape
        if bt != b * t:
            raise ValueError(f"Expected BT={b*t}, got {bt}")
        metric_bt = metric_depth.detach().reshape(b, t, 1, h, w)
        K_s = scale_intrinsics(K, src_hw, (h, w))

        with torch.no_grad():
            if self.error_signal in ('rgb', 'rgbfeat'):
                rgb = imagenet_denormalize(images.detach().float())
                rgb = F.interpolate(
                    rgb.flatten(0, 1), size=(h, w), mode='bilinear', align_corners=False)
                rgb = rgb.reshape(b, t, 3, h, w)
                rgb_error, rgb_valid = photometric_signal_error(
                    rgb, metric_bt, K_s, extrinsics, self.warp_offsets)
                rgb_error = normalize_error_map(rgb_error, rgb_valid)

            if self.error_signal in ('feat', 'rgbfeat'):
                feat_signal = feat.detach().reshape(b, t, channels, h, w)
                feat_signal = F.normalize(feat_signal, p=2, dim=2, eps=1e-6)
                feat_error, feat_valid = photometric_signal_error(
                    feat_signal, metric_bt, K_s, extrinsics, self.warp_offsets)
                feat_error = normalize_error_map(feat_error, feat_valid)

            if self.error_signal == 'geom':
                geom_error, geom_valid = geometric_depth_error_map(
                    metric_bt, K_s, extrinsics, self.warp_offsets)
                geom_error = normalize_error_map(geom_error, geom_valid)

            if self.error_signal == 'rgb':
                error_channels = torch.cat((rgb_error, torch.zeros_like(rgb_error), rgb_valid), dim=2)
                valid = rgb_valid
            elif self.error_signal == 'feat':
                error_channels = torch.cat((torch.zeros_like(feat_error), feat_error, feat_valid), dim=2)
                valid = feat_valid
            elif self.error_signal == 'rgbfeat':
                valid = torch.minimum(rgb_valid, feat_valid)
                error_channels = torch.cat((rgb_error, feat_error, valid), dim=2)
            else:  # geom
                error_channels = torch.cat((geom_error, torch.zeros_like(geom_error), geom_valid), dim=2)
                valid = geom_valid

        self.error_maps[stage] = error_channels.detach()
        self.valid_maps[stage] = valid.detach()
        return error_channels.reshape(bt, 3, h, w).to(feat.dtype)

    def forward(self, out_features, patch_h, patch_w, frame_length,
                init_depth=None, layer_3_att=None, layer_4_att=None, mode=None,
                images=None, gt_intrinsics=None, gt_extrinsics=None):
        paths = self._build_pyramid(
            out_features, patch_h, patch_w, frame_length,
            layer_3_att=layer_3_att, layer_4_att=layer_4_att, mode=False)
        b, t = paths[0].shape[0] // frame_length, frame_length
        if images is None or images.shape[:2] != (b, t):
            raise ValueError(
                f"Expected images (B,T,3,H,W) with B,T={(b,t)}, "
                f"got {None if images is None else tuple(images.shape)}")
        K, ext = self._prepare_geometry(
            gt_intrinsics, gt_extrinsics, b, t, paths[0].device)
        src_hw = images.shape[-2:]

        coarse_size = paths[0].shape[-2:]
        if init_depth is None:
            depth_prev = paths[0].new_zeros((paths[0].shape[0], 1, *coarse_size))
        else:
            depth_prev = F.interpolate(
                init_depth.detach().to(paths[0].dtype), size=coarse_size,
                mode='bilinear', align_corners=True)

        self.aux_depths = []
        self.metric_depths = []
        self.error_maps = {}
        self.valid_maps = {}

        # p4 and p3 are identical to the baseline; they establish global shape.
        for i in (0, 1):
            feat = paths[i]
            if depth_prev.shape[-2:] != feat.shape[-2:]:
                depth_prev = F.interpolate(
                    depth_prev, size=feat.shape[-2:], mode='bilinear', align_corners=True)
            depth_cur = depth_prev.detach() + self.delta_heads[i](feat)
            h, w = depth_cur.shape[-2:]
            self.aux_depths.append(depth_cur.reshape(b, t, 1, h, w))
            depth_prev = depth_cur

        # p2: baseline inverse-depth update + GT-camera error feedback into p1.
        p2 = paths[2]
        if depth_prev.shape[-2:] != p2.shape[-2:]:
            depth_prev = F.interpolate(
                depth_prev, size=p2.shape[-2:], mode='bilinear', align_corners=True)
        depth_p2 = depth_prev.detach() + self.delta_heads[2](p2)
        h2, w2 = depth_p2.shape[-2:]
        self.aux_depths.append(depth_p2.reshape(b, t, 1, h2, w2))
        metric_p2 = self.metric_depth_heads['p2'](p2.detach().float())
        self.metric_depths.append(metric_p2.reshape(b, t, 1, h2, w2))
        error_p2 = self._make_error_channels(
            p2, images, metric_p2, K, ext, b, t, src_hw, 'p2')
        p2_feedback = self.error_encoders['p2'](error_p2)

        # p1 was built by the original pyramid. Add upsampled p2 feedback to its
        # feature (zero at initialization), then perform the baseline p1 update.
        p1 = paths[3] + F.interpolate(
            p2_feedback, size=paths[3].shape[-2:], mode='bilinear', align_corners=True)
        depth_prev = F.interpolate(
            depth_p2, size=p1.shape[-2:], mode='bilinear', align_corners=True)
        depth_p1 = depth_prev.detach() + self.delta_heads[3](p1)
        h1, w1 = depth_p1.shape[-2:]
        metric_p1 = self.metric_depth_heads['p1'](p1.detach().float())
        self.metric_depths.append(metric_p1.reshape(b, t, 1, h1, w1))
        error_p1 = self._make_error_channels(
            p1, images, metric_p1, K, ext, b, t, src_hw, 'p1')
        depth_p1 = depth_p1 + self.p1_correction(torch.cat((p1, error_p1), dim=1))
        self.aux_depths.append(depth_p1.reshape(b, t, 1, h1, w1))

        return F.interpolate(
            depth_p1,
            (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
            mode='bilinear', align_corners=True)
