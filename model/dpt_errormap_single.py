# Single-stage error-map DPT head (user-specified design, 2026-06-30).
#
# For every frame of the clip, at the FINAL DPT decoder stage (path_1) ONLY:
#   1. z1      = depth_head1(path_1)                 # coarse per-frame depth -> auxiliary loss
#   2. e1,valid = reprojection error map: warp the t-1 / t+1 neighbours into the current frame
#                 using z1 + GEM pose (detached), keep the per-pixel min residual (monodepth2)
#   3. refined = path_1 + fuse(concat[path_1, enc(e1)])   # fuse last conv zero-init => identity at init
#   4. z'1     = output_conv(refined)                # FINAL per-frame depth -> main loss
#
# Differences vs the 4-stage dpt_errormap_refine.py:
#   * injects ONCE at path_1 (not s4/s3/s2/s1);
#   * the refined feature is routed through the ORIGINAL output_conv, so z'1 IS the model's
#     final depth (refine.py decodes depth2 with a separate auxiliary head instead);
#   * only one auxiliary depth (z1) is produced.
#
# ``warp_signal`` selects what is warped and differenced:
#   * 'rgb'  (default): photometric error on the (downsampled) input frames;
#   * 'feat'          : feature-metric error on the path_1 decoder feature itself.
#
# Stability: the last conv of ``fuse_block`` is zero-initialised, so at init ``refined == path_1``
# exactly and the head reproduces the baseline temporal head — head-only fine-tuning from the
# pretrained GemDepth weights starts as a no-op.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_temporal import DPTHeadTemporal
from .util.warp import signal_error_map, scale_intrinsics


def _make_depth_head(features):
    """Small conv stack mapping a stage feature (features ch) to a positive depth map."""
    return nn.Sequential(
        nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        nn.Softplus(),
    )


class DPTHeadErrorMapSingle(DPTHeadTemporal):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        warp_offsets=(-1, 1),
        warp_signal='rgb',
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken, num_frames, pe)

        assert warp_signal in ('rgb', 'feat'), f"warp_signal must be 'rgb' or 'feat', got {warp_signal}"
        self.warp_offsets = tuple(warp_offsets)
        self.warp_signal = warp_signal
        self.aux_depths = []
        # When set to a list, the warp is recorded for visualisation; default None -> no capture.
        self.capture_warps = None

        # z1: coarse depth that drives the warp and is auxiliary-supervised.
        self.depth_head1 = _make_depth_head(features)
        # Encode the 2-channel error map (residual + valid) up to ``features`` so the concat is balanced.
        self.error_encoder = nn.Sequential(
            nn.Conv2d(2, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, features, kernel_size=3, stride=1, padding=1),
        )
        # Concat[path_feat, err_feat] (2*features) -> correction (features); last conv zero-init
        # so the correction starts at 0 => identity at init.
        fuse = nn.Sequential(
            nn.Conv2d(2 * features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
        )
        nn.init.zeros_(fuse[-1].weight)
        nn.init.zeros_(fuse[-1].bias)
        self.fuse_block = fuse

    def _inject(self, path_feat, images, extrinsics, intrinsics, B, T):
        """z1 -> warp t±1 neighbours -> concat-fuse the error map -> refined feature."""
        if images is None or extrinsics is None or intrinsics is None:
            return path_feat

        BT, _, h, w = path_feat.shape
        z1 = self.depth_head1(path_feat.float())                   # (BT,1,h,w), > 0
        self.aux_depths.append(z1.reshape(B, T, 1, h, w))

        H0, W0 = images.shape[-2:]
        depth_bt = z1.reshape(B, T, 1, h, w)
        K = scale_intrinsics(intrinsics.detach().float(), (H0, W0), (h, w))
        ext = extrinsics.detach().float()

        if self.warp_signal == 'feat':
            signal = path_feat.reshape(B, T, -1, h, w).float()     # feature-metric error
            tag = 'single/feat'
        else:
            imgs = F.interpolate(images.flatten(0, 1).float(), size=(h, w),
                                 mode='bilinear', align_corners=False)
            signal = imgs.reshape(B, T, imgs.shape[1], h, w)       # photometric error
            tag = 'single/rgb'

        if self.capture_warps is not None:
            n0 = len(self.capture_warps)
            err, valid = signal_error_map(signal, depth_bt, K, ext, offsets=self.warp_offsets,
                                          capture=self.capture_warps, tag=tag)
            for rec in self.capture_warps[n0:]:
                rec['depth'] = depth_bt.detach().cpu()
        else:
            err, valid = signal_error_map(signal, depth_bt, K, ext, offsets=self.warp_offsets)

        err_in = torch.cat([err, valid], dim=2).reshape(BT, 2, h, w)
        err_feat = self.error_encoder(err_in.to(path_feat.dtype))

        fused = torch.cat([path_feat, err_feat], dim=1)            # (BT, 2*features, h, w)
        refined = path_feat + self.fuse_block(fused)               # zero-init -> identity at init
        return refined

    def forward(self, out_features, patch_h, patch_w, frame_length,
                images=None, extrinsics=None, intrinsics=None,
                layer_3_att=None, layer_4_att=None, mode=None):
        mode = False
        self.aux_depths = []
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        B, T = layer_1.shape[0] // frame_length, frame_length

        layer_3 = self.motion_modules[0](layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        layer_4 = self.motion_modules[1](layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)

        if layer_3_att is not None:
            layer_3 = layer_3_att + layer_3
        if layer_4_att is not None:
            layer_4 = layer_4_att + layer_4

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, mode, size=layer_3_rn.shape[2:])
        path_4 = self.motion_modules[2](path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)

        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, mode, size=layer_2_rn.shape[2:])
        path_3 = self.motion_modules[3](path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)

        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, mode, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, mode)

        # ---- single-stage error-map injection at the final decoder feature ----
        path_1 = self._inject(path_1, images, extrinsics, intrinsics, B, T)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        ori_type = out.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.scratch.output_conv2(out.float())

        return out.to(ori_type)
