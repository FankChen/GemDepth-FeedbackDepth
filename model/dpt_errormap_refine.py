# Refine error-map DPT head (师兄 / supervisor scheme: 4-stage, concat-fusion, depth1 -> depth2).
#
# This is a NEW controlled arm. It keeps the temporal DPT backbone of GemDepth untouched and,
# at EVERY top-down fusion stage (path_4 / path_3 / path_2 / path_1), performs the full
# "estimate -> warp -> correct" loop the supervisor described:
#
#   1. depth1 = depth_heads1[key](stage_feature)               # coarse depth at this stage
#   2. err    = photometric_error_map(neighbours warped by depth1 + GEM pose)   # cross-frame
#   3. err_feat = error_encoders[key]([err, valid])            # conv-encode the error map
#   4. refined = stage_feature + fuse_blocks[key]( concat[stage_feature, err_feat] )   # CONCAT fuse
#   5. depth2 = depth_heads2[key](refined)                     # refined depth (auxiliary)
#
# Differences vs the v1 error-map head (dpt_errormap.py):
#   * v1 injects at 3 stages (s4/s3/s2) and *adds* a zero-init error encoding.
#     This head injects at 4 stages (adds the finest s1 = path_1) and *concatenates* the error
#     encoding with the decoder feature before a fusion conv predicts the correction.
#   * v1 decodes one coarse depth per stage (only to drive the warp). This head decodes two:
#     depth1 (drives the warp) and depth2 (the post-fusion refined depth), realising the
#     explicit depth1 -> depth2 two-step refinement. Both are pushed to ``aux_depths`` so the
#     existing training.aux_depth_weight supervision covers them.
#
# Stability: the last conv of every ``fuse_blocks`` entry is zero-initialised, so at init
# ``refined == stage_feature`` exactly and the whole head reproduces the baseline temporal
# head — fine-tuning from the pretrained GemDepth weights starts as a no-op, just like v1.
#
# Each frame is warped against BOTH its previous (offset -1) and next (offset +1) neighbour and
# the per-pixel minimum reprojection residual is kept (handled inside photometric_error_map).

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_temporal import DPTHeadTemporal
from .util.warp import photometric_error_map, scale_intrinsics


def _make_depth_head(features):
    """Small conv stack that maps a stage feature (features ch) to a positive depth map."""
    return nn.Sequential(
        nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        nn.Softplus(),
    )


class DPTHeadErrorMapRefine(DPTHeadTemporal):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        error_stages=('s4', 's3', 's2', 's1'),
        warp_offsets=(-1, 1),
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken, num_frames, pe)

        self.error_stages = tuple(error_stages)
        self.warp_offsets = tuple(warp_offsets)
        self.aux_depths = []
        # When set to a list, every warp performed in _inject is recorded for visualisation
        # (see scripts/visualize_warp.py). Default None -> no capture, no behaviour change.
        self.capture_warps = None

        self.depth_heads1 = nn.ModuleDict()   # depth1: drives the warp
        self.depth_heads2 = nn.ModuleDict()   # depth2: refined depth after fusion (auxiliary)
        self.error_encoders = nn.ModuleDict()
        self.fuse_blocks = nn.ModuleDict()
        for key in self.error_stages:
            self.depth_heads1[key] = _make_depth_head(features)
            self.depth_heads2[key] = _make_depth_head(features)
            self.error_encoders[key] = nn.Sequential(
                nn.Conv2d(2, features // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features // 2, features, kernel_size=3, stride=1, padding=1),
            )
            # Concat[stage_feature, err_feat] (2*features) -> correction (features). The last
            # conv is zero-init so the correction starts at 0 => identity at init.
            fuse = nn.Sequential(
                nn.Conv2d(2 * features, features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            )
            nn.init.zeros_(fuse[-1].weight)
            nn.init.zeros_(fuse[-1].bias)
            self.fuse_blocks[key] = fuse

    def _inject(self, key, path_feat, images, extrinsics, intrinsics, B, T):
        """Estimate depth1, warp neighbours, concat-fuse the error map, predict depth2."""
        if key not in self.error_stages:
            return path_feat
        if images is None or extrinsics is None or intrinsics is None:
            return path_feat

        BT, _, h, w = path_feat.shape
        depth1 = self.depth_heads1[key](path_feat.float())          # (BT,1,h,w), > 0
        self.aux_depths.append(depth1.reshape(B, T, 1, h, w))

        H0, W0 = images.shape[-2:]
        imgs = F.interpolate(images.flatten(0, 1).float(), size=(h, w),
                             mode='bilinear', align_corners=False)
        imgs = imgs.reshape(B, T, imgs.shape[1], h, w)
        depth_bt = depth1.reshape(B, T, 1, h, w)

        K = scale_intrinsics(intrinsics.detach().float(), (H0, W0), (h, w))
        ext = extrinsics.detach().float()

        if self.capture_warps is not None:
            n0 = len(self.capture_warps)
            err, valid = photometric_error_map(imgs, depth_bt, K, ext, offsets=self.warp_offsets,
                                               capture=self.capture_warps, tag=f'{key}/rgb')
            for rec in self.capture_warps[n0:]:
                rec['depth'] = depth_bt.detach().cpu()
        else:
            err, valid = photometric_error_map(imgs, depth_bt, K, ext, offsets=self.warp_offsets)

        err_in = torch.cat([err, valid], dim=2).reshape(BT, 2, h, w)
        err_feat = self.error_encoders[key](err_in.to(path_feat.dtype))

        fused = torch.cat([path_feat, err_feat], dim=1)             # (BT, 2*features, h, w)
        refined = path_feat + self.fuse_blocks[key](fused)          # zero-init -> identity at init

        depth2 = self.depth_heads2[key](refined.float())            # (BT,1,h,w), > 0
        self.aux_depths.append(depth2.reshape(B, T, 1, h, w))
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
        path_4 = self._inject('s4', path_4, images, extrinsics, intrinsics, B, T)

        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, mode, size=layer_2_rn.shape[2:])
        path_3 = self.motion_modules[3](path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_3 = self._inject('s3', path_3, images, extrinsics, intrinsics, B, T)

        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, mode, size=layer_1_rn.shape[2:])
        path_2 = self._inject('s2', path_2, images, extrinsics, intrinsics, B, T)

        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, mode)
        path_1 = self._inject('s1', path_1, images, extrinsics, intrinsics, B, T)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        ori_type = out.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.scratch.output_conv2(out.float())

        return out.to(ori_type)
