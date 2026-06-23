# Error-map DPT head (method arm).
#
# Extends the temporal DPT head of GemDepth. After each top-down fusion stage we decode
# a coarse depth, inverse-warp neighbouring frames with it and the GEM-predicted camera
# pose, and turn the resulting photometric residual into an "error map". That error map is
# injected (added) into the stage feature before the next fusion, so the decoder can focus
# refinement where the current geometry is inconsistent across the video.
#
# The error-map encoders are zero-initialised so that, at init, the head is identical to the
# baseline temporal head — this keeps fine-tuning from the pretrained GemDepth weights stable.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_temporal import DPTHeadTemporal
from .util.warp import photometric_error_map, scale_intrinsics


class DPTHeadErrorMap(DPTHeadTemporal):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        error_stages=('s4', 's3', 's2'),
        warp_offsets=(-1, 1),
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken, num_frames, pe)

        self.error_stages = tuple(error_stages)
        self.warp_offsets = tuple(warp_offsets)
        self.aux_depths = []

        self.depth_heads = nn.ModuleDict()
        self.error_encoders = nn.ModuleDict()
        for key in self.error_stages:
            self.depth_heads[key] = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.Softplus(),
            )
            enc = nn.Sequential(
                nn.Conv2d(2, features // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features // 2, features, kernel_size=3, stride=1, padding=1),
            )
            # Zero-init the last conv so injection starts as a no-op (identity at init).
            nn.init.zeros_(enc[-1].weight)
            nn.init.zeros_(enc[-1].bias)
            self.error_encoders[key] = enc

    def _inject(self, key, path_feat, images, extrinsics, intrinsics, B, T):
        """Decode coarse depth, build the photometric error map and add it to ``path_feat``."""
        if key not in self.error_stages:
            return path_feat
        if images is None or extrinsics is None or intrinsics is None:
            return path_feat

        BT, _, h, w = path_feat.shape
        depth_s = self.depth_heads[key](path_feat.float())          # (BT,1,h,w), > 0
        self.aux_depths.append(depth_s.reshape(B, T, 1, h, w))

        H0, W0 = images.shape[-2:]
        imgs = F.interpolate(images.flatten(0, 1).float(), size=(h, w),
                             mode='bilinear', align_corners=False)
        imgs = imgs.reshape(B, T, imgs.shape[1], h, w)
        depth_bt = depth_s.reshape(B, T, 1, h, w)

        K = scale_intrinsics(intrinsics.detach().float(), (H0, W0), (h, w))
        ext = extrinsics.detach().float()

        err, valid = photometric_error_map(imgs, depth_bt, K, ext, offsets=self.warp_offsets)
        err_in = torch.cat([err, valid], dim=2).reshape(BT, 2, h, w)
        err_feat = self.error_encoders[key](err_in.to(path_feat.dtype))
        return path_feat + err_feat

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

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        ori_type = out.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.scratch.output_conv2(out.float())

        return out.to(ori_type)
