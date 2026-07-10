# Per-layer cascaded refine + ERROR-MAP DPT head (method arm). 2026-07-10.
#
# Built ON TOP of the cascaded refine (dpt_perlayer_refine.py). Coarse->fine over p4..p1:
#   z = refine(path, up(prev_z))                    # same cascade as method-1
#   z = relu(z + errmap_delta(z, pose, signal))     # then a warp error-map correction, per layer
# EVERY layer's z (after errmap) is a real depth with its OWN single-frame loss (4 losses), and the
# finest z is the FINAL output. errmap_delta is zero-init => at init it's a no-op (== pure cascade),
# so training starts from method-1 and learns the geometric correction on top. warp_signal picks the
# reprojection signal (rgb | feat | rgbfeat | hog). Reuses signal_error_map / hog_feature_map.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_temporal import DPTHeadTemporal
from .util.warp import signal_error_map, scale_intrinsics
from .util.hog import hog_feature_map


class DPTHeadPerLayerErrmap(DPTHeadTemporal):
    PATH_KEYS = ('p4', 'p3', 'p2', 'p1')

    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        warp_signal='feat',
        warp_offsets=(-1, 1),
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken, num_frames, pe)

        assert warp_signal in ('rgb', 'feat', 'rgbfeat', 'hog'), \
            f"warp_signal must be rgb|feat|rgbfeat|hog, got {warp_signal}"
        self.warp_signal = warp_signal
        self.warp_offsets = tuple(warp_offsets)
        self.hog_nbins = 9
        self.layer_depths = []       # z per layer AFTER errmap (4 deep-supervision targets)

        sig_ch = 32
        # cascaded refine (identical to method-1) --------------------------------------------------
        self.refine_p4 = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, sig_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(sig_ch, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus(),
        )
        self.refine_fine = nn.ModuleDict({
            s: nn.Sequential(
                nn.Conv2d(features + 1, features // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features // 2, sig_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(sig_ch, 1, kernel_size=3, stride=1, padding=1),
            ) for s in ('p3', 'p2', 'p1')
        })
        # per-layer error-map correction ----------------------------------------------------------
        self.sig_proj = nn.ModuleDict({
            s: nn.Conv2d(features, sig_ch, kernel_size=3, stride=1, padding=1) for s in self.PATH_KEYS
        })
        n_streams = 2 if warp_signal == 'rgbfeat' else 1
        self.error_encoders = nn.ModuleDict({
            s: nn.Sequential(
                nn.Conv2d(2 * n_streams, sig_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(sig_ch, 1, kernel_size=3, stride=1, padding=1),
            ) for s in self.PATH_KEYS
        })
        for s in self.PATH_KEYS:
            nn.init.zeros_(self.error_encoders[s][-1].weight)   # errmap delta = 0 at init (no-op)
            nn.init.zeros_(self.error_encoders[s][-1].bias)

    def _signals(self, s, path_s, images, B, T, Hs, Ws):
        def _imgs():
            im = F.interpolate(images.flatten(0, 1).float(), size=(Hs, Ws),
                               mode='bilinear', align_corners=False)
            return im.reshape(B, T, im.shape[1], Hs, Ws)

        streams = []
        if self.warp_signal in ('rgb', 'rgbfeat'):
            streams.append(_imgs())
        if self.warp_signal in ('feat', 'rgbfeat'):
            streams.append(self.sig_proj[s](path_s).reshape(B, T, -1, Hs, Ws).float())
        if self.warp_signal == 'hog':
            streams.append(hog_feature_map(_imgs(), nbins=self.hog_nbins).float())
        return streams

    def _errmap_delta(self, s, z, path_s, images, ext, intrinsics, B, T, H0, W0):
        Hs, Ws = z.shape[-2:]
        depth_bt = z.reshape(B, T, 1, Hs, Ws)
        K_s = scale_intrinsics(intrinsics.detach().float(), (H0, W0), (Hs, Ws))
        parts = []
        for sig in self._signals(s, path_s, images, B, T, Hs, Ws):
            err, valid = signal_error_map(sig, depth_bt, K_s, ext, offsets=self.warp_offsets)
            parts.append(err)
            parts.append(valid)
        err_in = torch.cat(parts, dim=2).reshape(B * T, -1, Hs, Ws)
        return self.error_encoders[s](err_in.to(z.dtype))

    def forward(self, out_features, patch_h, patch_w, frame_length,
                images=None, extrinsics=None, intrinsics=None,
                layer_3_att=None, layer_4_att=None, mode=None):
        mode = False
        self.layer_depths = []

        # ---- original gemdepth DPT backbone (verbatim from DPTHeadTemporal) ----
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

        # ---- cascaded refine + per-layer errmap; every layer's z (after errmap) gets a loss ----
        H0, W0 = (images.shape[-2:] if images is not None else (int(patch_h * 14), int(patch_w * 14)))
        ext = extrinsics.detach().float() if extrinsics is not None else None
        can_warp = images is not None and ext is not None and intrinsics is not None

        z = self.refine_p4(path_4)                                              # initial depth
        if can_warp:
            z = F.relu(z + self._errmap_delta('p4', z, path_4, images, ext, intrinsics, B, T, H0, W0))
        self.layer_depths.append(z.reshape(B, T, 1, *z.shape[-2:]))
        for s, path_s in (('p3', path_3), ('p2', path_2), ('p1', path_1)):
            Hs, Ws = path_s.shape[-2:]
            z_up = F.interpolate(z, size=(Hs, Ws), mode='bilinear', align_corners=True)
            delta = self.refine_fine[s](torch.cat([path_s, z_up], dim=1))       # refine step
            z = F.relu(z_up + delta)
            if can_warp:                                                        # errmap step
                z = F.relu(z + self._errmap_delta(s, z, path_s, images, ext, intrinsics, B, T, H0, W0))
            self.layer_depths.append(z.reshape(B, T, 1, Hs, Ws))

        out = F.interpolate(z, (int(patch_h * 14), int(patch_w * 14)),
                            mode='bilinear', align_corners=True)
        return out
