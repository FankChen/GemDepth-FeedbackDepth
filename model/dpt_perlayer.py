# Per-layer deep-supervised refine + warp DPT head (师兄 idea, 2026-07-09). STANDALONE.
#
# Template = the ORIGINAL gemdepth DPT (DPTHeadTemporal): 4 refinenet paths p4->p1, and the
# original output_conv on path_1 is KEPT as the main output (pretrained warm-start, == baseline).
#
# Added, at EACH of the 4 paths, the "two steps" the 师兄 described:
#   step 1 (refine): depth_head_l(path_l)         -> z_l      a coarse depth at that layer
#   step 2 (warp):   warp z_l with pose -> error  -> dz_l     a residual correction (zero-init)
#                    => z_l' = relu(z_l + dz_l)
# Every layer's z_l' is stored in self.layer_depths for DEEP SUPERVISION (one loss per layer),
# and z_l (pre-warp) in self.layer_depths_pre so the ablation can supervise refine-only vs
# refine+warp. This head is used ONLY for the per-layer ablation; it does not touch other heads.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_temporal import DPTHeadTemporal
from .util.warp import signal_error_map, scale_intrinsics
from .util.hog import hog_feature_map


class DPTHeadPerLayer(DPTHeadTemporal):
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
        use_warp=True,
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken, num_frames, pe)

        assert warp_signal in ('rgb', 'feat', 'rgbfeat', 'hog'), \
            f"warp_signal must be rgb|feat|rgbfeat|hog, got {warp_signal}"
        self.warp_signal = warp_signal
        self.warp_offsets = tuple(warp_offsets)
        self.use_warp = use_warp
        self.hog_nbins = 9
        self.layer_depths = []       # z_l' = z_l + dz_l per layer (deep supervision target)
        self.layer_depths_pre = []   # z_l (refine only) per layer (ablation)

        sig_ch = 32
        # step 1: per-layer refine head (path feat -> non-negative coarse depth z_l).
        # Softplus (not ReLU) at the end avoids a dead-zero z_l that would kill the layer's gradient.
        self.layer_depth_heads = nn.ModuleDict({
            s: nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features // 2, sig_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(sig_ch, 1, kernel_size=1, stride=1, padding=0),
                nn.Softplus(),
            ) for s in self.PATH_KEYS
        })
        # cheap feat signal for warping (project path -> sig_ch)
        self.sig_proj = nn.ModuleDict({
            s: nn.Conv2d(features, sig_ch, kernel_size=3, stride=1, padding=1) for s in self.PATH_KEYS
        })
        # step 2: per-layer warp head (error map -> residual dz_l). Last conv zero-init => dz=0 at init.
        n_streams = 2 if warp_signal == 'rgbfeat' else 1
        self.layer_delta_heads = nn.ModuleDict({
            s: nn.Sequential(
                nn.Conv2d(2 * n_streams, sig_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(sig_ch, 1, kernel_size=3, stride=1, padding=1),
            ) for s in self.PATH_KEYS
        })
        for s in self.PATH_KEYS:
            nn.init.zeros_(self.layer_delta_heads[s][-1].weight)
            nn.init.zeros_(self.layer_delta_heads[s][-1].bias)

    def _signals(self, s, path_s, images, B, T, Hs, Ws):
        """warp signals at this layer's resolution; mirrors errormap/batlin semantics."""
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

    def _warp_delta(self, s, z_l, path_s, images, ext, intrinsics, B, T, H0, W0):
        Hs, Ws = z_l.shape[-2:]
        depth_bt = z_l.reshape(B, T, 1, Hs, Ws)
        K_s = scale_intrinsics(intrinsics.detach().float(), (H0, W0), (Hs, Ws))
        parts = []
        for sig in self._signals(s, path_s, images, B, T, Hs, Ws):
            err, valid = signal_error_map(sig, depth_bt, K_s, ext, offsets=self.warp_offsets)
            parts.append(err)
            parts.append(valid)
        err_in = torch.cat(parts, dim=2).reshape(B * T, -1, Hs, Ws)
        return self.layer_delta_heads[s](err_in.to(z_l.dtype))

    def forward(self, out_features, patch_h, patch_w, frame_length,
                images=None, extrinsics=None, intrinsics=None,
                layer_3_att=None, layer_4_att=None, mode=None):
        mode = False
        self.layer_depths = []
        self.layer_depths_pre = []

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

        # ---- per-layer refine (z_l) + warp (dz_l): the 师兄 "two steps", deep supervision ----
        path_map = {'p4': path_4, 'p3': path_3, 'p2': path_2, 'p1': path_1}
        H0, W0 = (images.shape[-2:] if images is not None else (int(patch_h * 14), int(patch_w * 14)))
        ext = extrinsics.detach().float() if extrinsics is not None else None
        can_warp = self.use_warp and images is not None and ext is not None and intrinsics is not None

        for s in self.PATH_KEYS:                                # coarse -> fine
            path_s = path_map[s]
            Hs, Ws = path_s.shape[-2:]
            z_l = self.layer_depth_heads[s](path_s)             # step 1: refine -> z_l
            self.layer_depths_pre.append(z_l.reshape(B, T, 1, Hs, Ws))
            dz = torch.zeros_like(z_l)
            if can_warp:
                dz = self._warp_delta(s, z_l, path_s, images, ext, intrinsics, B, T, H0, W0)  # step 2
            z_ref = F.relu(z_l + dz)                            # z_l' = z_l + dz_l
            self.layer_depths.append(z_ref.reshape(B, T, 1, Hs, Ws))

        # ---- main output = ORIGINAL output_conv on path_1 (pretrained warm-start, == baseline) ----
        feat = self.scratch.output_conv1(path_1)
        feat = F.interpolate(feat, (int(patch_h * 14), int(patch_w * 14)),
                             mode="bilinear", align_corners=True)
        ori_type = feat.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.scratch.output_conv2(feat.float())
        return out.to(ori_type)
