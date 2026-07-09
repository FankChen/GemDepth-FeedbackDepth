# Per-layer PURE-REFINE DPT head (control for DPTHeadPerLayer). 2026-07-09. STANDALONE.
#
# Same template as dpt_perlayer.py (original gemdepth DPT, 4 refinenet paths p4->p1), same main
# output (original output_conv, == baseline), same per-layer deep supervision on z_l'. The ONLY
# difference: the per-layer residual dz_l is predicted from the decoder FEATURE (a plain learned
# delta), NOT from a warp / error map. So `perlayer` vs `perlayer_refine` isolates the value of the
# error-map (geometric) signal versus a pure feature refinement.
#   step 1 (refine): depth_head_l(path_l)        -> z_l
#   step 2 (delta) : delta_head_l(feat(path_l))  -> dz_l   (feature-based, zero-init; NO warp)
#                    => z_l' = relu(z_l + dz_l)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_temporal import DPTHeadTemporal


class DPTHeadPerLayerRefine(DPTHeadTemporal):
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
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken, num_frames, pe)

        self.layer_depths = []       # z_l' = z_l + dz_l per layer (deep supervision target)
        self.layer_depths_pre = []   # z_l (refine only) per layer (ablation)

        sig_ch = 32
        # step 1: per-layer refine head (path feat -> non-negative coarse depth z_l). Softplus end.
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
        # project the path to a compact feature that drives the residual (mirrors perlayer's sig_proj)
        self.sig_proj = nn.ModuleDict({
            s: nn.Conv2d(features, sig_ch, kernel_size=3, stride=1, padding=1) for s in self.PATH_KEYS
        })
        # step 2: per-layer FEATURE delta head (feat -> residual dz_l). Last conv zero-init => dz=0 at init.
        self.layer_delta_heads = nn.ModuleDict({
            s: nn.Sequential(
                nn.Conv2d(sig_ch, sig_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(sig_ch, 1, kernel_size=3, stride=1, padding=1),
            ) for s in self.PATH_KEYS
        })
        for s in self.PATH_KEYS:
            nn.init.zeros_(self.layer_delta_heads[s][-1].weight)
            nn.init.zeros_(self.layer_delta_heads[s][-1].bias)

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

        # ---- per-layer refine (z_l) + FEATURE delta (dz_l): NO warp, deep supervision ----
        path_map = {'p4': path_4, 'p3': path_3, 'p2': path_2, 'p1': path_1}
        for s in self.PATH_KEYS:                                # coarse -> fine
            path_s = path_map[s]
            Hs, Ws = path_s.shape[-2:]
            z_l = self.layer_depth_heads[s](path_s)             # step 1: refine -> z_l
            self.layer_depths_pre.append(z_l.reshape(B, T, 1, Hs, Ws))
            dz = self.layer_delta_heads[s](self.sig_proj[s](path_s))  # step 2: feature delta (zero-init)
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
