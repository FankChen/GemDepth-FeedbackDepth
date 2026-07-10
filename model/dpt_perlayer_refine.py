# Per-layer cascaded PURE-REFINE DPT head. 2026-07-10.
#
# Template = original gemdepth DPT backbone (4 refinenet paths p4->p1). Coarse->fine cascade:
# p4 predicts an initial depth z; each finer layer takes cat(path, upsampled z) and predicts a
# residual, z = relu(up(z) + residual). EVERY layer's z is a real depth prediction with its OWN
# single-frame loss (4 losses total, via train.py compute_deep_sup_loss), and the FINEST z is the
# FINAL output -- so each layer's loss directly optimises the output depth (nothing frozen out).
# NO warp / error map here (that is the perlayer_errmap head, method arm).

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

        self.layer_depths = []       # z per layer (4 deep-supervision targets, coarse->fine)

        sig_ch = 32
        # Coarsest layer p4: path feature -> initial depth (Softplus keeps it > 0).
        self.refine_p4 = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, sig_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(sig_ch, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus(),
        )
        # Finer layers p3,p2,p1: cat(path, upsampled prev depth) -> residual added onto prev depth.
        self.refine_fine = nn.ModuleDict({
            s: nn.Sequential(
                nn.Conv2d(features + 1, features // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features // 2, sig_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(sig_ch, 1, kernel_size=3, stride=1, padding=1),
            ) for s in ('p3', 'p2', 'p1')
        })

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

        # ---- cascaded coarse->fine refine. Each layer's z is a real depth with its own loss; ----
        # ---- the finest z is the FINAL output, so every layer's loss optimises the output. ----
        z = self.refine_p4(path_4)                              # (BT,1,H4,W4) initial depth
        self.layer_depths.append(z.reshape(B, T, 1, *z.shape[-2:]))
        for s, path_s in (('p3', path_3), ('p2', path_2), ('p1', path_1)):
            Hs, Ws = path_s.shape[-2:]
            z_up = F.interpolate(z, size=(Hs, Ws), mode='bilinear', align_corners=True)
            delta = self.refine_fine[s](torch.cat([path_s, z_up], dim=1))
            z = F.relu(z_up + delta)                            # refine on top of previous depth
            self.layer_depths.append(z.reshape(B, T, 1, Hs, Ws))

        # final output = finest cascaded depth, upsampled to full resolution
        out = F.interpolate(z, (int(patch_h * 14), int(patch_w * 14)),
                            mode='bilinear', align_corners=True)
        return out
