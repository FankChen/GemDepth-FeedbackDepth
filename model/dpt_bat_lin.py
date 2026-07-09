# BAT+Lin head: warp error-map (main idea) fused with multi-scale iterative delta-d
# (BA-T + LinStereo inspired). 2026-07-09. STANDALONE — reuses signal_error_map / hog_feature_map
# but does NOT modify the existing errormap_single head or the warp/hog utilities.
#
# Main idea (kept): the temporal-reprojection ERROR MAP — warp t±1 neighbours with the current
# depth + GEM pose and measure a per-pixel residual (rgb | feat | rgbfeat | hog). See warp.py.
#
# Add-ons (icing), fusing two 2026 papers:
#   * BA-T (arXiv:2606.03287): bundle adjustment as an ITERATIVE residual update implemented by a
#     single lightweight, REPEATABLE layer (not deep stacks). -> one shared ``BAUpdateBlock``
#     applied across scales, producing a residual depth correction from the latent + error map.
#   * LinStereo (arXiv:2606.25437): MULTI-SCALE, scale-aligned refinement (HSCV) with a monocular
#     depth WARM-START (DPI). -> warm-start from the baseline depth ``z1`` and refine coarse->fine
#     over the DPT decoder paths (p4->p1), a configurable subset (``scales``) => "any #layers".
#
# Stability / diagnostic: each scale's correction is z += interp(gate * delta), with the delta conv
# ZERO-INITIALISED and a learnable per-pixel sigmoid ``gate``. At init every correction is 0, so the
# head output == the baseline temporal depth exactly (a no-op), and ``gate`` lets the model fall
# back to baseline (safety net for the D1 finding that a free delta was net-harmful).

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_temporal import DPTHeadTemporal
from .util.warp import signal_error_map, scale_intrinsics
from .util.hog import hog_feature_map


class BAUpdateBlock(nn.Module):
    """BA-T-style shared, lightweight, repeatable update layer.

    Given the (scale-projected) decoder latent and the encoded error map, predict a residual
    depth correction ``delta`` and a per-pixel ``gate`` in [0,1]. The last delta conv is
    zero-initialised so the block is a no-op at init (output correction == 0).
    """

    def __init__(self, feat_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * feat_ch, feat_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.delta = nn.Conv2d(feat_ch, 1, kernel_size=1, stride=1, padding=0)
        self.gate = nn.Conv2d(feat_ch, 1, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)  # zero-init => delta == 0 at init (no-op / warm-start kept)

    def forward(self, latent, err_feat):
        h = self.fuse(torch.cat([latent, err_feat], dim=1))
        delta = self.delta(h)                       # residual depth correction
        gate = torch.sigmoid(self.gate(h))          # per-pixel [0,1] selectivity
        return delta, gate


class DPTHeadBATLin(DPTHeadTemporal):
    """Multi-scale iterative error-map refinement head (warp main-idea + BA-T + LinStereo).

    Flow: run the ORIGINAL temporal decoder to get paths p4..p1 and the baseline depth z1
    (warm-start, aux-supervised). Then, coarse->fine over ``scales``, warp-difference the current
    depth to form an error map, and add a gated residual correction (shared ``BAUpdateBlock``).
    """

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
        warp_offsets=(-1, 1),
        warp_signal='feat',
        scales=('p2', 'p1'),
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken, num_frames, pe)

        assert warp_signal in ('rgb', 'feat', 'rgbfeat', 'hog'), \
            f"warp_signal must be one of rgb|feat|rgbfeat|hog, got {warp_signal}"
        for s in scales:
            assert s in self.PATH_KEYS, f"scale {s} not in {self.PATH_KEYS}"
        self.warp_offsets = tuple(warp_offsets)
        self.warp_signal = warp_signal
        self.scales = tuple(scales)          # coarse->fine subset of p4..p1 ("any #layers" = scale)
        self.hog_nbins = 9
        self.aux_depths = []
        self.capture_warps = None

        feat_ch = features // 2
        self.feat_ch = feat_ch
        # Project each decoder path (``features`` ch) down to the refine channel, per scale.
        self.scale_proj = nn.ModuleDict({
            s: nn.Conv2d(features, feat_ch, kernel_size=3, stride=1, padding=1) for s in self.scales
        })
        # Error-map encoder: 2ch per warped stream (residual+valid); 'rgbfeat' uses two streams (4ch).
        n_streams = 2 if warp_signal == 'rgbfeat' else 1
        self.error_encoder = nn.Sequential(
            nn.Conv2d(2 * n_streams, max(feat_ch // 2, 1), kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(max(feat_ch // 2, 1), feat_ch, kernel_size=3, stride=1, padding=1),
        )
        # Single SHARED update layer, repeated across scales (BA-T "repeatable layer").
        self.ba_update = BAUpdateBlock(feat_ch)

    def _signals_at_scale(self, feat_s, images, B, T, Hs, Ws):
        """Return [(signal, tag), ...] warped at scale (Hs,Ws); mirrors errormap_single semantics."""
        def _imgs():
            im = F.interpolate(images.flatten(0, 1).float(), size=(Hs, Ws),
                               mode='bilinear', align_corners=False)
            return im.reshape(B, T, im.shape[1], Hs, Ws)

        streams = []
        if self.warp_signal in ('rgb', 'rgbfeat'):
            streams.append((_imgs(), 'batlin/rgb'))
        if self.warp_signal in ('feat', 'rgbfeat'):
            streams.append((feat_s.reshape(B, T, -1, Hs, Ws).float(), 'batlin/feat'))
        if self.warp_signal == 'hog':
            streams.append((hog_feature_map(_imgs(), nbins=self.hog_nbins).float(), 'batlin/hog'))
        return streams

    def _refine_multiscale(self, z1, path_map, images, extrinsics, intrinsics, B, T):
        """Warm-start from z1, refine coarse->fine over ``scales`` with gated residual deltas."""
        if images is None or extrinsics is None or intrinsics is None:
            return z1

        BT, _, Hf, Wf = z1.shape
        # z1 is detached everywhere it feeds the main output (warp geometry + warm-start), so the
        # MAIN loss trains only the refine/BA modules; z1 is supervised by its own aux loss.
        z = z1.detach()
        H0, W0 = images.shape[-2:]
        ext = extrinsics.detach().float()

        for s in self.scales:                                  # coarse -> fine
            path_s = path_map[s]
            Hs, Ws = path_s.shape[-2:]
            feat_s = self.scale_proj[s](path_s)                # (BT, feat_ch, Hs, Ws)
            z_s = F.interpolate(z, size=(Hs, Ws), mode='bilinear', align_corners=True)
            depth_bt = z_s.reshape(B, T, 1, Hs, Ws)
            K_s = scale_intrinsics(intrinsics.detach().float(), (H0, W0), (Hs, Ws))

            err_parts = []
            for signal, _tag in self._signals_at_scale(feat_s, images, B, T, Hs, Ws):
                err, valid = signal_error_map(signal, depth_bt, K_s, ext, offsets=self.warp_offsets)
                err_parts.append(err)
                err_parts.append(valid)
            err_in = torch.cat(err_parts, dim=2).reshape(BT, 2 * len(err_parts) // 2, Hs, Ws)
            err_feat = self.error_encoder(err_in.to(feat_s.dtype))

            delta_s, gate_s = self.ba_update(feat_s, err_feat)  # (BT,1,Hs,Ws) each
            upd = F.interpolate(gate_s * delta_s, size=(Hf, Wf), mode='bilinear', align_corners=True)
            z = z + upd                                        # accumulate correction at full-res

        return z

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

        # ORIGINAL head: penultimate feature -> baseline depth z1 (warm-start + aux supervision).
        feat = self.scratch.output_conv1(path_1)
        feat = F.interpolate(feat, (int(patch_h * 14), int(patch_w * 14)),
                             mode="bilinear", align_corners=True)
        ori_type = feat.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            z1 = self.scratch.output_conv2(feat.float())        # (BT,1,Hf,Wf) baseline depth
        Hf, Wf = z1.shape[-2:]
        self.aux_depths.append(z1.reshape(B, T, 1, Hf, Wf))     # warm-start supervised by aux loss

        path_map = {'p4': path_4, 'p3': path_3, 'p2': path_2, 'p1': path_1}
        z_refined = self._refine_multiscale(z1, path_map, images, extrinsics, intrinsics, B, T)
        return z_refined.to(ori_type)
