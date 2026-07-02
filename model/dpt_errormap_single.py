# Single-stage error-map DPT head (user-specified design, 2026-06-30).
#
# The ORIGINAL temporal DPT head is run to completion FIRST, so the warp operates on the real
# full-resolution output depth (not a hidden intermediate feature). For every frame:
#   1. feat = interpolate(output_conv1(path_1), x14)   # penultimate full-res feature (features//2 ch)
#   2. z1   = output_conv2(feat)                        # ORIGINAL full-res depth   -> auxiliary loss
#   3. e1,valid = reprojection error map: warp the t-1 / t+1 neighbours into the current frame
#                 using z1 + GEM pose (detached), keep the per-pixel min residual (monodepth2)
#   4. z'1  = z1.detach() + refine_head(concat[feat, enc(e1)])   # refined depth -> main loss
#
# z1 is DETACHED in step 4 (and in the warp geometry), so the MAIN loss trains only the refine
# modules; z1 itself is supervised solely by its own auxiliary loss. z'1 is the main / inference output.
#
# ``warp_signal`` selects what is warped and differenced to form the error map:
#   * 'rgb'  (default): photometric error on the (full-res) input frames;
#   * 'feat'          : feature-metric error on the penultimate decoder feature ``feat``;
#   * 'hog'           : feature error on a dense HOG descriptor of the frames;
#   * 'rgbfeat'       : rgb-error and feat-error streams concatenated (4ch) then encoded.
#
# Stability: the last conv of ``refine_head`` is zero-initialised, so at init ``z'1 == z1``
# exactly and the head reproduces the baseline temporal head — head-only fine-tuning from the
# pretrained GemDepth weights starts as a no-op.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_temporal import DPTHeadTemporal
from .util.warp import signal_error_map, scale_intrinsics
from .util.hog import hog_feature_map


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

        assert warp_signal in ('rgb', 'feat', 'rgbfeat', 'hog'), \
            f"warp_signal must be one of rgb|feat|rgbfeat|hog, got {warp_signal}"
        self.warp_offsets = tuple(warp_offsets)
        self.warp_signal = warp_signal
        self.hog_nbins = 9
        self.aux_depths = []
        # When set to a list, the warp is recorded for visualisation; default None -> no capture.
        self.capture_warps = None

        # The penultimate full-res feature ``feat`` has ``features // 2`` channels (output_conv1).
        feat_ch = features // 2
        # Each warped signal yields a 2-channel error map (residual + valid). Single-stream arms
        # (rgb|feat|hog) feed 2ch; 'rgbfeat' concatenates the rgb-error and feat-error streams (4ch).
        n_streams = 2 if warp_signal == 'rgbfeat' else 1
        err_in_ch = 2 * n_streams
        # Encode the error map up to ``feat_ch`` so the concat with ``feat`` is balanced.
        self.error_encoder = nn.Sequential(
            nn.Conv2d(err_in_ch, max(feat_ch // 2, 1), kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(max(feat_ch // 2, 1), feat_ch, kernel_size=3, stride=1, padding=1),
        )
        # Concat[feat, err_feat] (2*feat_ch) -> a 1-channel depth correction; last conv zero-init
        # so the correction starts at 0 => z'1 == z1 at init (reproduces the baseline head).
        refine = nn.Sequential(
            nn.Conv2d(2 * feat_ch, feat_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feat_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )
        nn.init.zeros_(refine[-1].weight)
        nn.init.zeros_(refine[-1].bias)
        self.refine_head = refine

    def _build_signals(self, feat, images, B, T, Hf, Wf):
        """Return [(signal, tag), ...] to warp, per ``warp_signal`` arm.

        Each signal is (B,T,C,H,W) at output resolution; ``signal_error_map`` collapses any C
        into a 1-channel residual, so every stream yields a 2ch (residual+valid) error map:
          * rgb     : input frames (photometric)                    -> 1 stream
          * feat    : penultimate decoder feature (feature-metric)  -> 1 stream
          * hog     : dense HOG descriptor of the frames            -> 1 stream
          * rgbfeat : rgb and feat streams together                 -> 2 streams (4ch)
        """
        def _imgs():
            im = F.interpolate(images.flatten(0, 1).float(), size=(Hf, Wf),
                               mode='bilinear', align_corners=False)
            return im.reshape(B, T, im.shape[1], Hf, Wf)

        streams = []
        if self.warp_signal in ('rgb', 'rgbfeat'):
            streams.append((_imgs(), 'single/rgb'))
        if self.warp_signal in ('feat', 'rgbfeat'):
            streams.append((feat.reshape(B, T, -1, Hf, Wf).float(), 'single/feat'))
        if self.warp_signal == 'hog':
            streams.append((hog_feature_map(_imgs(), nbins=self.hog_nbins).float(), 'single/hog'))
        return streams

    def _refine(self, z1, feat, images, extrinsics, intrinsics, B, T):
        """warp t±1 neighbours with z1 -> error map -> concat with feat -> depth correction.

        ``z1``   : (BT,1,Hf,Wf) the original full-res output depth (drives the warp geometry).
        ``feat`` : (BT,feat_ch,Hf,Wf) the penultimate full-res feature (mixed with the error map).
        z1 is DETACHED here (warp + residual), so the main loss trains only the refine modules;
        z1 itself is supervised solely by its auxiliary loss. Returns z'1 == z1 at init (zero-init).
        """
        if images is None or extrinsics is None or intrinsics is None:
            return z1

        BT, _, Hf, Wf = z1.shape
        # Detach z1 everywhere it feeds z'1 (warp geometry below + residual at the end) so the
        # MAIN loss never updates z1 -- z1 is trained ONLY by its own auxiliary loss.
        z1_det = z1.detach()
        depth_bt = z1_det.reshape(B, T, 1, Hf, Wf)
        # Scale intrinsics from the input-image resolution to the full output resolution, and
        # detach the GEM pose/camera so the warp does not back-propagate into them.
        H0, W0 = images.shape[-2:]
        K = scale_intrinsics(intrinsics.detach().float(), (H0, W0), (Hf, Wf))
        ext = extrinsics.detach().float()

        # Build the per-frame signal(s) to warp+difference (see ``_build_signals``). Each stream
        # yields a 2ch (residual+valid) map; 'rgbfeat' stacks rgb-error and feat-error -> 4ch.
        streams = self._build_signals(feat, images, B, T, Hf, Wf)

        err_parts = []
        for signal, tag in streams:
            if self.capture_warps is not None:
                n0 = len(self.capture_warps)
                err, valid = signal_error_map(signal, depth_bt, K, ext, offsets=self.warp_offsets,
                                              capture=self.capture_warps, tag=tag)
                for rec in self.capture_warps[n0:]:
                    rec['depth'] = depth_bt.detach().cpu()
            else:
                err, valid = signal_error_map(signal, depth_bt, K, ext, offsets=self.warp_offsets)
            err_parts.append(err)
            err_parts.append(valid)

        err_in = torch.cat(err_parts, dim=2).reshape(BT, 2 * len(streams), Hf, Wf)
        err_feat = self.error_encoder(err_in.to(feat.dtype))
        # errmap 编码后与全分辨率特征 feat 拼接，卷出 1 通道深度修正量，残差叠加到 z1。
        fused = torch.cat([feat, err_feat], dim=1)                 # (BT, 2*feat_ch, Hf, Wf)
        # 只在「预测 delta z」的最后一步用 fp32（同 output_conv2）；上面的 warp/编码跟随外层混合精度。
        with torch.autocast(device_type="cuda", enabled=False):
            delta = self.refine_head(fused.float())
        return z1_det + delta                                       # z1 detached => main loss trains only refine; == z1 at init

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

        # --- run the ORIGINAL head to completion: feat (penultimate, full-res) -> z1 (depth) ---
        feat = self.scratch.output_conv1(path_1)
        feat = F.interpolate(feat, (int(patch_h * 14), int(patch_w * 14)),
                             mode="bilinear", align_corners=True)
        ori_type = feat.dtype
        # 只有「预测 z1」的最后一层(output_conv2)用 fp32 保精度(同 baseline)；其余保持外层混合精度。
        with torch.autocast(device_type="cuda", enabled=False):
            z1 = self.scratch.output_conv2(feat.float())           # (BT,1,Hf,Wf) ORIGINAL depth
        Hf, Wf = z1.shape[-2:]
        self.aux_depths.append(z1.reshape(B, T, 1, Hf, Wf))        # z1 -> auxiliary supervision
        # --- error-map refine on the real output depth -> z'1 (main / inference output) ---
        z_refined = self._refine(z1, feat, images, extrinsics, intrinsics, B, T)

        return z_refined.to(ori_type)
