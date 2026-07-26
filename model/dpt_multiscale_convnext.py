# ConvNeXt (hierarchical) multi-scale coarse-to-fine depth refinement head.
#
# Same coarse-to-fine residual recursion as ``DPTHeadMultiScaleRefine``
# (``depth_i = upsample(depth_{i-1}).detach() + delta_Z_i``), but the input
# adapter is the ConvNeXt one inherited from ``DPTHeadTemporalConvNeXt``:
#   * per-level 1x1 projections (each ConvNeXt stage has a distinct channel count)
#   * resize_layers replaced by Identity (the native NCHW pyramid already exists)
#   * NO token -> NCHW reshape (features arrive as NCHW maps)
#
# This mirrors how ``DPTHeadTemporalConvNeXt`` relates to ``DPTHeadTemporal``, so
# ConvNeXt can run the exact same from-scratch multiscale experiment as the ViT
# backbones (experiment 7). Everything downstream (scratch / refinenet / temporal
# motion modules) is reused unchanged.
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dpt_convnext import DPTHeadTemporalConvNeXt


class DPTHeadMultiScaleRefineConvNeXt(DPTHeadTemporalConvNeXt):
    """ConvNeXt multi-scale refine head (hierarchical counterpart of
    :class:`~model.dpt_multiscale.DPTHeadMultiScaleRefine`)."""

    def __init__(self, in_channels_list, features=256, use_bn=False,
                 out_channels=[256, 512, 1024, 1024], use_clstoken=False,
                 num_frames=32, pe='ape', use_temporal=False, patch_size=4,
                 fullres_mode='none', depth_feedback=False, fp32_head=False):
        super().__init__(in_channels_list, features, use_bn, out_channels,
                         use_clstoken, num_frames=num_frames, pe=pe,
                         use_temporal=use_temporal, patch_size=patch_size)

        # Resolution at which each scale's delta head runs / outputs (see forward):
        #   'none'       -> native pyramid resolution (original behaviour)
        #   'last'       -> finest scale only: conv & output at full input resolution (Method A)
        #   'all'        -> every scale: conv & output at full resolution (outputs all full-res)
        #   'all_native' -> every scale: conv at full resolution, output resampled back to native
        #                   resolution (full-res convolution, coarse-to-fine supervision preserved)
        assert fullres_mode in ('none', 'last', 'all', 'all_native'), fullres_mode
        self.fullres_mode = fullres_mode
        # Depth feedback: encode the running depth (previous scale's accumulated z, detached) into
        # 32 channels and concat onto this scale's feature, so the delta head refines with knowledge
        # of the current depth instead of predicting it blind from features alone.
        self.depth_feedback = bool(depth_feedback)
        enc_ch = 32 if self.depth_feedback else 0
        if self.depth_feedback:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
            )

        # fp32_head: run each scale's delta head convs in FP32 (autocast disabled), matching the
        # baseline temporal head which computes its final depth regression (output_conv2) in FP32.
        # Purely protective for the precision-sensitive final regression; costs extra compute/memory
        # and is most expensive under full-res modes (all / all_native run the convs at full res).
        self.fp32_head = bool(fp32_head)

        # One residual-regression head per pyramid scale (coarse -> fine).
        # No final activation: delta_Z may be positive or negative.
        self.delta_heads = nn.ModuleList([
            self._make_delta_head(features, in_ch=features + enc_ch) for _ in range(4)
        ])
        # 【抗死 ReLU 修复】同 DPTHeadMultiScaleRefine：multiscale 绕过 output_conv2 的抗死初始化，
        # 默认初始化的带符号 delta 经外层 F.relu 会塌成全零。末层清零权重 + 最粗尺度 +0.5，
        # 初始深度为正常数、存活 relu，delta 权重从 0 学起，不进塌缩盆地。
        for scale_index, delta_head in enumerate(self.delta_heads):
            nn.init.zeros_(delta_head[-1].weight)                                     # 末层权重清零
            nn.init.constant_(delta_head[-1].bias, 0.5 if scale_index == 0 else 0.0)  # 最粗尺度正 bias 0.5

        # Per-scale depth predictions cached on every forward pass so the
        # training loop can apply multi-scale supervision (head.aux_depths).
        self.aux_depths = []

    @staticmethod
    def _make_delta_head(features, in_ch=None):
        if in_ch is None:
            in_ch = features
        return nn.Sequential(
            nn.Conv2d(in_ch, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def _build_pyramid(self, out_features, frame_length,
                       layer_3_att=None, layer_4_att=None, mode=False):
        """Run the (temporal) ConvNeXt DPT pyramid and return the 4 scale
        features ``[path_4, path_3, path_2, path_1]`` (coarse -> fine).

        ConvNeXt features arrive as native NCHW maps, so there is no token
        reshape; per-level 1x1 projections + Identity resize feed the shared
        scratch / refinenet decoder (same body as DPTHeadTemporalConvNeXt)."""
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)       # NCHW in -> project channels (resolution kept)
            x = self.resize_layers[i](x)  # Identity (pyramid already native)
            out.append(x)
        layer_1, layer_2, layer_3, layer_4 = out
        B, T = layer_1.shape[0] // frame_length, frame_length

        if self.use_temporal:
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
        if self.use_temporal:
            path_4 = self.motion_modules[2](path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, mode, size=layer_2_rn.shape[2:])
        if self.use_temporal:
            path_3 = self.motion_modules[3](path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, mode, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, mode)

        return [path_4, path_3, path_2, path_1]

    def forward(self, out_features, patch_h, patch_w, frame_length,
                init_depth=None, layer_3_att=None, layer_4_att=None, mode=None):
        """Coarse-to-fine multi-scale depth refinement over the ConvNeXt pyramid.

        In training mode returns a list of per-scale depth predictions
        (coarse -> fine), each upsampled to full input resolution
        ``[B*T, 1, H, W]``. In eval/inference mode returns a single tensor,
        the finest-scale refined depth.
        """
        mode = False

        paths = self._build_pyramid(out_features, frame_length,
                                    layer_3_att=layer_3_att, layer_4_att=layer_4_att, mode=mode)

        B, T = paths[0].shape[0] // frame_length, frame_length

        # Initialise depth at the coarsest scale. init_depth is detached so the
        # loss gradient never flows back into whatever produced it.
        coarse_size = paths[0].shape[2:]
        if init_depth is None:
            # Match DPTHeadMultiScaleRefine (ViT): start from a positive metric constant (40) so the
            # coarsest depth survives the outer ReLU and the L2 / metric target scale is reachable
            # (ConvNeXt previously started from zeros, which differs from the ViT head).
            depth_prev = paths[0].new_ones((paths[0].shape[0], 1, *coarse_size)) * 40
        else:
            depth_prev = F.interpolate(init_depth.detach().to(paths[0].dtype), size=coarse_size,
                                       mode="bilinear", align_corners=True)

        # Full input resolution (patch grid * head patch size), e.g. 288x960.
        full_size = (int(patch_h * self.head_patch_size), int(patch_w * self.head_patch_size))

        scale_depths = []
        deltas = []
        prev_upsampled = []
        last_index = len(paths) - 1
        for i, feat in enumerate(paths):
            native_size = feat.shape[2:]
            # Decide the resolution the delta head runs at (conv_size) and the resolution this
            # scale's output depth lives at (out_size), from fullres_mode:
            #   'none'       -> conv & output at native pyramid resolution (original).
            #   'last'       -> finest scale only: conv & output at full input resolution (Method A).
            #   'all'        -> every scale: conv & output at full resolution (all outputs full-res,
            #                   so coarse-to-fine survives only in the FEATURES, not the depth res).
            #   'all_native' -> every scale: conv at full resolution, output resampled back to
            #                   native resolution -> full-res convolution while keeping real
            #                   coarse-to-fine resolution supervision.
            if self.fullres_mode == 'last':
                conv_size = full_size if i == last_index else native_size
                out_size = conv_size
            elif self.fullres_mode == 'all':
                conv_size = full_size
                out_size = full_size
            elif self.fullres_mode == 'all_native':
                conv_size = full_size
                out_size = native_size
            else:  # 'none'
                conv_size = native_size
                out_size = native_size

            # Feature at conv_size (upsampled only when a full-res mode asks for it).
            feat_c = feat if feat.shape[2:] == conv_size else \
                F.interpolate(feat, size=conv_size, mode="bilinear", align_corners=True)
            # Running depth brought to this scale's output resolution for the residual add.
            if depth_prev.shape[2:] != out_size:
                depth_prev = F.interpolate(depth_prev, size=out_size,
                                           mode="bilinear", align_corners=True)

            if self.depth_feedback:
                # Encode the running depth (previous scale's accumulated z) and concat onto the
                # feature. Detached, so the loss at this scale still only trains this scale's delta
                # head (cross-scale gradient truncation preserved), but the head now *sees* the
                # depth it is correcting instead of predicting blind from features alone.
                dfeat = depth_prev.detach()
                if dfeat.shape[2:] != conv_size:
                    dfeat = F.interpolate(dfeat, size=conv_size, mode="bilinear", align_corners=True)
                feat_in = torch.cat([feat_c, self.depth_encoder(dfeat)], dim=1)
            else:
                feat_in = feat_c

            if self.fp32_head:
                # Match the baseline temporal head: compute the depth regression conv in FP32.
                ori_dtype = feat_in.dtype
                with torch.autocast(device_type="cuda", enabled=False):
                    delta_z = self.delta_heads[i](feat_in.float())  # at conv_size, FP32
                delta_z = delta_z.to(ori_dtype)
            else:
                delta_z = self.delta_heads[i](feat_in)              # at conv_size
            if delta_z.shape[2:] != out_size:                        # bring delta to output res
                delta_z = F.interpolate(delta_z, size=out_size,
                                        mode="bilinear", align_corners=True)
            # Gradient truncation across scales: the running depth is detached,
            # so the loss at this scale only trains this scale's delta_Z.
            depth_cur = depth_prev.detach() + delta_z

            scale_depths.append(depth_cur)
            deltas.append(delta_z)
            prev_upsampled.append(depth_prev)
            depth_prev = depth_cur

        resized_multilevel_depths = [
            F.interpolate(zi, full_size, mode="bilinear", align_corners=True)
            for zi in scale_depths
        ]

        # Cache per-scale intermediates (eval only) for the multi-level visualization script,
        # identical to DPTHeadMultiScaleRefine (ViT). Detached, GPU-side, overwritten each forward;
        # does NOT change the return value or training behaviour.
        if not self.training:
            self.viz_cache = {
                'multilevel_native': [d.detach() for d in scale_depths],
                'deltas': [dz.detach() for dz in deltas],
                'prev_upsampled': [p.detach() for p in prev_upsampled],
            }

        # Training: return every scale at its NATIVE pyramid resolution (coarse -> fine), matching
        # DPTHeadMultiScaleRefine (ViT). gemdepth._postprocess_depth then keeps them native
        # (multiscale_native_res=True -> real coarse-to-fine, loss downsamples GT per scale) or
        # upsamples every scale to full input resolution (False -> original all-scales-align-to-GT).
        # Eval/inference: return only the finest full-resolution depth.
        if self.training:
            return scale_depths
        return resized_multilevel_depths[-1]
