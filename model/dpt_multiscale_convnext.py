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
#
# ============================================================================
# 【实验对照表】三个构造参数 fullres_mode / depth_feedback / fp32_head 是本文件全部
# 消融实验的开关，下面把「实验名 -> 具体取值」列出来，配合 forward() 里的详细注释看：
#
#   实验名                      fullres_mode   depth_feedback   fp32_head   备注
#   C (基线多尺度)                'none'         False            False       起点，4 层各自原生分辨率
#   C_fullres / 方法A            'last'         False            False       仅最细层放大到全分辨率
#   E1 (·全分辨率)        'all'          False            False       4 层全放大，输出也全 448
#                                                                            （丢失分辨率 coarse-to-fine）
#   E3 (卷积full/输出native)     'all_native'   False            False       卷积在448算，delta 缩回原生尺寸
#                                                                            （全分辨率卷积 + 保留 c2f）
#   E2a (native+深度反馈)        'none'         True             False       delta_head 额外看到编码后的
#                                                                            当前累积深度，不再盲改
#   E2b (全分辨率+反馈)          'all'          True             False       E1 + E2a 叠加（实测变差）
#   E4 (E3+反馈)                 'all_native'   True             False       E3 + E2a 叠加（新，待验证）
#   E2a_fp32head / E4_..._fp32   同 E2a/E4      同 E2a/E4        True        对照组：delta_head 卷积强制
#                                                                            FP32，对齐 baseline 的
#                                                                            output_conv2 精度保护
#
# 层间 loss 权重（粗多细少/细多粗少/两头多）不是本文件的开关，是 train.py 里读取
# config 顶层 `multiscale_scale_weights` 传给 MultiScaleVideoDepthLoss 的，见该文件。
# ============================================================================
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

        # Resolution at which each scale's delta head runs / outputs (see forward). 用法：由
        # config 里的 model.multiscale_fullres_mode 传入（见 model/factory.py），对应实验见
        # 上面的文件头【实验对照表】：
        #   'none'       -> native pyramid resolution (original behaviour)             == C / E2a
        #   'last'       -> finest scale only: conv & output at full input resolution  == C_fullres(方法A)
        #   'all'        -> every scale: conv & output at full resolution (outputs     == E1 / E2b
        #                   all full-res, 丢失分辨率 coarse-to-fine 监督)
        #   'all_native' -> every scale: conv at full resolution, output resampled     == E3 / E4
        #                   back to native resolution (全分辨率卷积、保留 c2f 监督)
        assert fullres_mode in ('none', 'last', 'all', 'all_native'), fullres_mode
        self.fullres_mode = fullres_mode
        # Depth feedback（是否让 delta_head 看到「当前累积深度」，而不是只看特征盲改）。
        # 由 config 里的 model.multiscale_depth_feedback 传入。True 用于 E2a / E2b / E4；
        # False（默认）用于 C / C_fullres / E1 / E3。
        # 具体做法：把上一层累积的深度 depth_prev（detach 过，不回传梯度）过 Conv(1->32)+ReLU
        # 编码成 32 通道，再 cat 到本层 256 通道特征后面，delta_head 的输入通道从 256 变 288。
        self.depth_feedback = bool(depth_feedback)
        enc_ch = 32 if self.depth_feedback else 0
        if self.depth_feedback:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
            )

        # fp32_head: run each scale's delta head convs in FP32 (autocast disabled), matching the
        # baseline temporal head which computes its final depth regression (output_conv2) in FP32
        # (see model/dpt_convnext.py DPTHeadTemporalConvNeXt.forward, the same autocast(enabled=False)
        # guard). 由 config 里的 model.multiscale_fp32_head 传入，默认 False（除 baseline 外所有
        # 实验都没有这层精度保护）。True 用于对照组 E2a_fp32head / E4_convfull_feedback_fp32，
        # 用来单独检验"delta_head 补上 FP32 精度保护"这一项是否有效。
        # 纯粹是精度保护，不改变数学语义；代价是算力/显存（在 all / all_native 全分辨率模式下最贵，
        # 因为那时卷积本来就是在 448 全分辨率上做的，再套 FP32 更慢）。
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
            # 主要改动和变量控制：
            # Decide the resolution the delta head runs at (conv_size) and the resolution this
            # scale's output depth lives at (out_size), from fullres_mode (实验对应见文件头注释)：
            #   'none'       -> conv & output at native pyramid resolution (original).      [C / E2a]
            #   'last'       -> finest scale only: conv & output at full input resolution.  [C_fullres]
            #   'all'        -> every scale: conv & output at full resolution (all outputs full-res,
            #                   so coarse-to-fine survives only in the FEATURES, not the depth res). [E1 / E2b]
            #   'all_native' -> every scale: conv at full resolution, output resampled back to
            #                   native resolution -> full-res convolution while keeping real
            #                   coarse-to-fine resolution supervision.                      [E3 / E4]
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
            # 这块是给全分辨率，fea上采样到 conv_size，delta_head卷积在全分辨率上算；否则fea保持原生分辨率，delta_head卷积在原生分辨率上算。
            feat_c = feat if feat.shape[2:] == conv_size else \
                F.interpolate(feat, size=conv_size, mode="bilinear", align_corners=True)
            # Running depth brought to this scale's output resolution for the residual add.
            if depth_prev.shape[2:] != out_size:
                depth_prev = F.interpolate(depth_prev, size=out_size,
                                           mode="bilinear", align_corners=True)
            #
            if self.depth_feedback:
                # 【E2a / E2b / E4 专用分支】这就是有关于深度反馈的部分：编码当前累积深度（上一尺度的累计 z）
                # and concat onto the feature. Detached, so the loss at this scale still only trains
                # this scale's delta head (cross-scale gradient truncation preserved), but the head
                # now *sees* the depth it is correcting instead of predicting blind from features alone.
                dfeat = depth_prev.detach()
                if dfeat.shape[2:] != conv_size:
                    dfeat = F.interpolate(dfeat, size=conv_size, mode="bilinear", align_corners=True)
                feat_in = torch.cat([feat_c, self.depth_encoder(dfeat)], dim=1)
            else:
                feat_in = feat_c

            if self.fp32_head:
                # 【E2a_fp32head / E4_..._fp32 对照组专用分支】Match the baseline temporal head:
                # compute the depth regression conv in FP32.
                ori_dtype = feat_in.dtype
                with torch.autocast(device_type="cuda", enabled=False):
                    delta_z = self.delta_heads[i](feat_in.float())  # at conv_size, FP32
                delta_z = delta_z.to(ori_dtype)
            else:
                delta_z = self.delta_heads[i](feat_in)              # at conv_size
            
            #此处就是E3和E1的区别，E3是卷积在全分辨率上算，但是输出还是回到原生分辨率，E1是卷积和输出都是全分辨率
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
