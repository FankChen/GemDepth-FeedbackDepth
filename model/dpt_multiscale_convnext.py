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
                 num_frames=32, pe='ape', use_temporal=False, patch_size=4):
        super().__init__(in_channels_list, features, use_bn, out_channels,
                         use_clstoken, num_frames=num_frames, pe=pe,
                         use_temporal=use_temporal, patch_size=patch_size)

        # One residual-regression head per pyramid scale (coarse -> fine).
        # No final activation: delta_Z may be positive or negative.
        self.delta_heads = nn.ModuleList([
            self._make_delta_head(features) for _ in range(4)
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
    def _make_delta_head(features):
        return nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
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
            # [Method A] The finest scale is the final output and receives no
            # further refinement from a later scale, yet previously its whole
            # delta head ran only at the native pyramid resolution and the depth
            # was merely bilinearly upsampled afterwards (no convolution ever ran
            # at full input resolution). Upsample this scale's FEATURE to full
            # resolution first, so every conv layer of delta_heads[-1] runs at
            # full resolution -- mirroring the temporal head's output_conv2 pass.
            # Coarser scales are left untouched: they are refined by subsequent
            # iterations, so they do not need a full-resolution pass.
            if i == last_index and feat.shape[2:] != full_size:
                feat = F.interpolate(feat, size=full_size, mode="bilinear", align_corners=True)
            # Upsample the running depth to the current feature resolution.
            if depth_prev.shape[2:] != feat.shape[2:]:
                depth_prev = F.interpolate(depth_prev, size=feat.shape[2:],
                                           mode="bilinear", align_corners=True)

            delta_z = self.delta_heads[i](feat)
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
