# ConvNeXt (hierarchical) multi-scale coarse-to-fine depth refinement head.
#
# At every pyramid level the decoder first projects the refined path feature,
# upsamples it by 2, and only then predicts a residual. Each refined path is
# already 2x its corresponding native ConvNeXt stage, so the resulting depth
# is 4x the native stage resolution:
#
#   delta_i = output_conv2_i(upsample_2x(output_conv1_i(path_i)))
#   depth_i = resize(depth_{i-1}).detach() + delta_i
#
# This avoids predicting a low-resolution depth residual and then upsampling
# the depth itself. The input adapter is inherited from
# ``DPTHeadTemporalConvNeXt``:
#   * per-level 1x1 projections (each ConvNeXt stage has a distinct channel count)
#   * resize_layers replaced by Identity (the native NCHW pyramid already exists)
#   * NO token -> NCHW reshape (features arrive as NCHW maps)
#
# This mirrors how ``DPTHeadTemporalConvNeXt`` relates to ``DPTHeadTemporal``, so
# ConvNeXt can run the exact same from-scratch multiscale experiment as the ViT
# backbones (experiment 7). Everything downstream (scratch / refinenet / temporal
# motion modules) is reused unchanged.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dpt_convnext import DPTHeadTemporalConvNeXt
from model.decoder_registry import register


@register
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

        # Kept for configuration/checkpoint API compatibility. Feature-first
        # upsampling now applies to every scale, independently of this legacy
        # experiment switch.
        assert fullres_mode in ('none', 'last', 'all', 'all_native'), fullres_mode
        self.fullres_mode = fullres_mode
        # depth_feedback is retained in the signature for old config
        # compatibility. Feedback is now always enabled: resize and encode the
        # previous depth at the current scale, then concatenate it with the
        # upsampled current-scale feature.
        enc_ch = 32
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, enc_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        # fp32_head runs both parts of every per-scale regression head in FP32.
        self.fp32_head = bool(fp32_head)

        # Per-scale counterpart of:
        # output_conv2(upsample(output_conv1(path_i))).
        # output_conv2 deliberately has no final ReLU because a residual must
        # be able to increase or decrease the running depth.
        head_features = features // 2
        self.output_conv1_heads = nn.ModuleList([
            nn.Conv2d(features, head_features, kernel_size=3, stride=1, padding=1)
            for _ in range(4)
        ])
        self.delta_heads = nn.ModuleList([
            self._make_delta_head(head_features + enc_ch) for _ in range(4)
        ])
        # Centre the initial residuals around zero without zeroing the weights:
        # zero weights would block the first backward pass from reaching the
        # newly added output_conv1 feature projections.
        for delta_head in self.delta_heads:
            nn.init.zeros_(delta_head[-1].bias)

        # Per-scale depth predictions cached on every forward pass so the
        # training loop can apply multi-scale supervision (head.aux_depths).
        self.aux_depths = []

    @staticmethod
    def _make_delta_head(in_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def _predict_delta(self, scale_index, feat, depth_prev, output_size):
        depth_feature = self.output_conv1_heads[scale_index](feat)
        depth_feature = F.interpolate(
            depth_feature, size=output_size,
            mode="bilinear", align_corners=True
        )

        if depth_prev is None:
            last_depth_feature = depth_feature.new_zeros(
                depth_feature.shape[0], 32, *output_size
            )
        else:
            depth_prev = F.interpolate(
                depth_prev.detach(), size=output_size,
                mode="bilinear", align_corners=True
            )
            last_depth_feature = self.depth_encoder(
                depth_prev.to(depth_feature.dtype)
            )
        depth_feature = torch.cat(
            [depth_feature, last_depth_feature], dim=1
        )

        return self.delta_heads[scale_index](depth_feature)

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

        In training mode returns four depth predictions from coarse to fine.
        Each prediction is 4x the corresponding native ConvNeXt stage
        resolution. In eval/inference mode returns the finest prediction.
        """
        mode = False

        paths = self._build_pyramid(out_features, frame_length,
                                    layer_3_att=layer_3_att, layer_4_att=layer_4_att, mode=mode)

        # Full input resolution (patch grid * head patch size), e.g. 288x960.
        full_size = (int(patch_h * self.head_patch_size), int(patch_w * self.head_patch_size))

        depth_prev = None if init_depth is None else init_depth.detach().to(paths[0].dtype)

        scale_depths = []
        for i, feat in enumerate(paths):
            # refinenet has already enlarged each native ConvNeXt stage by 2x;
            # another 2x here gives depth resolutions of 4x the native stages.
            output_size = (
                min(feat.shape[-2] * 2, full_size[0]),
                min(feat.shape[-1] * 2, full_size[1]),
            )
            if self.fp32_head:
                ori_dtype = feat.dtype
                with torch.autocast(device_type=feat.device.type, enabled=False):
                    delta_z = self._predict_delta(
                        i,
                        feat.float(),
                        None if depth_prev is None else depth_prev.float(),
                        output_size,
                    )
                delta_z = delta_z.to(ori_dtype)
            else:
                delta_z = self._predict_delta(
                    i, feat, depth_prev, output_size
                )

            if depth_prev is None:
                depth_cur = delta_z
            else:
                depth_prev_resized = F.interpolate(
                    depth_prev.detach(), size=output_size,
                    mode="bilinear", align_corners=True
                )
                depth_cur = depth_prev_resized + delta_z

            scale_depths.append(depth_cur)
            depth_prev = depth_cur

        if self.training:
            return scale_depths
        return scale_depths[-1]
