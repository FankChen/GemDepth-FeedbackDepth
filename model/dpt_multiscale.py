# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_temporal import DPTHeadTemporal

class DPTHeadMultiScaleRefine(DPTHeadTemporal):
    """Coarse-to-fine multi-scale depth refinement head.

    Built on top of :class:`DPTHeadTemporal`. The top-down feature pyramid
    produces four features of increasing resolution
    (``path_4 -> path_3 -> path_2 -> path_1``, coarse -> fine). Each scale is
    attached to its own regression head that predicts a depth residual
    ``delta_Z``. Starting from an initial depth map, the depth is refined
    scale by scale::

        depth_0 = init_depth (resized to the coarsest scale)
        depth_i = upsample(depth_{i-1}).detach() + delta_Z_i

    The ``.detach()`` cuts the gradient between scales, so the loss computed at
    each scale only trains that scale's ``delta_Z`` head (and the features that
    feed it), never the depth accumulated from previous scales.
    """

    def __init__(self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        use_temporal=True,
        patch_size=14,
    ):
        super().__init__(in_channels, features, use_bn, out_channels,
                         use_clstoken, num_frames, pe, use_temporal, patch_size)

        # One residual-regression head per pyramid scale (coarse -> fine).
        # No final activation: delta_Z may be positive or negative.
        self.delta_heads = nn.ModuleList([
            self._make_delta_head(features) for _ in range(4)
        ])

        # Per-scale depth predictions cached on every forward pass so the
        # training loop can apply multi-scale supervision (see train.py's
        # ``compute_aux_depth_loss`` / ``head.aux_depths``).
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

    def _build_pyramid(self, out_features, patch_h, patch_w, frame_length,
                       layer_3_att=None, layer_4_att=None, mode=False):
        """Run the (temporal) DPT pyramid and return the 4 scale features
        [path_4, path_3, path_2, path_1] (coarse -> fine)."""
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
        """
        Args:
            out_features: list of ViT token features (same as DPTHeadTemporal).
            patch_h, patch_w: patch grid size.
            frame_length: number of frames T (batch is B*T).
            init_depth: initial depth map [B*T, 1, H0, W0]. If None, starts from
                zeros at the coarsest scale.

        Returns:
            final_depth: refined depth at full input resolution.

        Side effects:
            self.aux_depths is set to the list of per-scale depth predictions
            (coarse -> fine), each reshaped to [B, T, 1, h, w], for multi-scale
            supervision by the training loop.
        """
        mode = False

        paths = self._build_pyramid(out_features, patch_h, patch_w, frame_length,
                                    layer_3_att=layer_3_att, layer_4_att=layer_4_att, mode=mode)

        B, T = paths[0].shape[0] // frame_length, frame_length

        # Initialise depth at the coarsest scale. init_depth is detached so the
        # loss gradient never flows back into whatever produced it.
        coarse_size = paths[0].shape[2:]
        if init_depth is None:
            depth_prev = paths[0].new_zeros((paths[0].shape[0], 1, *coarse_size))
        else:
            depth_prev = F.interpolate(init_depth.detach().to(paths[0].dtype), size=coarse_size,
                                       mode="bilinear", align_corners=True)

        scale_depths = []
        self.aux_depths = []
        for i, feat in enumerate(paths):
            # Upsample the running depth to the current feature resolution.
            if depth_prev.shape[2:] != feat.shape[2:]:
                depth_prev = F.interpolate(depth_prev, size=feat.shape[2:],
                                           mode="bilinear", align_corners=True)

            delta_z = self.delta_heads[i](feat)
            # Gradient truncation across scales: the running depth is detached,
            # so the loss at this scale only trains this scale's delta_Z.
            depth_cur = depth_prev.detach() + delta_z

            scale_depths.append(depth_cur)
            h, w = depth_cur.shape[-2:]
            self.aux_depths.append(depth_cur.reshape(B, T, 1, h, w))
            depth_prev = depth_cur

        final_depth = F.interpolate(scale_depths[-1],
                                    (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
                                    mode="bilinear", align_corners=True)

        return final_depth
