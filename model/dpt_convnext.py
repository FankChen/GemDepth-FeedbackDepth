# ConvNeXt (hierarchical) DPT head.
#
# ConvNeXt backbones already emit a 4-level NCHW pyramid (strides 4/8/16/32) with
# per-level channel dims (e.g. [96, 192, 384, 768]). Unlike the ViT / DINOv2 path we
# do NOT reshape tokens and do NOT synthesize a pyramid via resize_layers -- the
# native feature maps are fed straight into the shared scratch / refinenet decoder.
#
# Only the input adapter differs from DPTHeadTemporal:
#   * per-level 1x1 projections (each ConvNeXt stage has a distinct channel count)
#   * resize_layers replaced by Identity (the pyramid already exists)
#   * forward skips the token -> NCHW reshape
# Everything downstream (scratch, refinenet, temporal motion modules) is reused.
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dpt_temporal import DPTHeadTemporal


class DPTHeadTemporalConvNeXt(DPTHeadTemporal):
    def __init__(self, in_channels_list, features=256, use_bn=False,
                 out_channels=[256, 512, 1024, 1024], use_clstoken=False,
                 num_frames=32, pe='ape', use_temporal=False, patch_size=4):
        # Build the parent with the first-level dim as a placeholder; scratch /
        # refinenet / motion_modules are reused as-is, projects / resize_layers below.
        super().__init__(in_channels_list[0], features, use_bn, out_channels,
                         use_clstoken, num_frames=num_frames, pe=pe, use_temporal=use_temporal)
        self.head_patch_size = patch_size
        # Per-level 1x1 projections: ConvNeXt stages have distinct channel counts.
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels_list[i], out_channels[i], kernel_size=1, stride=1, padding=0)
            for i in range(len(out_channels))
        ])
        # Native pyramid already at strides 4/8/16/32 -> no synthetic resize.
        self.resize_layers = nn.ModuleList([nn.Identity() for _ in out_channels])

    def forward(self, out_features, patch_h, patch_w, frame_length,
                layer_3_att=None, layer_4_att=None, mode=None):
        mode = False
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

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * self.head_patch_size), int(patch_w * self.head_patch_size)),
                            mode="bilinear", align_corners=True)
        ori_type = out.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.scratch.output_conv2(out.float())
        return out.to(ori_type)
