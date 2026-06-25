# Co-attention error-map DPT head (方案 C: bidirectional / symmetric multi-modal co-attention).
#
# Unlike the v1 error-map head (which simply *adds* a zero-initialised RGB error encoding into
# the decoder feature), this head fuses one or more error modalities with the decoder feature
# through a SYMMETRIC co-attention block: the decoder "anchor" stream and every error-modality
# stream attend to one another, so the model learns *where* and *how* the cross-frame
# inconsistency of each modality should reshape the refinement feature.
#
# Controlled-experiment design — the ONLY thing that changes between the four arms is the set of
# error modalities fed to the (otherwise identical) co-attention mechanism:
#   * 'rgb'      : pure RGB photometric error
#   * 'feat'     : pure decoder-feature error
#   * 'hog'      : pure HOG (gradient-orientation) error
#   * 'rgbfeat'  : RGB error + feature error, fused together by co-attention
#
# The co-attention output projection is zero-initialised, so at init the head reproduces the
# baseline temporal head exactly — keeping fine-tuning from the pretrained weights stable.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_temporal import DPTHeadTemporal
from .util.warp import signal_error_map, scale_intrinsics
from .util.hog import hog_feature_map


# Preset modality sets -> the four controlled arms.
MODALITY_PRESETS = {
    'rgb': ('rgb',),
    'feat': ('feat',),
    'hog': ('hog',),
    'rgbfeat': ('rgb', 'feat'),
}


class CoAttentionFusion(nn.Module):
    """Symmetric co-attention over {anchor (decoder feature)} ∪ {error-modality streams}.

    Each stream is pooled to a fixed ``grid`` × ``grid`` token map, tagged with a learned
    stream embedding and a shared spatial positional embedding, then all streams are
    concatenated and passed through one pre-norm self-attention + MLP block. Because every
    token attends to every other token, the anchor attends to the errors *and* the errors
    attend to the anchor (and to each other) — i.e. a bidirectional / co-attention fusion.

    The refined anchor tokens are upsampled back to the stage resolution and passed through a
    zero-initialised 1×1 conv, so the injection is identity at init.
    """

    def __init__(self, dim, n_modalities, grid=8, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.grid = grid
        self.n_streams = n_modalities + 1  # +1 for the decoder-feature anchor stream

        self.stream_emb = nn.Parameter(torch.zeros(self.n_streams, 1, dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, grid * grid, dim))

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        nn.init.trunc_normal_(self.stream_emb, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, anchor, mod_maps):
        """anchor: (N,dim,h,w); mod_maps: list of (N,dim,h,w). Returns (N,dim,h,w)."""
        N, dim, h, w = anchor.shape
        g = self.grid

        streams = [anchor] + list(mod_maps)
        toks = []
        for i, s in enumerate(streams):
            t = F.adaptive_avg_pool2d(s, (g, g))             # (N,dim,g,g)
            t = t.flatten(2).transpose(1, 2)                 # (N,g*g,dim)
            t = t + self.pos_emb + self.stream_emb[i]
            toks.append(t)
        seq = torch.cat(toks, dim=1)                         # (N, n_streams*g*g, dim)

        q = self.norm1(seq)
        seq = seq + self.attn(q, q, q, need_weights=False)[0]
        seq = seq + self.mlp(self.norm2(seq))

        anchor_out = self.out_norm(seq[:, : g * g])          # (N,g*g,dim) — anchor stream
        anchor_out = anchor_out.transpose(1, 2).reshape(N, dim, g, g)
        anchor_out = F.interpolate(anchor_out, size=(h, w), mode='bilinear', align_corners=False)
        return self.out_proj(anchor_out)                     # zero-init -> identity at init


class DPTHeadErrorMapCoAttn(DPTHeadTemporal):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        error_modalities='rgbfeat',
        error_stages=('s4', 's3', 's2'),
        warp_offsets=(-1, 1),
        attn_grid=8,
        attn_heads=4,
        hog_bins=9,
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken, num_frames, pe)

        if isinstance(error_modalities, str):
            if error_modalities not in MODALITY_PRESETS:
                raise ValueError(
                    f"Unknown error_modalities preset '{error_modalities}'. "
                    f"Choose from {list(MODALITY_PRESETS)} or pass an explicit tuple.")
            self.modalities = MODALITY_PRESETS[error_modalities]
        else:
            self.modalities = tuple(error_modalities)

        self.error_stages = tuple(error_stages)
        self.warp_offsets = tuple(warp_offsets)
        self.hog_bins = hog_bins
        self.aux_depths = []

        self.depth_heads = nn.ModuleDict()
        self.modality_encoders = nn.ModuleDict()
        self.coattn = nn.ModuleDict()
        for key in self.error_stages:
            self.depth_heads[key] = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.Softplus(),
            )
            # One encoder per modality: [err, valid] (2ch) -> features-dim token map.
            for m in self.modalities:
                self.modality_encoders[f'{key}_{m}'] = nn.Sequential(
                    nn.Conv2d(2, features // 2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(features // 2, features, kernel_size=3, stride=1, padding=1),
                )
            self.coattn[key] = CoAttentionFusion(
                features, n_modalities=len(self.modalities), grid=attn_grid, heads=attn_heads)

    def _modality_signal(self, m, images_hw, path_feat, B, T, h, w):
        """Return the (B,T,C,h,w) per-pixel signal to warp/compare for modality ``m``.

        ``images_hw`` is the RGB image already resized to (h, w).
        """
        if m == 'rgb':
            return images_hw.reshape(B, T, images_hw.shape[1], h, w)
        if m == 'feat':
            return path_feat.reshape(B, T, path_feat.shape[1], h, w).float()
        if m == 'hog':
            imgs = images_hw.reshape(B, T, images_hw.shape[1], h, w)
            return hog_feature_map(imgs, nbins=self.hog_bins)
        raise ValueError(f"Unknown modality '{m}'")

    def _inject(self, key, path_feat, images, extrinsics, intrinsics, B, T):
        """Build each modality's error map and fuse them into ``path_feat`` via co-attention."""
        if key not in self.error_stages:
            return path_feat
        if images is None or extrinsics is None or intrinsics is None:
            return path_feat

        BT, _, h, w = path_feat.shape
        depth_s = self.depth_heads[key](path_feat.float())          # (BT,1,h,w), > 0
        self.aux_depths.append(depth_s.reshape(B, T, 1, h, w))
        depth_bt = depth_s.reshape(B, T, 1, h, w)

        H0, W0 = images.shape[-2:]
        imgs_hw = F.interpolate(images.flatten(0, 1).float(), size=(h, w),
                                mode='bilinear', align_corners=False)
        K = scale_intrinsics(intrinsics.detach().float(), (H0, W0), (h, w))
        ext = extrinsics.detach().float()

        mod_maps = []
        for m in self.modalities:
            sig = self._modality_signal(m, imgs_hw, path_feat, B, T, h, w)
            err, valid = signal_error_map(sig, depth_bt, K, ext, offsets=self.warp_offsets)
            err_in = torch.cat([err, valid], dim=2).reshape(BT, 2, h, w)
            mod_maps.append(self.modality_encoders[f'{key}_{m}'](err_in.to(path_feat.dtype)))

        fused = self.coattn[key](path_feat, mod_maps)
        return path_feat + fused

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
        path_4 = self._inject('s4', path_4, images, extrinsics, intrinsics, B, T)

        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, mode, size=layer_2_rn.shape[2:])
        path_3 = self.motion_modules[3](path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_3 = self._inject('s3', path_3, images, extrinsics, intrinsics, B, T)

        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, mode, size=layer_1_rn.shape[2:])
        path_2 = self._inject('s2', path_2, images, extrinsics, intrinsics, B, T)

        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, mode)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        ori_type = out.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.scratch.output_conv2(out.float())

        return out.to(ori_type)
