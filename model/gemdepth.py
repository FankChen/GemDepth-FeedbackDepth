import torch
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
import math
import inspect
from tqdm import tqdm
import numpy as np
import gc
from model.tools.blocks import Block
from functools import partial
from model.tools.pos_embed import  get_1d_sincos_pos_embed_from_grid,RoPE2D
from model.backbone_registry import build_backbone
from model.decoder_registry import build_decoder, get_decoder_class
from model.feature_adapters import get_adapter
from model.util.transform import Resize, NormalizeImage, PrepareForNet
from model.utils.util import compute_scale_and_shift, get_interpolate_frames
from model.tools.geometry import GlobalRepresentationEncoder,normalize_pose_translations,transform_pose_using_quats_and_trans_2_to_1
from model.tools.pose_enc import pose_encoding_to_extri_intri
from model.tools.camera import CameraHead
from model.vggt.layers.rope import RotaryPositionEmbedding2D,PositionGetter
from model.vggt.layers.block import Block as vggt_Block

# infer settings
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class GemDepth(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        num_heads=32,
        depth1=2,
        depth2=2,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_implementation="flash_attention",
        pos_embed="RoPE100",
        num_register_tokens=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        init_values=0.01,
        decoder='DPTHeadTemporal',
        decoder_kwargs=None,
        use_gem=True,
        use_astt=True,
        use_temporal=True,
        lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        backbone='DINOv2Backbone',
        backbone_kwargs=None,
        backbone_weights=None,
        load_backbone_pretrained=True,
        multiscale_native_res=True,
        multiscale_fullres_mode='none',
        multiscale_depth_feedback=False,
        multiscale_fp32_head=False,
    ):
        super(GemDepth, self).__init__()

        self.use_gem = use_gem
        self.use_astt = use_astt
        self.encoder = encoder
        self.backbone_name = backbone
        self.lora = lora
        # Multi-scale training-list resolution mode.
        #   True  -> keep each scale at its native pyramid resolution; the loss downsamples GT per
        #            scale (real coarse-to-fine). This is the native-resolution fix.
        #   False -> upsample every scale to full input resolution, so all scales fit the same full
        #            GT (reproduces the original "all scales align to GT" behaviour).
        self.multiscale_native_res = multiscale_native_res
        self.multiscale_fullres_mode = multiscale_fullres_mode
        self.multiscale_depth_feedback = multiscale_depth_feedback
        self.multiscale_fp32_head = multiscale_fp32_head

        self.pretrained = build_backbone(
            backbone,
            backbone_kwargs=backbone_kwargs,
            encoder=encoder,
            weights=backbone_weights,
            lora=lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            pretrained=load_backbone_pretrained,
        )
        self.backbone_format = self.pretrained.feature_format
        self.adapter = get_adapter(self.backbone_format)
        self.patch_size = self.pretrained.patch_size
        self.enc_embed_dim = getattr(
            self.pretrained, 'embed_dim', self.pretrained.embed_dims[0])
        # GEM and ASTT attach to the deepest backbone level, so their width is
        # that level's channel count. Deriving it here instead of restating it in
        # every config keeps the two in sync: it reproduces the historical 1024
        # for DINOv2-L and gives ConvNeXt its stride-32 width (768) for free.
        embed_dim = self.pretrained.embed_dims[3]
        print(
            f"[backbone] {backbone} format={self.backbone_format} "
            f"patch_size={self.patch_size} "
            f"embed_dims={self.pretrained.embed_dims}")

        # ---- ASTT spatial RoPE (only used when ASTT is enabled) ----
        self.rope = None
        if pos_embed.startswith("RoPE"):
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(pos_embed[len("RoPE") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + pos_embed)

        self.pos = PositionGetter()   # spatial position generator (shared)

        # ---- GEM: Geometry-Embedding Module (optional) ----
        if self.use_gem:
            self.rope_vggt = RotaryPositionEmbedding2D(frequency=100)
            self.pos_encoder = PositionalEncoding(
                embed_dim,
                dropout=0.,
                max_len=num_frames
            )
            self.global_blocks = nn.ModuleList([
                vggt_Block(
                    dim=embed_dim,
                    num_heads=num_heads//2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope_vggt,
                ) for _ in range(depth1)
            ])
            self.frame_blocks = nn.ModuleList([
                vggt_Block(
                    dim=embed_dim,
                    num_heads=num_heads//2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope_vggt,
                ) for _ in range(depth1)
            ])
            self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
            self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
            self.patch_start_idx = 1 + num_register_tokens
            self.camera_head = CameraHead(dim_in=2*embed_dim)
            self.cam_rot_encoder = GlobalRepresentationEncoder(
                name="cam_rot_quats_encoder", in_chans=4, enc_embed_dim=embed_dim)
            self.cam_trans_encoder = GlobalRepresentationEncoder(
                name="cam_trans_encoder", in_chans=3, enc_embed_dim=embed_dim)
            self.cam_trans_scale_encoder = GlobalRepresentationEncoder(
                name="scale_encoder", in_chans=1, enc_embed_dim=embed_dim)
        else:
            self.patch_start_idx = 0

        # ---- ASTT: Alternating Spatio-Temporal Transformer (optional) ----
        if self.use_astt:
            self.spatial_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    rope=self.rope,
                    attn_implementation=attn_implementation
                ) for _ in range(depth2)
            ])
            self.time_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads*4,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    rope=None,
                    attn_implementation=attn_implementation
                ) for _ in range(depth2)
            ])
            self.dec_norm = norm_layer(embed_dim)
            self.register_buffer(
                "image_idx_emb",
                torch.from_numpy(
                    get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(50))
                ).float(),
                persistent=False,
            )

        decoder_cls = get_decoder_class(decoder)
        self.decoder_name = decoder_cls.__name__
        self.head = build_decoder(
            decoder_cls,
            decoder_kwargs=decoder_kwargs,
            in_channels=self.enc_embed_dim,
            in_channels_list=getattr(self.pretrained, 'embed_dims', None),
            features=features,
            use_bn=use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            num_frames=num_frames,
            pe=pe,
            use_temporal=use_temporal,
            patch_size=self.patch_size,
            fullres_mode=self.multiscale_fullres_mode,
            depth_feedback=self.multiscale_depth_feedback,
            fp32_head=self.multiscale_fp32_head,
        )
        forward_parameters = inspect.signature(self.head.forward).parameters
        self.decoder_requires_geometry_inputs = all(
            name in forward_parameters
            for name in ('images', 'extrinsics', 'intrinsics')
        )
        print(f"[decoder] {self.decoder_name}")

    def _postprocess_depth(self, depth, B, T, H, W):
        """Resize + ReLU a raw head depth map and reshape to (B, T, H, W).

        Accepts either a single tensor or a list/tuple of per-scale depth maps
        (multi-scale refinement head in training mode). Returns the same
        container type (single tensor -> tensor, list -> list).
        """
        def _one(d, resize=True):
            if resize:
                d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=True)
            d = F.relu(d)
            return d.squeeze(1).unflatten(0, (B, T))

        # Multi-scale training list. multiscale_native_res=True (default): keep each scale at its
        # NATIVE resolution (no upsample) so the loss can downsample GT to each scale (real
        # coarse-to-fine). multiscale_native_res=False: upsample every scale to the full input
        # resolution (all scales fit the same full GT = the original behaviour). Single tensor
        # (eval / temporal head): always upsample to the full input resolution.
        if isinstance(depth, (list, tuple)):
            return [_one(d, resize=not self.multiscale_native_res) for d in depth]
        return _one(depth)

    def _pose_input_mask(self, B, T, device):
        """Which frames get GEM's predicted pose injected; the rest get identity.

        This is dropout on the geometry *input* (paper Tab. 9: injecting pose
        into ~50% of the frames scores the same as 100%), so like any dropout it
        must only fire while training. At eval every frame keeps its pose, which
        also makes inference deterministic -- without this guard the same
        checkpoint scores differently on every run, because a random ~57% of the
        frames silently fall back to an identity pose.
        """
        if not self.training:
            return torch.ones(B * T, dtype=torch.bool, device=device)

        # Kept bit-exact with the runs already on record.
        # NOTE: `.repeat(T)` tiles the (B,) draw as [b_0..b_{B-1}] x T, but the
        # axis being masked is (B, T) flattened, i.e. index b*T + t. So the
        # clip-level draws land on the wrong frames and "a whole clip loses its
        # pose" degrades into per-frame noise. The marginal rates (0.9 / 0.95 /
        # 0.5, ~43% of frames injected) are unchanged, which is why it went
        # unnoticed. Correcting it means `repeat_interleave(T)` plus a retrain.
        clip_keeps_geometry = (torch.rand(B, device=device) < 0.9).repeat(T)
        frame_keeps_geometry = (
            (torch.rand(B * T, device=device) < 0.95) & clip_keeps_geometry
        )
        clip_keeps_pose = (torch.rand(B, device=device) < 0.5).repeat(T)
        return clip_keeps_pose & frame_keeps_geometry

    def forward(self, x):
        image_ids=[]
        B, T, C, H, W = x.shape
        input_images = x
        frame_idx = 0
        global_idx = 0
        # Two grids that coincide only for uniform-stride ViTs: the DPT head
        # upsamples from the patch grid, while GEM/ASTT run on the deepest
        # backbone level (stride 32, i.e. 8x coarser, for ConvNeXt).
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        feats, tokens, geometry_hw = self.adapter.encode(
            self.pretrained, x.flatten(0, 1))
        gh, gw = geometry_hw
        BT,L,C=feats[3].shape

        # Spatial positions for GEM/ASTT. Every level shares one grid and only the
        # deepest is consumed, so a single tensor replaces the old per-level list.
        # The +1 offset keeps index 0 free for the camera/register tokens.
        pos = pos_cam = None
        if self.use_gem or self.use_astt:
            pos = self.pos(BT, gh, gw, feats[3].device)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(
                    BT, self.patch_start_idx, 2,
                    device=pos.device, dtype=pos.dtype)
                pos_cam = torch.cat([pos_special, pos], dim=1)

        # ---- GEM: predict camera pose + geometric embeddings (optional) ----
        pose_enc_list = extrinsic = intrinsic = None
        pose_quats_features = pose_trans_features = pose_trans_scale_features = None
        if self.use_gem:
            camera_token = self.slice_expand_and_flatten(self.camera_token, B, T)
            register_token = self.slice_expand_and_flatten(self.register_token, B, T)
            tokens_all = torch.cat([camera_token, register_token, feats[3]], dim=1)
            _,P,_=tokens_all.shape
            tokens_all,frame_idx,frame_intermediates=self._process_frame_attention(tokens_all, B, T, P, C, frame_idx, pos=pos_cam)
            tokens_all,global_idx,global_intermediates=self._process_global_attention(tokens_all, B, T, P, C, global_idx, pos=pos_cam)
            concat_inter = torch.cat([frame_intermediates,global_intermediates], dim=-1)
            with torch.autocast("cuda", enabled=False):
                pose_enc_list = self.camera_head(concat_inter)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc_list[-1], H, W)
                #pose mask
                device =feats[3].device
                dtype = feats[3].dtype
                per_sample_cam_input_mask = self._pose_input_mask(B, T, device)
                # Initialize the pose quats and trans for all views as identity
                pose_quats_across_views = torch.tensor(
                    [0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device
                ).repeat(B*T, 1)  # (q_x, q_y, q_z, q_w)
                pose_trans_across_views = torch.zeros(
                    (B*T, 3), dtype=dtype, device=device
                )
                # pose embedding
                trans = pose_enc_list[-1][..., :3]
                quat = pose_enc_list[-1][..., 3:7]
                trans_0=trans[:,0].unsqueeze(1).repeat(1,T,1)
                quat_0=quat[:,0].unsqueeze(1).repeat(1,T,1)
                trans=trans.flatten(0,1)[per_sample_cam_input_mask]
                quat=quat.flatten(0,1)[per_sample_cam_input_mask]
                trans_0=trans_0.flatten(0,1)[per_sample_cam_input_mask]
                quat_0=quat_0.flatten(0,1)[per_sample_cam_input_mask]
                (quat,trans) = transform_pose_using_quats_and_trans_2_to_1(quat_0,trans_0,quat,trans)   
                pose_quats_across_views[per_sample_cam_input_mask] = (quat.to(dtype=dtype))
                pose_trans_across_views[per_sample_cam_input_mask] = (trans.to(dtype=dtype))  
                pose_quats_features = self.cam_rot_encoder(pose_quats_across_views) # B*T, embed_dim
                pose_trans_across_views=pose_trans_across_views.unflatten(0, (B, T))
                scaled_pose_trans, pose_trans_norm_factors = (
                    normalize_pose_translations(
                        pose_trans_across_views, return_norm_factor=True
                    )
                )
                pose_trans_norm_factors = pose_trans_norm_factors.unsqueeze(-1).repeat(T, 1)
                pose_trans_features = self.cam_trans_encoder(scaled_pose_trans.flatten(0,1)) 
                log_pose_trans_norm_factors_across_views = torch.log(
                    pose_trans_norm_factors + 1e-8
                )
                pose_trans_scale_features = self.cam_trans_scale_encoder(log_pose_trans_norm_factors_across_views)

        #ASTT module
        if self.use_astt:
            #index embbeding
            for b in range(B):
                for i in range(T):
                    image_ids.extend([i] * L)
            image_ids = torch.tensor(image_ids).reshape(B * T, L).to(feats[3].device)
            num_images = (torch.max(image_ids) + 1).cpu().item()
            image_idx_emb = self.image_idx_emb[:num_images]
            image_pos = image_idx_emb[image_ids]
            #这个循环实现的功能是对每一层的特征图进行处理，主要包括以下几个步骤：
            #1. 如果使用GEM模块，将pose_quats_features、pose_trans_features和pose_trans_scale_features加到当前层的特征图上。
            #2. 对当前层的特征图进行归一化处理。
            #3. 将图像位置编码加到当前层的特征图上。
            #4. 将特征图的形状从(B*T, L, C)转换为(B, T*L, C)，以便进行后续的空间和时间块处理。
            #5. 对每一层的特征图进行空间块和时间块的处理，使用spatial_blocks和time_blocks进行交替处理。
            #6. 将特征图的形状从(B, T*L, C)转换回(B*T, L, C)。
            for m in range(3,4):
                if self.use_gem:
                    feats[m]=feats[m]+pose_quats_features.unsqueeze(1)+pose_trans_features.unsqueeze(1)+pose_trans_scale_features.unsqueeze(1)
                feats[m]=self.dec_norm(feats[m])
                feats[m]=feats[m]+image_pos
                feats[m] = rearrange(feats[m], "(b t) l c -> b (t l) c",b=B,t=T,l=L,c=C)
                astt_pos = rearrange(pos, "(b t) l two -> b (t l) two",b=B,t=T,l=L)
                for blk1,blk2 in zip(self.spatial_blocks,self.time_blocks):
                    feats[m] = blk1(feats[m],astt_pos)
                    feats[m] = rearrange(feats[m], "b (t l) c -> (b l) t c",b=B,t=T,l=L,c=C)
                    feats[m] = blk2(feats[m])
                    feats[m] = rearrange(feats[m], "(b l) t c -> b (t l) c",b=B,t=T,l=L,c=C)
                feats[m] = rearrange(feats[m], "b (t l) c -> (b t) l c",b=B,t=T,l=L,c=C)
        # The DPT heads intentionally run in fp32. Explicit conversion is required:
        # disabling autocast alone does not promote bf16 backbone activations.
        head_features = self.adapter.decode(feats, tokens, geometry_hw)

        #dpt_head
        with torch.autocast(device_type=input_images.device.type, enabled=False):
            if self.decoder_requires_geometry_inputs:
                depth = self.head(head_features, patch_h, patch_w, T,
                                  images=input_images, extrinsics=extrinsic, intrinsics=intrinsic)
            else:
                depth = self.head(head_features, patch_h, patch_w,T)
            depth = self._postprocess_depth(depth, B, T, H, W)
        return depth, pose_enc_list, extrinsic, intrinsic
    
    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.contiguous().view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.contiguous().view(B, S, P, 2).view(B * S, P, 2)

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(1):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            # tokens = cp.checkpoint(self.frame_blocks[frame_idx],tokens, pos, use_reentrant=False)
            frame_idx += 1
        intermediates=tokens.view(B, S, P, C)
        return tokens, frame_idx,intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.contiguous().view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.contiguous().view(B, S, P, 2).view(B, S * P, 2)
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(1):
            tokens = self.global_blocks[global_idx](tokens, pos=pos)
            # tokens = cp.checkpoint(self.global_blocks[global_idx],tokens, pos, use_reentrant=False)
            global_idx += 1
        intermediates=tokens.view(B, S, P, C)
        return tokens, global_idx,intermediates    

    def closed_form_inverse_se3(self,se3, R=None, T=None):
        B,N,_,_=se3.shape
        if R is None:
            R = se3[:, :, :3, :3].reshape(-1,3,3)   
        if T is None:
            T = se3[:, :, :3, 3:] .reshape(-1,3,1)  

        # Transpose R
        R_transposed = R.transpose(1, 2) 
        top_right = -torch.bmm(R_transposed, T)
        inverted_matrix = torch.eye(4, 4)[None].repeat(B*N, 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

        inverted_matrix[:, :3, :3] = R_transposed
        inverted_matrix[:, :3, 3:] = top_right
        inverted_matrix = inverted_matrix.reshape(B, N, 4, 4)
        return inverted_matrix
    
    
    def slice_expand_and_flatten(self,token_tensor, B, S):
        """
        Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
        1) Uses the first position (index=0) for the first frame only
        2) Uses the second position (index=1) for all remaining frames (S-1 frames)
        3) Expands both to match batch size B
        4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
        followed by (S-1) second-position tokens
        5) Flattens to (B*S, X, C) for processing

        Returns:
            torch.Tensor: Processed tokens with shape (B*S, X, C)
        """

        # Slice out the "query" tokens => shape (1, 1, ...)
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
        # Slice out the "other" tokens => shape (1, S-1, ...)
        others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
        # Concatenate => shape (B, S, ...)
        combined = torch.cat([query, others], dim=1)
        B,S,X,C=combined.shape
        # Finally flatten => shape (B*S, ...)
        combined = combined.view(B * S, *combined.shape[2:])
        # combined = combined.view(B,S*X,C)
        return combined
    
    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', fp32=False,
                          clip_len=None):
        """Infer inverse depth for a video.

        ``clip_len=None`` preserves GemDepth's original 32-frame overlapping
        inference and cross-window affine alignment. A positive ``clip_len``
        runs independent non-overlapping clips (padding only the final clip),
        which is the exact protocol for scratch models trained with T=1 or T=4.
        """
        frame_height, frame_width = frames[0].shape[:2]
        input_divisor = 32 if self.backbone_format == 'pyramid' else self.patch_size
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = max(input_divisor, round(input_size / input_divisor) * input_divisor)

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=input_divisor,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        frame_list = [frames[i] for i in range(frames.shape[0])]
        org_video_len = len(frame_list)

        if clip_len is not None:
            assert isinstance(clip_len, int) and clip_len > 0, \
                f"clip_len must be a positive integer or None, got {clip_len}"
            depth_list = []
            device_type = torch.device(device).type
            for frame_id in tqdm(range(0, org_video_len, clip_len)):
                actual_len = min(clip_len, org_video_len - frame_id)
                clip_frames = frame_list[frame_id:frame_id + actual_len]
                clip_frames += [clip_frames[-1].copy()] * (clip_len - actual_len)
                cur_list = [
                    torch.from_numpy(transform({
                        'image': frame.astype(np.float32) / 255.0
                    })['image']).unsqueeze(0).unsqueeze(0)
                    for frame in clip_frames
                ]
                cur_input = torch.cat(cur_list, dim=1).to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                        enabled=(not fp32)):
                        depth, _, _, _ = self.forward(cur_input)
                if depth.shape[:2] != (1, clip_len):
                    raise ValueError(
                        f"Scratch clip output shape mismatch at frame {frame_id}: "
                        f"got {tuple(depth.shape)}, expected B,T=(1,{clip_len})")
                if not torch.isfinite(depth).all():
                    bad = int((~torch.isfinite(depth)).sum().item())
                    raise FloatingPointError(
                        f"Non-finite scratch inference output at frame {frame_id}: "
                        f"bad={bad}/{depth.numel()} clip_len={clip_len}")
                depth = F.interpolate(
                    depth.flatten(0, 1).unsqueeze(1).float(),
                    size=(frame_height, frame_width), mode='bilinear', align_corners=True)
                depth_list.extend(depth[i, 0].cpu().numpy() for i in range(actual_len))

            assert len(depth_list) == org_video_len
            return np.stack(depth_list, axis=0), target_fps

        frame_step = INFER_LEN - OVERLAP
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=torch.device(device).type,
                                    dtype=torch.bfloat16, enabled=(not fp32)):
                    depth,_,_,_= self.forward(cur_input) # depth shape: [1, T, H, W]

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

            pre_input = cur_input

        del frame_list
        gc.collect()

        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id+i])

                
                scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                           np.concatenate(ref_align),
                                                           np.concatenate(np.ones_like(ref_align)==1))

                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i]<0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id+i] * scale + shift
                    new_depth[new_depth<0] = 0
                    depth_list_aligned.append(new_depth)

                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id+kf_id] * scale + shift
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth)

        depth_list = depth_list_aligned

        return np.stack(depth_list[:org_video_len], axis=0), target_fps

    def infer_video_geometry(self, frames, target_fps, input_size=518, device='cuda', fp32=False):
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

        depth_list = []
        extrinsic_list = []
        intrinsic_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth,_,extrinsic,intrinsic= self.forward(cur_input) # depth shape: [1, T, H, W]

            input_height, input_width = cur_input.shape[-2:]
            intrinsic = intrinsic.clone()
            intrinsic[..., 0, :] *= frame_width / input_width
            intrinsic[..., 1, :] *= frame_height / input_height
            extrinsic = extrinsic[0].detach().cpu().numpy()
            intrinsic = intrinsic[0].detach().cpu().numpy()

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]
            extrinsic_list += [extrinsic[i] for i in range(extrinsic.shape[0])]
            intrinsic_list += [intrinsic[i] for i in range(intrinsic.shape[0])]

            pre_input = cur_input

        del frame_list
        gc.collect()

        depth_list_aligned = []
        extrinsic_list_aligned = []
        intrinsic_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                extrinsic_list_aligned += extrinsic_list[:INFER_LEN]
                intrinsic_list_aligned += intrinsic_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id+i])

                
                scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                           np.concatenate(ref_align),
                                                           np.concatenate(np.ones_like(ref_align)==1))

                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i]<0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)
                extrinsic_list_aligned[-INTERP_LEN:] = extrinsic_list[frame_id+align_len:frame_id+OVERLAP]
                intrinsic_list_aligned[-INTERP_LEN:] = intrinsic_list[frame_id+align_len:frame_id+OVERLAP]

                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id+i] * scale + shift
                    new_depth[new_depth<0] = 0
                    depth_list_aligned.append(new_depth)
                    extrinsic_list_aligned.append(extrinsic_list[frame_id+i])
                    intrinsic_list_aligned.append(intrinsic_list[frame_id+i])

                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id+kf_id] * scale + shift
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth)

        depth_list = depth_list_aligned
        extrinsic_list = extrinsic_list_aligned
        intrinsic_list = intrinsic_list_aligned

        return (np.stack(depth_list[:org_video_len], axis=0),
                np.stack(extrinsic_list[:org_video_len], axis=0),
                np.stack(intrinsic_list[:org_video_len], axis=0),
                target_fps)
    
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout = 0.,
        max_len = 32
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.dtype)
        return self.dropout(x)