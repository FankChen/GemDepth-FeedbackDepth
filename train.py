import os
import sys  
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.factory import build_gemdepth_from_config
from dataset.dataset_mix import DepthVideoDataset,safe_collate
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from loss.videoloss import VideoDepthLoss, compute_scale_and_shift
from loss.multiscale_videoloss import MultiScaleVideoDepthLoss
from loss.multiscale_video_l1_loss import MultiScaleVideoL1Loss
from torch.utils.tensorboard import SummaryWriter
import glob
import re


# Far-depth cap (metres) for the training supervision mask, keyed by dataset label
# (data['label']). Near-range driving / indoor sets keep the original 80m so their
# baseline behaviour is unchanged; MVS-Synth is an outdoor GTA-V set whose depths
# reach ~200m, where an 80m cap would discard ~all of its valid pixels. Unknown
# labels fall back to 80m.
DATASET_MAX_DEPTH = {
    'vkitti': 80.0,          # VKITTI2 (driving)
    'TartanAir': 80.0,
    'vkitti1': 80.0,         # VKITTI1.3.1 (driving, same range as VKITTI2)
    'mvs_synth': 500.0,      # outdoor GTA-V, real geometry to ~1000m; 500m keeps ~69% valid
    'pointodyssey': 80.0,
    'dynamic_replica': 80.0, # indoor, few metres
}
DATASET_MAX_DEPTH_DEFAULT = 80.0


def tensor_health(tensor):
    """Compact finite/range diagnostics, evaluated only on a failure path."""
    if tensor is None:
        return None
    value = tensor.detach()
    finite = torch.isfinite(value)
    finite_count = int(finite.sum().item())
    summary = {
        'finite': finite_count == value.numel(),
        'nonfinite': value.numel() - finite_count,
    }
    if finite_count:
        finite_values = value.float()[finite]
        summary['min'] = float(finite_values.min().item())
        summary['max'] = float(finite_values.max().item())
    return summary


def find_latest_ckpt(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint in the directory (by step number)."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    # Match checkpoint_{step}.pth
    pattern = re.compile(r'checkpoint_(\d+)\.pth')
    candidates = []
    for f in ckpt_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            candidates.append((int(m.group(1)), str(f)))
    # Also check final.pth
    final_pth = ckpt_dir / 'final.pth'
    if final_pth.exists():
        print(f"[resume] Found final.pth — will resume from completed checkpoint")
        return str(final_pth)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    latest_step, latest_path = candidates[-1]
    print(f"[resume] Found checkpoint_{latest_step}.pth — will resume from step {latest_step}")
    return latest_path


def load_checkpoint(checkpoint_path: str, model, optimizer, scheduler, accelerator):
    """Load full training state from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Unwrap model if it's wrapped in DDP
    model_sd = accelerator.unwrap_model(model)
    model_sd.load_state_dict(ckpt['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    
    step = ckpt.get('total_step', 0)
    print(f"[resume] Loaded checkpoint from step {step}")
    return step


def save_checkpoint(checkpoint_dir: str, step: int, model, optimizer, scheduler, accelerator, is_final=False):
    """Save a full training checkpoint."""
    # Every rank reaches the surrounding barriers, but only the global main rank
    # may write or prune files. Concurrent torch.save calls corrupt checkpoints.
    if not accelerator.is_main_process:
        return

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    if is_final:
        save_path = ckpt_dir / 'final.pth'
    else:
        save_path = ckpt_dir / f'checkpoint_{step}.pth'
    
    model_sd = accelerator.unwrap_model(model)
    ckpt = {
        'model_state_dict': model_sd.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'total_step': step,
    }
    tmp_path = save_path.with_suffix(save_path.suffix + '.tmp')
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, save_path)
    
    # Also save a lightweight model-only copy for easy eval loading
    if is_final:
        model_only_path = ckpt_dir / 'final_model.pth'
        model_only_tmp = model_only_path.with_suffix(model_only_path.suffix + '.tmp')
        torch.save(model_sd.state_dict(), model_only_tmp)
        os.replace(model_only_tmp, model_only_path)
    
    if accelerator.is_main_process:
        print(f"[save] Checkpoint saved to {save_path} (step {step})")
    
    # Clean up old checkpoints: keep only the last 3
    if not is_final:
        pattern = re.compile(r'checkpoint_(\d+)\.pth')
        checkpoints = []
        for f in ckpt_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                checkpoints.append((int(m.group(1)), f))
        checkpoints.sort(key=lambda x: x[0])
        while len(checkpoints) > 3:
            _, old_path = checkpoints.pop(0)
            old_path.unlink(missing_ok=True)
            if accelerator.is_main_process:
                print(f"[save] Removed old checkpoint {old_path.name}")


@hydra.main(version_base=None, config_path='config', config_name='stages/stage1.yaml')
def main(cfg):
    set_seed(cfg.training.seed)
    Path(cfg.training.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    # DPT heads have inherently-unused parameters BY DESIGN, not from a bug:
    #   * refinenet4.resConfUnit1 is skipped (the top fusion block gets a single input),
    #     true for EVERY DPT head incl. the plain temporal one;
    #   * the multiscale refine head additionally never calls output_conv.
    # find_unused_parameters=True + the multiscale head's multi-branch backward
    # double-marks a param ("marked ready twice"). static_graph=True is the robust fix:
    # it records the (static) used/unused set once and handles both the unused params
    # and the multi-branch graph. Enumerating+freezing every inherently-unused module
    # instead is fragile (there are several by design), so we keep static_graph.
    kwargs = DistributedDataParallelKwargs(static_graph=True)
    grad_accum = int(OmegaConf.select(cfg, 'training.grad_accum', default=1))
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=grad_accum, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),  kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True))

    world_size = accelerator.num_processes
    expected_world_size = int(OmegaConf.select(cfg, 'num_gpus', default=world_size))
    assert world_size == expected_world_size, \
        f"Expected {expected_world_size} processes from config, got {world_size}. Launch with NUM_PROC={expected_world_size}."
    global_micro_batch = int(cfg.dataloader.batch_size)
    assert global_micro_batch > 0 and global_micro_batch % world_size == 0, \
        f"dataloader.batch_size={global_micro_batch} must be positive and divisible by world_size={world_size}"
    per_gpu_batch = global_micro_batch // world_size
    effective_clip_batch = global_micro_batch * grad_accum
    if accelerator.is_main_process:
        print(f"[runtime] world_size={world_size} per_gpu_clip_batch={per_gpu_batch} "
              f"global_micro_batch={global_micro_batch} grad_accum={grad_accum} "
              f"effective_clip_batch={effective_clip_batch}")

    dataset_train = DepthVideoDataset(**cfg.dataset.train)
    assert len(dataset_train) >= global_micro_batch, \
        f"Training dataset is too small/empty: len={len(dataset_train)}, global batch={global_micro_batch}"
    num_workers = int(OmegaConf.select(cfg, 'dataloader.num_workers', default=8))
    train_loader = DataLoader(dataset=dataset_train, batch_size=per_gpu_batch,
                              pin_memory=True, shuffle=True, num_workers=num_workers,
                              drop_last=True, collate_fn=safe_collate,
                              timeout=3600 if num_workers > 0 else 0)
    # Load model. The shared factory is also used by inference and smoke tests,
    # preventing structural drift between checkpoint writer and reader.
    use_gem = bool(OmegaConf.select(cfg, 'model.use_gem', default=True))
    use_astt = bool(OmegaConf.select(cfg, 'model.use_astt', default=True))
    use_temporal = bool(OmegaConf.select(cfg, 'model.use_temporal', default=True))
    lora = bool(OmegaConf.select(cfg, 'model.lora', default=False))
    lora_r = int(OmegaConf.select(cfg, 'model.lora_r', default=8))
    lora_alpha = int(OmegaConf.select(cfg, 'model.lora_alpha', default=16))
    lora_dropout = float(OmegaConf.select(cfg, 'model.lora_dropout', default=0.0))
    backbone = OmegaConf.select(
        cfg, 'model.backbone', default='DINOv2Backbone')
    backbone_weights = OmegaConf.select(cfg, 'model.backbone_weights', default=None)
    video_path = OmegaConf.select(cfg, 'model.video_path', default=None)
    require_pretrained_backbone = bool(OmegaConf.select(
        cfg, 'model.require_pretrained_backbone', default=False))
    encoder_decoder_only = bool(OmegaConf.select(
        cfg, 'model.encoder_decoder_only', default=False))

    if encoder_decoder_only:
        assert not use_gem and not use_astt, \
            "encoder_decoder_only requires model.use_gem=false and model.use_astt=false"
        assert not video_path, \
            "encoder_decoder_only scratch experiments must not load a full GemDepth checkpoint"
    if require_pretrained_backbone:
        if backbone_weights and not str(backbone_weights).startswith('timm://'):
            assert Path(backbone_weights).is_file(), \
                f"Configured local backbone checkpoint not found: {backbone_weights}"

    model = build_gemdepth_from_config(cfg, load_backbone_pretrained=True).to(accelerator.device)
    decoder_name = model.decoder_name
    
    # --- Load pretrained GemDepth weights (stage0) ---
    # If resuming, this will be overwritten by the checkpoint load
    if video_path and Path(video_path).exists():
        print(f"[init] Loading pretrained full-model weights from {video_path}")
        checkpoint = torch.load(video_path, map_location='cpu',weights_only=False)
        # load_backbone_only: keep only DINOv2 'pretrained.*' keys so head/GEM/ASTT stay
        # random-init (true from-scratch of everything except the frozen backbone).
        backbone_only = bool(OmegaConf.select(cfg, 'model.load_backbone_only', default=False))
        if backbone_only:
            checkpoint = {k: v for k, v in checkpoint.items() if k.startswith('pretrained.')}
            if accelerator.is_main_process:
                print(f"[init] backbone-only: keeping {len(checkpoint)} DINOv2 'pretrained.*' keys; head/GEM/ASTT random-init")
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if accelerator.is_main_process:
            print(f"[init] loaded pretrained: missing={len(missing)} unexpected={len(unexpected)}")
            if len(unexpected) > 0:
                print(f"[init] unexpected keys (first 10): {list(unexpected)[:10]}")
            # When GEM/ASTT are disabled or LoRA is enabled or a research head is used, the model
            # legitimately differs from the full-GemDepth checkpoint, so extra/missing keys are OK.
            allow_extra = (
                backbone_only or (not use_gem) or (not use_astt) or lora
                or decoder_name != 'DPTHeadTemporal'
            )
            if backbone_only:
                pass  # everything except the backbone is intentionally random-init -> large missing set expected
            elif not allow_extra:
                assert len(missing) == 0 and len(unexpected) == 0, \
                    f"Unexpected mismatch loading baseline weights: missing={missing}, unexpected={unexpected}"
            else:
                # New sub-modules (research heads / LoRA adapters) and disabled GEM/ASTT modules
                # are the only allowed key mismatches.
                new_key_tags = ('depth_heads', 'error_encoders',
                                'layer_depth_heads', 'layer_delta_heads', 'sig_proj',
                                'delta_heads', 'output_conv1_heads')
                non_new_missing = [m for m in missing
                                   if not any(t in m for t in new_key_tags)
                                   and 'lora_A' not in m and 'lora_B' not in m]
                assert len(non_new_missing) == 0, f"Unexpected missing keys beyond new modules: {non_new_missing}"
    elif video_path:
        raise FileNotFoundError(f"Configured model.video_path does not exist: {video_path}")
    else:
        source = backbone_weights or 'backbone default pretrained weights'
        print(f"[init] No GemDepth checkpoint loaded; backbone source={source}; decoder=random-init")
    
    model.pretrained.requires_grad_(False)

    # `head.proj` is a legacy, unused projection list (the active path uses
    # `head.projects`). Keep it for checkpoint compatibility but exclude it from
    # encoder+decoder-only optimization.
    if encoder_decoder_only and hasattr(model.head, 'proj'):
        model.head.proj.requires_grad_(False)

    # --- Optional freeze: restrict trainable params (e.g. only the DPT head) ---
    freeze_mode = OmegaConf.select(cfg, 'training.freeze_mode', default='default')
    if freeze_mode == 'head_only':
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(True)
        if accelerator.is_main_process:
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[freeze] head_only: only DPT head trainable ({n_train/1e6:.2f}M params)")
    elif freeze_mode not in (None, 'default'):
        raise ValueError(f"Unknown training.freeze_mode={freeze_mode}")

    # Keep LoRA adapters trainable even though the DINOv2 backbone weights are frozen
    # (this re-enables them regardless of the freeze policy above).
    if getattr(model, 'lora', False):
        n_lora = 0
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad_(True)
                n_lora += param.numel()
        if accelerator.is_main_process:
            print(f"[lora] LoRA adapters trainable: {n_lora/1e6:.3f}M params")

    # Fail closed on experiment semantics: no frozen-random backbone weights may
    # accidentally enter training, and every requested LoRA adapter must train.
    trainable_backbone_base = [name for name, param in model.pretrained.named_parameters()
                               if param.requires_grad and 'lora_A' not in name and 'lora_B' not in name]
    assert not trainable_backbone_base, \
        f"Backbone base parameters unexpectedly trainable: {trainable_backbone_base[:10]}"
    lora_named = [(name, param) for name, param in model.pretrained.named_parameters()
                  if 'lora_A' in name or 'lora_B' in name]
    if lora:
        assert lora_named and all(param.requires_grad for _, param in lora_named), \
            "LoRA requested but adapters are missing or frozen"
    elif lora_named:
        raise AssertionError("LoRA adapters exist although model.lora=false")
    if encoder_decoder_only:
        assert not hasattr(model, 'spatial_blocks') and not hasattr(model, 'global_blocks'), \
            "GEM/ASTT modules were instantiated in an encoder+decoder-only experiment"
        if not use_temporal:
            assert len(model.head.motion_modules) == 0, \
                "Static experiment unexpectedly contains temporal motion modules"

    if accelerator.is_main_process:
        backbone_total = sum(p.numel() for p in model.pretrained.parameters())
        lora_trainable = sum(p.numel() for _, p in lora_named if p.requires_grad)
        head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        print(f"[params] backbone={backbone_total/1e6:.3f}M (base frozen), "
              f"LoRA={lora_trainable/1e6:.3f}M trainable, "
              f"decoder={head_trainable/1e6:.3f}M trainable")

    # --- Split parameters into groups (dec blocks / lora / other) ---
    dec_blocks_params = []
    lora_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        elif name.startswith('spatial_blocks') or  name.startswith('time_blocks') :
            dec_blocks_params.append(param)
        else:
            other_params.append(param)

    if cfg.optimizer.kind == 'adam':
        optim = torch.optim.Adam
    elif cfg.optimizer.kind == 'adamw':
        optim = torch.optim.AdamW
    else:
        print('Optimizer error')
        sys.exit(0)

    dec_lr = cfg.optimizer.dec_lr if hasattr(cfg.optimizer, 'dec_lr') else 1e-5
    other_lr = cfg.optimizer.other_lr if hasattr(cfg.optimizer, 'other_lr') else 1e-6
    lora_lr = cfg.optimizer.lora_lr if hasattr(cfg.optimizer, 'lora_lr') else 1e-4
    weight_decay = cfg.optimizer.weight_decay if hasattr(cfg.optimizer, 'weight_decay') else 0.01

    # Build param groups, skipping empty ones (e.g. when blocks are frozen)
    param_groups = []
    max_lrs = []
    if len(dec_blocks_params) > 0:
        param_groups.append({'params': dec_blocks_params, 'lr': dec_lr})
        max_lrs.append(dec_lr)
    if len(lora_params) > 0:
        param_groups.append({'params': lora_params, 'lr': lora_lr})
        max_lrs.append(lora_lr)
    if len(other_params) > 0:
        param_groups.append({'params': other_params, 'lr': other_lr})
        max_lrs.append(other_lr)
    assert len(param_groups) > 0, "No trainable parameters found"

    trainable_param_ids = {id(param) for param in model.parameters() if param.requires_grad}
    grouped_params = [param for group in param_groups for param in group['params']]
    grouped_param_ids = [id(param) for param in grouped_params]
    assert len(grouped_param_ids) == len(set(grouped_param_ids)), \
        "A trainable parameter appears in more than one optimizer group"
    assert set(grouped_param_ids) == trainable_param_ids, \
        "Optimizer groups do not exactly cover all trainable parameters"

    optimizer = optim(param_groups, weight_decay=weight_decay)
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Param group {i}: lr = {param_group['lr']}")

    # Scheduler: total_step may be adjusted if resuming
    total_target_steps = cfg.total_step
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
            max_lrs,
            total_target_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    
    train_loader, model, optimizer, lr_scheduler= accelerator.prepare(train_loader, model, optimizer, scheduler)
    model.to(accelerator.device)
    
    # --- Resume from checkpoint if available ---
    start_step = 0
    resume_path = None
    if cfg.get('resume', False):
        # If resume is a string path, use it; otherwise auto-detect
        if isinstance(cfg.resume, str) and cfg.resume:
            resume_path = cfg.resume
        else:
            resume_path = find_latest_ckpt(cfg.training.checkpoint_dir)
    
    if resume_path and Path(resume_path).exists():
        start_step = load_checkpoint(resume_path, model, optimizer, lr_scheduler, accelerator)
        print(f"[resume] Resuming training from step {start_step}")
    else:
        print(f"[resume] No checkpoint found, starting from step 0")
    
    # Camera (pose) loss only makes sense when GEM predicts poses; disable it otherwise.
    pose_flag = bool(cfg.pose_flag) and use_gem
    invariant_loss_func = VideoDepthLoss(pose_flag = pose_flag)
    # Multi-scale loss selectable by config so the L2-vs-native-video-loss ablation runs from one
    # codebase without editing source between runs. multiscale_loss: 'l2' (default) or 'video'.
    multiscale_loss_name = str(OmegaConf.select(cfg, 'multiscale_loss', default='l2')).lower()
    # IGEV/RAFT-style per-scale weighting. multiscale_gamma unset/<=0 -> uniform (unchanged).
    # Set (e.g. 0.8) -> weights (coarse->fine) = [g^(N-1), ..., g^1, g^0], so the finest scale
    # (the one used at eval) gets weight 1 and coarser scales get exponentially less. The loss
    # normalizes these to sum 1, so only the ratios matter.
    multiscale_gamma = OmegaConf.select(cfg, 'multiscale_gamma', default=None)
    multiscale_scale_weights = None
    if multiscale_gamma is not None and float(multiscale_gamma) > 0:
        g = float(multiscale_gamma)
        n_ms_scales = int(OmegaConf.select(cfg, 'multiscale_scales', default=4))
        multiscale_scale_weights = [g ** (n_ms_scales - 1 - i) for i in range(n_ms_scales)]
        print(f"[loss] multiscale_gamma={g} -> scale_weights(coarse->fine)="
              f"{[round(w, 4) for w in multiscale_scale_weights]}")
    # Explicit per-scale weights (coarse->fine) override gamma. Lets the coarse-heavy /
    # fine-heavy / ends-heavy weighting ablation run from config without touching source. The loss
    # normalizes them to sum 1, so only the ratios matter.
    # 【实验对应】config 顶层 multiscale_scale_weights: [w0,w1,w2,w3]（粗->细）。
    #   E2a_wcoarse -> [0.4, 0.3, 0.2, 0.1]  粗多细少：偏重残差链的"地基"
    #   E2a_wfine   -> [0.1, 0.2, 0.3, 0.4]  细多粗少：偏重评测实际用到的最细层
    #   E2a_wends   -> [0.4, 0.1, 0.1, 0.4]  两头多中间少：U 型，粗细都重、中间两层陪跑
    #   其余实验（C/E1/E2a/E2b/E3/E4/...）都不设这个 key -> None -> 均匀 1/4 各占 25%（原始行为）。
    explicit_scale_weights = OmegaConf.select(cfg, 'multiscale_scale_weights', default=None)
    if explicit_scale_weights is not None:
        multiscale_scale_weights = [float(w) for w in explicit_scale_weights]
        print(f"[loss] multiscale_scale_weights(coarse->fine)={multiscale_scale_weights}")
    normalize_scale_weights = bool(OmegaConf.select(
        cfg, 'multiscale_normalize_scale_weights', default=True))
    if multiscale_loss_name in ('video', 'videoloss', 'video_loss', 'multiscale_video'):
        multiscale_loss_func = MultiScaleVideoDepthLoss(pose_flag=pose_flag,
                                                        scale_weights=multiscale_scale_weights,
                                                        normalize_scale_weights=normalize_scale_weights)
        print(f"[loss] multiscale_loss=video -> MultiScaleVideoDepthLoss("
              f"pose_flag={pose_flag}, normalize_scale_weights={normalize_scale_weights})")
    else:
        multiscale_loss_func = MultiScaleVideoL1Loss(scale_weights=multiscale_scale_weights)
        print("[loss] multiscale_loss=l2 -> MultiScaleVideoL1Loss()")
    total_step = start_step
    should_keep_training = True
    writer = SummaryWriter(log_dir=cfg.training.log_dir if hasattr(cfg.training, 'log_dir') else "./logs/train") \
        if accelerator.is_main_process else None
    
    while should_keep_training:
        model.train()
        for data in tqdm(train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            if data is None:
                continue 
            image = data['image']
            depth_gt = data['depth']
            # Per-dataset far-depth cap: a global 80m cap throws away almost all of
            # MVS-Synth's (outdoor) valid pixels, so use each sample's dataset label
            # to pick its cap (see DATASET_MAX_DEPTH). depth_gt is [B,T,1,H,W].
            labels = data['label']
            far_cap = depth_gt.new_tensor(
                [DATASET_MAX_DEPTH.get(l, DATASET_MAX_DEPTH_DEFAULT) for l in labels])
            far_cap = far_cap.view(-1, *([1] * (depth_gt.dim() - 1)))
            mask = ((depth_gt > 1e-3) & (depth_gt <= far_cap)).float()
            intrinsic_gt=data['IntM']
            extrinsic_gt=data['poses'] 
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    depth_pred,pose_enc_list,extrinsic_pred,intrinsic_pred= model(image)
                # The multi-scale refinement head returns a list of per-scale depth
                # predictions -> route to the multi-scale video depth loss. A single
                # tensor -> the standard video depth loss.
                if isinstance(depth_pred, (list, tuple)):
                    preds = [d.squeeze(2) for d in depth_pred]
                    loss_dict = multiscale_loss_func(preds, depth_gt.squeeze(2), mask.squeeze(2),
                                                     intrinsic_gt, extrinsic_gt, pose_enc_list, extrinsic_pred)
                else:
                    loss_dict=invariant_loss_func(depth_pred.squeeze(2), depth_gt.squeeze(2),mask.squeeze(2),intrinsic_gt,extrinsic_gt,pose_enc_list,extrinsic_pred)
                loss=loss_dict['total_loss']
                if total_step % 200 == 0 and accelerator.is_main_process:
                    if isinstance(depth_pred, (list, tuple)):
                        log_items = {
                            key: value
                            for key, value in loss_dict.items()
                            if key.startswith('scale_') or key == 'total_loss'
                        }
                    else:
                        log_items = loss_dict
                    print("[loss] step %d " % total_step + " ".join(
                        f"{key}={value.item():.4f}"
                        for key, value in log_items.items()
                        if torch.is_tensor(value)
                    ))
                aux_loss = None
                if not torch.isfinite(loss).all():
                    tracked = {
                        'combined_loss': loss,
                        'aux_loss': aux_loss,
                    }
                    if isinstance(depth_pred, (list, tuple)):
                        tracked.update({f'depth_pred/{i}': d for i, d in enumerate(depth_pred)})
                    else:
                        tracked['depth_pred'] = depth_pred
                    tracked.update({
                        f'main/{key}': value for key, value in loss_dict.items()
                        if torch.is_tensor(value)
                    })
                    finite_summary = {
                        key: tensor_health(value) for key, value in tracked.items()
                    }
                    raise FloatingPointError(
                        f"Non-finite training loss at step {total_step}, "
                        f"rank {accelerator.process_index}: {finite_summary}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(grad_norm).all():
                        bad_gradients = [
                            name for name, parameter in model.named_parameters()
                            if parameter.grad is not None
                            and not torch.isfinite(parameter.grad).all()
                        ]
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            f"Non-finite gradient norm at step {total_step}, "
                            f"rank {accelerator.process_index}; "
                            f"parameters={bad_gradients[:20]}")
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                lr_scheduler.step()
                total_step += 1
                loss_r = accelerator.reduce(loss.detach(), reduction='mean')
                aux_r = accelerator.reduce(aux_loss.detach(), reduction='mean') \
                    if aux_loss is not None else None
                if accelerator.is_main_process:
                    writer.add_scalar('train/loss', loss_r.item(), total_step)
                    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], total_step)
                    if aux_r is not None:
                        writer.add_scalar('train/aux_loss', aux_r.item(), total_step)
                    used_memory_MB = torch.cuda.memory_allocated() / 1024 / 1024
                    max_used_memory_MB = torch.cuda.max_memory_allocated() / 1024 / 1024
                    writer.add_scalar('train/memory_MB', used_memory_MB, total_step)
                    writer.add_scalar('train/max_memory_MB', max_used_memory_MB, total_step)

                if total_step % cfg.training.save_freq == 0:
                    accelerator.wait_for_everyone()
                    save_checkpoint(cfg.training.checkpoint_dir, total_step, model, optimizer, lr_scheduler, accelerator)
                    accelerator.wait_for_everyone()

                if total_step >= cfg.total_step:
                    should_keep_training = False
                    del loss, loss_dict
                    break

            del loss
            del loss_dict
            torch.cuda.empty_cache()
    
    # Final save: every rank participates in barriers; only main writes.
    accelerator.wait_for_everyone()
    save_checkpoint(cfg.training.checkpoint_dir, total_step, model, optimizer, lr_scheduler, accelerator, is_final=True)
    accelerator.wait_for_everyone()
    if writer is not None:
        writer.close()
    accelerator.end_training()

if __name__ == '__main__':
    main()