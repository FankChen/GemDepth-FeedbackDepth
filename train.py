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
from torch.utils.tensorboard import SummaryWriter
import glob
import re


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


def compute_aux_depth_loss(aux_depths, depth_gt, mask):
    """Masked multi-scale L1 supervision for the error-map head's intermediate depths.

    aux_depths: list of tensors shaped (B,T,1,h,w) or (B*T,1,h,w).
    depth_gt / mask: (B,T,1,H,W).
    """
    gt = depth_gt.flatten(0, 1).float()
    m = mask.flatten(0, 1).float()
    total = gt.new_zeros(())
    for d in aux_depths:
        if d.dim() == 5:
            d = d.flatten(0, 1)
        h, w = d.shape[-2:]
        gt_s = F.interpolate(gt, size=(h, w), mode='nearest')
        m_s = F.interpolate(m, size=(h, w), mode='nearest')
        diff = (d.float() - gt_s).abs() * m_s
        total = total + diff.sum() / m_s.sum().clamp(min=1.0)
    return total / max(len(aux_depths), 1)


def compute_aux_depth_loss_disp(aux_depths, depth_gt, mask):
    """Inverse-depth-space multi-scale aux supervision with per-sample scale+shift alignment.

    Companion to ``compute_aux_depth_loss``. The original does absolute L1 against
    metric depth, which pulls the model output into metric-depth space and conflicts
    with the main SSI loss (which supervises inverse depth = 1/depth). This version
    supervises each scale in the SAME inverse-depth space: GT inverse depth = 1/depth_gt,
    and each scale's prediction is aligned to it with a DETACHED closed-form scale+shift
    before a masked L1, so the aux branch no longer fights the main loss.

    Note: this inverse-depth quantity is loosely called "disparity" in MiDaS-style
    terminology (hence the gt_disp / _disp names), but strictly it is inverse depth,
    not binocular stereo disparity.

    aux_depths: list of tensors shaped (B,T,1,h,w) or (B*T,1,h,w).
    depth_gt / mask: (B,T,1,H,W).
    """
    gt = depth_gt.flatten(0, 1).float()
    m = mask.flatten(0, 1).float()
    gt_disp = torch.zeros_like(gt)
    valid = m > 0.5
    gt_disp[valid] = 1.0 / gt[valid].clamp(min=1e-3)
    total = gt.new_zeros(())
    for d in aux_depths:
        if d.dim() == 5:
            d = d.flatten(0, 1)
        d = d.float()
        h, w = d.shape[-2:]
        gt_s = F.interpolate(gt_disp, size=(h, w), mode='nearest')
        m_s = F.interpolate(m, size=(h, w), mode='nearest')
        # Detached closed-form scale+shift aligning the prediction to GT inverse depth,
        # so only the shape (not absolute scale) is supervised — same idea as the main SSI loss.
        with torch.no_grad():
            scale, shift = compute_scale_and_shift(d.squeeze(1), gt_s.squeeze(1), m_s.squeeze(1))
        d_aligned = scale.view(-1, 1, 1, 1) * d + shift.view(-1, 1, 1, 1)
        diff = (d_aligned - gt_s).abs() * m_s
        total = total + diff.sum() / m_s.sum().clamp(min=1.0)
    return total / max(len(aux_depths), 1)


def compute_metric_depth_loss(metric_depths, depth_gt, mask):
    """Masked multi-scale log-L1 for positive metric depths used by GT-camera warp.

    Log depth preserves metric scale (unlike SSI alignment) while preventing far
    pixels from dominating an absolute-depth L1. GT depth is supervision only;
    it is never passed into the model/error feedback path.
    """
    gt = depth_gt.flatten(0, 1).float()
    valid = mask.flatten(0, 1).float()
    total = gt.new_zeros(())
    for pred in metric_depths:
        if pred.dim() == 5:
            pred = pred.flatten(0, 1)
        h, w = pred.shape[-2:]
        gt_s = F.interpolate(gt, size=(h, w), mode='nearest')
        valid_s = F.interpolate(valid, size=(h, w), mode='nearest')
        log_error = (pred.float().clamp(min=1e-3).log()
                     - gt_s.clamp(min=1e-3).log()).abs() * valid_s
        total = total + log_error.sum() / valid_s.sum().clamp(min=1.0)
    return total / max(len(metric_depths), 1)


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


@hydra.main(version_base=None, config_path='config', config_name='stage1.yaml')
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
    head_type = OmegaConf.select(cfg, 'model.head_type', default='temporal')
    use_gem = bool(OmegaConf.select(cfg, 'model.use_gem', default=True))
    use_astt = bool(OmegaConf.select(cfg, 'model.use_astt', default=True))
    use_temporal = bool(OmegaConf.select(cfg, 'model.use_temporal', default=True))
    lora = bool(OmegaConf.select(cfg, 'model.lora', default=False))
    lora_r = int(OmegaConf.select(cfg, 'model.lora_r', default=8))
    lora_alpha = int(OmegaConf.select(cfg, 'model.lora_alpha', default=16))
    lora_dropout = float(OmegaConf.select(cfg, 'model.lora_dropout', default=0.0))
    dinov2_weights = OmegaConf.select(cfg, 'model.dinov2_weights', default=None)
    backbone = OmegaConf.select(cfg, 'model.backbone', default='dinov2')
    backbone_weights = OmegaConf.select(cfg, 'model.backbone_weights', default=None)
    video_path = OmegaConf.select(cfg, 'model.video_path', default=None)
    require_pretrained_backbone = bool(OmegaConf.select(
        cfg, 'model.require_pretrained_backbone', default=False))
    encoder_decoder_only = bool(OmegaConf.select(
        cfg, 'model.encoder_decoder_only', default=False))

    if head_type == 'multiscale_gt_error':
        assert encoder_decoder_only and not use_gem and not use_astt, \
            "GT-error oracle must keep the encoder+decoder-only protocol (GEM/ASTT off)"
        assert int(cfg.dataset.train.seq_len) > 1, \
            "GT-error temporal warp requires seq_len > 1"
        assert str(OmegaConf.select(cfg, 'training.aux_depth_space', default='')) == 'disparity', \
            "GT-error main multiscale auxiliaries must use inverse-depth/disparity space"
        assert float(OmegaConf.select(cfg, 'training.metric_depth_weight', default=0.0)) > 0, \
            "GT-error warp requires a supervised positive metric-depth branch"

    if encoder_decoder_only:
        assert not use_gem and not use_astt, \
            "encoder_decoder_only requires model.use_gem=false and model.use_astt=false"
        assert not video_path, \
            "encoder_decoder_only scratch experiments must not load a full GemDepth checkpoint"
    if require_pretrained_backbone:
        if backbone == 'dinov2':
            assert dinov2_weights, "A required DINOv2 pretrained source was not configured"
            if not str(dinov2_weights).startswith('timm://'):
                assert Path(dinov2_weights).is_file(), \
                    f"Required official DINOv2 checkpoint not found: {dinov2_weights}"
        elif backbone_weights:
            assert Path(backbone_weights).is_file(), \
                f"Configured local backbone checkpoint not found: {backbone_weights}"
        # DINOv3 + null weights is valid: build_backbone uses timm pretrained=True

    model = build_gemdepth_from_config(cfg, load_backbone_pretrained=True).to(accelerator.device)
    
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
            allow_extra = backbone_only or (not use_gem) or (not use_astt) or lora or head_type in ('errormap', 'perlayer', 'multiscale')
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
                                'delta_heads')
                non_new_missing = [m for m in missing
                                   if not any(t in m for t in new_key_tags)
                                   and 'lora_A' not in m and 'lora_B' not in m]
                assert len(non_new_missing) == 0, f"Unexpected missing keys beyond new modules: {non_new_missing}"
    elif video_path:
        raise FileNotFoundError(f"Configured model.video_path does not exist: {video_path}")
    else:
        source = dinov2_weights if backbone == 'dinov2' else (backbone_weights or 'timm/HuggingFace official weights')
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
    aux_depth_weight = float(OmegaConf.select(cfg, 'training.aux_depth_weight', default=0.0))
    # 'depth' (default) = original absolute metric-depth L1; 'disparity' = SSI-aligned
    # inverse-depth-space aux loss (matches the main loss). The keyword value stays
    # 'disparity' (MiDaS-style alias for inverse depth). Only the fix config sets it.
    aux_depth_space = str(OmegaConf.select(cfg, 'training.aux_depth_space', default='depth'))
    metric_depth_weight = float(OmegaConf.select(
        cfg, 'training.metric_depth_weight', default=0.0))
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
            mask = (depth_gt>0).float()
            intrinsic_gt=data['IntM']
            extrinsic_gt=data['poses'] 
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    if head_type == 'multiscale_gt_error':
                        batch_size, num_frames = image.shape[:2]
                        intrinsic_gt = intrinsic_gt.to(
                            accelerator.device, non_blocking=True)
                        gt_ext_tensor = (torch.stack(extrinsic_gt, dim=1)
                                         if isinstance(extrinsic_gt, (list, tuple))
                                         else extrinsic_gt)
                        gt_ext_tensor = gt_ext_tensor.to(accelerator.device, non_blocking=True)
                        if intrinsic_gt.shape not in (
                                (batch_size, 3, 3),
                                (batch_size, num_frames, 3, 3)):
                            raise ValueError(
                                f"Unexpected GT intrinsics shape {tuple(intrinsic_gt.shape)}")
                        if gt_ext_tensor.shape != (batch_size, num_frames, 4, 4):
                            raise ValueError(
                                f"Unexpected GT extrinsics shape {tuple(gt_ext_tensor.shape)}")
                        depth_pred,pose_enc_list,extrinsic_pred,intrinsic_pred = model(
                            image,
                            gt_intrinsics=intrinsic_gt,
                            gt_extrinsics=gt_ext_tensor)
                    else:
                        depth_pred,pose_enc_list,extrinsic_pred,intrinsic_pred= model(image)
                loss_dict=invariant_loss_func(depth_pred.squeeze(2), depth_gt.squeeze(2),mask.squeeze(2),intrinsic_gt,extrinsic_gt,pose_enc_list,extrinsic_pred) 
                loss=loss_dict['total_loss']
                aux_loss = None
                metric_loss = None
                if aux_depth_weight > 0:
                    head = accelerator.unwrap_model(model).head
                    aux_depths = getattr(head, 'aux_depths', None)
                    if aux_depths:
                        if aux_depth_space == 'disparity':
                            aux_loss = compute_aux_depth_loss_disp(aux_depths, depth_gt, mask)
                        else:
                            aux_loss = compute_aux_depth_loss(aux_depths, depth_gt, mask)
                        loss = loss + aux_depth_weight * aux_loss
                if metric_depth_weight > 0:
                    head = accelerator.unwrap_model(model).head
                    metric_depths = getattr(head, 'metric_depths', None)
                    if not metric_depths:
                        raise RuntimeError(
                            "metric_depth_weight > 0 but head produced no metric_depths")
                    metric_loss = compute_metric_depth_loss(metric_depths, depth_gt, mask)
                    loss = loss + metric_depth_weight * metric_loss
                if not torch.isfinite(loss).all():
                    tracked = {
                        'combined_loss': loss,
                        'depth_pred': depth_pred,
                        'aux_loss': aux_loss,
                        'metric_loss': metric_loss,
                    }
                    tracked.update({
                        f'main/{key}': value for key, value in loss_dict.items()
                        if torch.is_tensor(value)
                    })
                    head = accelerator.unwrap_model(model).head
                    for index, value in enumerate(getattr(head, 'aux_depths', ())):
                        tracked[f'aux_depth/{index}'] = value
                    for index, value in enumerate(getattr(head, 'metric_depths', ())):
                        tracked[f'metric_depth/{index}'] = value
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
                metric_r = accelerator.reduce(metric_loss.detach(), reduction='mean') \
                    if metric_loss is not None else None
                error_stats = {}
                if head_type == 'multiscale_gt_error':
                    head = accelerator.unwrap_model(model).head
                    for stage in ('p2', 'p1'):
                        error_map = head.error_maps[stage]
                        valid_map = head.valid_maps[stage]
                        residual = error_map[:, :, :2]
                        denom = (valid_map.sum() * residual.shape[2]).clamp(min=1.0)
                        residual_mean = (residual * valid_map).sum() / denom
                        valid_fraction = valid_map.mean()
                        error_stats[f'{stage}_residual'] = accelerator.reduce(
                            residual_mean, reduction='mean')
                        error_stats[f'{stage}_valid'] = accelerator.reduce(
                            valid_fraction, reduction='mean')
                if accelerator.is_main_process:
                    writer.add_scalar('train/loss', loss_r.item(), total_step)
                    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], total_step)
                    if aux_r is not None:
                        writer.add_scalar('train/aux_loss', aux_r.item(), total_step)
                    if metric_r is not None:
                        writer.add_scalar('train/metric_depth_loss', metric_r.item(), total_step)
                    for key, value in error_stats.items():
                        writer.add_scalar(f'train/error_{key}', value.item(), total_step)
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