import os
import sys  
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from model.gemdepth import GemDepth
from dataset.dataset_mix import DepthVideoDataset,safe_collate
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from loss.videoloss import *
from torch.utils.tensorboard import SummaryWriter
import glob
import re


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
        # Use total_step from config to compare — final.pth is considered the latest
        pass
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
    torch.save(ckpt, save_path)
    
    # Also save a lightweight model-only copy for easy eval loading
    if is_final:
        model_only_path = ckpt_dir / 'final_model.pth'
        torch.save(model_sd.state_dict(), model_only_path)
    
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
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    grad_accum = int(OmegaConf.select(cfg, 'training.grad_accum', default=1))
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=grad_accum, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),  kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True))
    dataset_train = DepthVideoDataset(**cfg.dataset.train)
    # Determine effective batch size per GPU
    world_size = accelerator.num_processes
    per_gpu_batch = cfg.dataloader.batch_size // world_size if world_size > 0 else cfg.dataloader.batch_size
    train_loader = DataLoader(dataset=dataset_train,batch_size=per_gpu_batch ,pin_memory=True, shuffle=True, num_workers=int(8), drop_last=True,collate_fn=safe_collate,timeout=3600)
    #load model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    head_type = OmegaConf.select(cfg, 'model.head_type', default='temporal')
    error_modalities = OmegaConf.select(cfg, 'model.error_modalities', default='rgbfeat')
    warp_signal = OmegaConf.select(cfg, 'model.warp_signal', default='rgb')
    model = GemDepth(**model_configs[cfg.encoder], head_type=head_type, error_modalities=error_modalities, warp_signal=warp_signal).to(accelerator.device)
    
    # --- Load pretrained GemDepth weights (stage0) ---
    # If resuming, this will be overwritten by the checkpoint load
    if cfg.model.video_path and Path(cfg.model.video_path).exists():
        print(f"[init] Loading pretrained weights from {cfg.model.video_path}")
        checkpoint = torch.load(cfg.model.video_path, map_location='cpu',weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if accelerator.is_main_process:
            print(f"[init] loaded pretrained: missing={len(missing)} unexpected={len(unexpected)}")
            if len(unexpected) > 0:
                print(f"[init] unexpected keys (first 10): {list(unexpected)[:10]}")
            if head_type not in ('errormap', 'errormap_coattn', 'errormap_refine', 'errormap_single'):
                assert len(missing) == 0 and len(unexpected) == 0, \
                    f"Unexpected mismatch loading baseline weights: missing={missing}, unexpected={unexpected}"
            else:
                allowed = ('depth_head', 'error_encoder', 'modality_encoder', 'coattn', 'fuse_block')
                non_em_missing = [m for m in missing if not any(a in m for a in allowed)]
                assert len(non_em_missing) == 0, f"Unexpected missing keys beyond error-map modules: {non_em_missing}"
    else:
        print(f"[init] WARNING: No pretrained weights found at {cfg.model.video_path}, training from scratch!")
    
    model.pretrained.requires_grad_(False)

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

    # --- Split parameters into 2 groups ---
    dec_blocks_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('spatial_blocks') or  name.startswith('time_blocks') :
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
    weight_decay = cfg.optimizer.weight_decay if hasattr(cfg.optimizer, 'weight_decay') else 0.01

    # Build param groups, skipping empty ones (e.g. when blocks are frozen)
    param_groups = []
    max_lrs = []
    if len(dec_blocks_params) > 0:
        param_groups.append({'params': dec_blocks_params, 'lr': dec_lr})
        max_lrs.append(dec_lr)
    if len(other_params) > 0:
        param_groups.append({'params': other_params, 'lr': other_lr})
        max_lrs.append(other_lr)
    assert len(param_groups) > 0, "No trainable parameters found"

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
    
    invariant_loss_func = VideoDepthLoss(pose_flag = cfg.pose_flag)
    aux_depth_weight = float(OmegaConf.select(cfg, 'training.aux_depth_weight', default=0.0))
    total_step = start_step
    should_keep_training = True
    writer = SummaryWriter(log_dir=cfg.training.log_dir if hasattr(cfg.training, 'log_dir') else "./logs/train") 
    
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
                    depth_pred,pose_enc_list,extrinsic_pred,intrinsic_pred= model(image)
                loss_dict=invariant_loss_func(depth_pred.squeeze(2), depth_gt.squeeze(2),mask.squeeze(2),intrinsic_gt,extrinsic_gt,pose_enc_list,extrinsic_pred) 
                loss=loss_dict['total_loss']
                if aux_depth_weight > 0:
                    head = accelerator.unwrap_model(model).head
                    aux_depths = getattr(head, 'aux_depths', None)
                    if aux_depths:
                        aux_loss = compute_aux_depth_loss(aux_depths, depth_gt, mask)
                        loss = loss + aux_depth_weight * aux_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                lr_scheduler.step()
                total_step += 1
                loss_r = accelerator.reduce(loss.detach(), reduction='mean')
                if accelerator.is_main_process:
                    writer.add_scalar('train/loss', loss_r.item(), total_step)
                    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], total_step)
                    used_memory_MB = torch.cuda.memory_allocated() / 1024 / 1024
                    max_used_memory_MB = torch.cuda.max_memory_allocated() / 1024 / 1024
                    writer.add_scalar('train/memory_MB', used_memory_MB, total_step)
                    writer.add_scalar('train/max_memory_MB', max_used_memory_MB, total_step)

                if total_step % cfg.training.save_freq == 0:
                    save_checkpoint(cfg.training.checkpoint_dir, total_step, model, optimizer, lr_scheduler, accelerator)

                if total_step >= cfg.total_step:
                    should_keep_training = False
                    del loss, loss_dict
                    break

            del loss
            del loss_dict
            torch.cuda.empty_cache()
    
    # Final save
    if accelerator.is_main_process:
        save_checkpoint(cfg.training.checkpoint_dir, total_step, model, optimizer, lr_scheduler, accelerator, is_final=True)

if __name__ == '__main__':
    main()