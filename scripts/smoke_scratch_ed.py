#!/usr/bin/env python3
"""GPU smoke test for the five scratch encoder+decoder experiments.

This intentionally uses synthetic RGB/depth data so it tests model, loss, bf16,
gradients, optimizer, and memory independently of dataset availability. It loads
the exact experiment config and the real pretrained backbone weights, but never
loads GemDepth weights and never writes a checkpoint.
"""

import argparse
import gc
from pathlib import Path

import torch
from omegaconf import OmegaConf

from loss.videoloss import VideoDepthLoss
from model.factory import build_gemdepth_from_config
from train import compute_aux_depth_loss_disp


ARMS = {
    "ed_dinov2_static": "scratch_ed_dinov2_static",
    "ed_dinov2_temporal": "scratch_ed_dinov2_temporal",
    "ed_dinov2_multiscale": "scratch_ed_dinov2_multiscale",
    "ed_dinov3vits_static": "scratch_ed_dinov3vits_static",
    "ed_dinov3convnext_static": "scratch_ed_dinov3convnext_static",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--batch", type=int, default=1, help="Per-GPU clip batch")
    parser.add_argument("--crop", type=int, default=None, help="Override config crop for staged smoke")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def cfg_value(cfg, path, default=None):
    return OmegaConf.select(cfg, path, default=default)


def build_model(cfg):
    model_cfg = cfg.model
    backbone = cfg_value(cfg, "model.backbone", "dinov2")
    video_path = cfg_value(cfg, "model.video_path", None)
    assert not video_path, f"Smoke refuses GemDepth weights: model.video_path={video_path}"
    assert not bool(cfg_value(cfg, "model.use_gem", True))
    assert not bool(cfg_value(cfg, "model.use_astt", True))

    dinov2_weights = cfg_value(cfg, "model.dinov2_weights", None)
    backbone_weights = cfg_value(cfg, "model.backbone_weights", None)
    if backbone == "dinov2":
        assert dinov2_weights, "Missing official DINOv2 source"
        if not str(dinov2_weights).startswith("timm://"):
            assert Path(dinov2_weights).is_file(), \
                f"Missing official DINOv2 weights: {dinov2_weights}"
    elif backbone_weights:
        assert Path(backbone_weights).is_file(), f"Missing backbone weights: {backbone_weights}"

    model = build_gemdepth_from_config(cfg, load_backbone_pretrained=True)

    model.pretrained.requires_grad_(False)
    if hasattr(model.head, "proj"):
        model.head.proj.requires_grad_(False)
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)

    bad_base = [name for name, param in model.pretrained.named_parameters()
                if param.requires_grad and "lora_A" not in name and "lora_B" not in name]
    assert not bad_base, f"Trainable backbone base parameters: {bad_base[:5]}"
    lora = [(name, param) for name, param in model.pretrained.named_parameters()
            if "lora_A" in name or "lora_B" in name]
    assert lora and all(param.requires_grad for _, param in lora)
    if not bool(cfg_value(cfg, "model.use_temporal", True)):
        assert len(model.head.motion_modules) == 0
    return model


def finite_grad_stats(named_params):
    with_grad = [(name, param.grad) for name, param in named_params if param.grad is not None]
    assert with_grad, "No gradients found"
    assert all(torch.isfinite(grad).all() for _, grad in with_grad), "Non-finite gradient found"
    nonzero = sum(bool(torch.count_nonzero(grad).item()) for _, grad in with_grad)
    return len(with_grad), nonzero


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required for this smoke test"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "config" / f"{ARMS[args.arm]}.yaml")
    crop = args.crop or int(cfg.dataset.train.crop_size)
    frames = int(cfg.dataset.train.seq_len)
    patch_divisor = 14 if cfg_value(cfg, "model.backbone", "dinov2") == "dinov2" else \
        (16 if cfg_value(cfg, "model.backbone", "dinov2") == "dinov3_vitsplus" else 32)
    assert crop % patch_divisor == 0, f"crop={crop} is not divisible by {patch_divisor}"

    print(f"[smoke] arm={args.arm} batch/GPU={args.batch} T={frames} crop={crop} "
          f"bf16=True forward_only={args.forward_only}", flush=True)
    model = build_model(cfg).cuda().train()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW([param for _, param in trainable], lr=1e-4, weight_decay=0.01)

    image = torch.randn(args.batch, frames, 3, crop, crop, device="cuda")
    target = torch.rand(args.batch, frames, crop, crop, device="cuda") * 79.0 + 1.0
    mask = torch.ones_like(target)
    intrinsic = torch.eye(3, device="cuda").repeat(args.batch, 1, 1)
    poses = [torch.eye(4, device="cuda").repeat(args.batch, 1, 1) for _ in range(frames)]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction, pose_enc, extrinsic, _ = model(image)
    assert prediction.shape == target.shape, (prediction.shape, target.shape)
    assert torch.isfinite(prediction).all(), "Non-finite depth prediction"

    loss_dict = VideoDepthLoss(pose_flag=False)(
        prediction, target, mask, intrinsic, poses, pose_enc, extrinsic)
    loss = loss_dict["total_loss"]
    aux_loss = None
    if float(cfg_value(cfg, "training.aux_depth_weight", 0.0)) > 0:
        aux_loss = compute_aux_depth_loss_disp(model.head.aux_depths, target.unsqueeze(2), mask.unsqueeze(2))
        loss = loss + float(cfg.training.aux_depth_weight) * aux_loss
    assert torch.isfinite(loss), f"Non-finite loss: {loss_dict}"

    if not args.forward_only:
        loss.backward()
        lora_stats = finite_grad_stats([
            (name, param) for name, param in model.pretrained.named_parameters()
            if "lora_A" in name or "lora_B" in name
        ])
        head_stats = finite_grad_stats([
            (name, param) for name, param in model.head.named_parameters()
            if param.requires_grad
        ])
        assert all(param.grad is None for name, param in model.pretrained.named_parameters()
                   if "lora_A" not in name and "lora_B" not in name)
        optimizer.step()
        print(f"[smoke] gradients LoRA={lora_stats[0]} tensors/{lora_stats[1]} nonzero "
              f"head={head_stats[0]} tensors/{head_stats[1]} nonzero", flush=True)

    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    aux_text = "none" if aux_loss is None else f"{aux_loss.item():.6f}"
    print(f"[smoke] PASS arm={args.arm} output={tuple(prediction.shape)} "
          f"loss={loss.item():.6f} aux={aux_text} peak_allocated={peak_gib:.2f}GiB", flush=True)

    del optimizer, model, image, target, mask, prediction, loss
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"[smoke] CUDA OOM: {exc}", flush=True)
            print(f"[smoke] peak_allocated={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GiB", flush=True)
        raise
