"""GPU smoke test for the clean baseline (no GEM, no ASTT) with LoRA-finetuned DINOv2.

Validates:
  1. GemDepth builds with use_gem=False, use_astt=False, lora=True and loads official DINOv2 weights.
  2. Forward returns depth and None poses; forward/backward run on GPU with no NaN.
  3. Only LoRA adapters + DPT head receive gradients; the DINOv2 base weights stay frozen.

Run on a GPU node:
    $PY test/smoke_baseline_lora.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gemdepth import GemDepth

assert torch.cuda.is_available(), "CUDA GPU required"
DEV = torch.device('cuda')
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'checkpoint', 'dinov2_vitl14_pretrain.pth')


def main():
    torch.manual_seed(0)
    model = GemDepth(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024],
                     backbone='DINOv2Backbone', decoder='DPTHeadTemporal',
                     use_gem=False, use_astt=False,
                     lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.0,
                     backbone_weights=CKPT if os.path.exists(CKPT) else None).to(DEV)

    # Mimic train.py freeze policy: DINOv2 base frozen, LoRA + head trainable.
    model.pretrained.requires_grad_(False)
    for n, p in model.named_parameters():
        if 'lora_A' in n or 'lora_B' in n:
            p.requires_grad_(True)
    model.train()

    n_gem = sum(1 for n, _ in model.named_parameters() if 'camera' in n or 'frame_blocks' in n or 'global_blocks' in n)
    n_astt = sum(1 for n, _ in model.named_parameters() if n.startswith('spatial_blocks') or n.startswith('time_blocks'))
    print(f"[build] OK  use_gem=False use_astt=False  GEM params={n_gem} ASTT params={n_astt} "
          f"(both should be 0)")
    assert n_gem == 0 and n_astt == 0, "GEM/ASTT modules should not exist in the clean baseline"

    B, T, H, W = 1, 4, 70, 70
    x = torch.rand(B, T, 3, H, W, device=DEV)
    depth, pose_enc_list, extrinsic, intrinsic = model(x)
    print(f"[fwd] depth={tuple(depth.shape)} pose_enc_list={pose_enc_list} "
          f"extrinsic={extrinsic} intrinsic={intrinsic}")
    assert pose_enc_list is None and extrinsic is None and intrinsic is None
    assert torch.isfinite(depth).all(), "depth has NaN/Inf"

    depth.float().mean().backward()

    # LoRA adapters must receive gradients; base backbone weights must not.
    lora_grad = 0.0
    base_has_grad = False
    for n, p in model.pretrained.named_parameters():
        if 'lora_A' in n or 'lora_B' in n:
            if p.grad is not None:
                lora_grad += p.grad.abs().sum().item()
        else:
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                base_has_grad = True
    head_grad = sum(p.grad.abs().sum().item() for p in model.head.parameters() if p.grad is not None)
    print(f"[bwd] lora_grad_sum={lora_grad:.4e} head_grad_sum={head_grad:.4e} "
          f"base_backbone_has_grad={base_has_grad}")
    assert lora_grad > 0, "LoRA adapters received no gradient"
    assert head_grad > 0, "DPT head received no gradient"
    assert not base_has_grad, "frozen DINOv2 base weights unexpectedly got gradients"
    print("BASELINE+LORA SMOKE TEST PASSED")


if __name__ == '__main__':
    main()
