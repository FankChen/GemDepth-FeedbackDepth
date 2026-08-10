"""GPU smoke test for the pluggable DINOv3 backbones (ViT-S+ and ConvNeXt-S) + LoRA.

Run on a GPU node:
    $PY test/smoke_backbones.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.backbone_registry import build_backbone

assert torch.cuda.is_available(), "CUDA required"
DEV = torch.device('cuda')
CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoint')

CASES = [
    ('DINOv3ViTSPlusBackbone', f'{CK}/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'),
    ('DINOv3ConvNeXtSmallBackbone', f'{CK}/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth'),
]


def run(name, weights):
    bb = build_backbone(
        name,
        weights=weights,
        lora=True,
        lora_r=8,
        lora_alpha=16,
    ).to(DEV)
    # Freeze base, enable LoRA only (mimic train.py policy).
    bb.requires_grad_(False)
    for n, p in bb.named_parameters():
        if 'lora_A' in n or 'lora_B' in n:
            p.requires_grad_(True)
    bb.train()

    H = W = 224
    x = torch.rand(2, 3, H, W, device=DEV)
    feats = bb(x)
    print(f"[{name}] is_hier={bb.is_hierarchical} embed_dims={bb.embed_dims} "
          f"strides={bb.feat_strides}")
    print(f"[{name}] feats={[tuple(f.shape) for f in feats]}")
    assert len(feats) == 4
    for f, c in zip(feats, bb.embed_dims):
        assert f.shape[1] == c, (f.shape, c)

    loss = sum(f.float().mean() for f in feats)
    loss.backward()
    lora_grad = sum(p.grad.abs().sum().item() for n, p in bb.named_parameters()
                    if ('lora_A' in n or 'lora_B' in n) and p.grad is not None)
    base_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                    for n, p in bb.named_parameters()
                    if 'lora_A' not in n and 'lora_B' not in n)
    print(f"[{name}] lora_grad_sum={lora_grad:.4e} base_has_grad={base_grad}")
    assert lora_grad > 0, "LoRA got no gradient"
    assert not base_grad, "frozen base got gradient"
    print(f"[{name}] OK\n")


if __name__ == '__main__':
    for name, w in CASES:
        run(name, w)
    print("ALL BACKBONE SMOKE TESTS PASSED")
