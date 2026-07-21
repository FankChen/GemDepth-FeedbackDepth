# Minimal LoRA (Low-Rank Adaptation) for fine-tuning a frozen backbone.
#
# A LoRALinear wraps a frozen nn.Linear and adds a trainable low-rank update
# ``ΔW = (alpha/r) * B @ A`` (A: r×in, B: out×r). B is zero-initialised so the adapter
# is an exact no-op at start — the backbone behaves identically until LoRA learns.
#
# Used to fine-tune the frozen DINOv2 encoder without touching its pretrained weights:
# only the small lora_A/lora_B matrices (and the DPT head) receive gradients.

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert r > 0, "LoRA rank r must be > 0"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)          # freeze the pretrained weight/bias

        self.r = r
        self.scaling = alpha / r
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)          # zero-init => ΔW = 0 at start (identity)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + lora * self.scaling


def inject_lora(root: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0,
                targets=('qkv', 'proj')) -> int:
    """Replace every child nn.Linear whose attribute name is in ``targets`` with a LoRALinear.

    Returns the number of layers wrapped. Collects targets first, then swaps, to avoid
    mutating modules while iterating.
    """
    to_replace = []
    for module in root.modules():
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name in targets:
                to_replace.append((module, name, child))
    for module, name, child in to_replace:
        setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return len(to_replace)


def mark_only_lora_trainable(root: nn.Module) -> None:
    """Freeze everything under ``root`` except LoRA adapter parameters."""
    for n, p in root.named_parameters():
        p.requires_grad_('lora_A' in n or 'lora_B' in n)
