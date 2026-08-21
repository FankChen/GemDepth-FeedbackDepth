"""Built-in freeze policies. Add a new one here (or in another ``freeze_*``
module) with ``@register("name")`` instead of branching in ``train.py``."""

from model.freeze_registry import register
from model.module_groups import GEM_PREFIXES


def _trainable_millions(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


@register("default")
def default():
    """Train everything the model assembly left trainable."""
    return None


@register("head_only")
def head_only(model):
    """Freeze the whole network except the DPT head."""
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.head.parameters():
        param.requires_grad_(True)
    return f"head_only: only DPT head trainable ({_trainable_millions(model):.2f}M params)"


@register("gem_frozen")
def gem_frozen(model, use_gem):
    """Stage 2 of the paper: freeze GEM, fine-tune everything else.

    GEM keeps running in the forward pass, it just stops learning, so the rest of
    the network adapts to the geometry GEM actually produces. Pair this with
    ``pose_flag=false``: with GEM frozen the camera loss has no trainable
    parameter left to update anyway.
    """
    if not use_gem:
        raise ValueError(
            "training.freeze_mode=gem_frozen requires model.use_gem=true")

    n_frozen = 0
    for name, param in model.named_parameters():
        if name.startswith(GEM_PREFIXES):
            param.requires_grad_(False)
            n_frozen += param.numel()
    if n_frozen == 0:
        raise ValueError("freeze_mode=gem_frozen matched no GEM parameters")

    return (f"gem_frozen: GEM frozen ({n_frozen/1e6:.2f}M params), "
            f"trainable={_trainable_millions(model):.2f}M")
