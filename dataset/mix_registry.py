"""Registry of dataset mixing policies.

A mixing policy answers one question: given how many clips each training source
contributed, how many times should each source's list be repeated? Sampling is
uniform over the concatenated list, so those repetition counts *are* the
sampling distribution.

Adding a policy means writing a function in a ``dataset/mix_*.py`` module and
decorating it with ``@register("<name>")`` -- no existing file changes, and
``dataset_mix.py`` never grows a branch.

Same shape as model/freeze_registry.py, and the same deviation for the same
reason: policies are functions keyed by an explicit name rather than classes
keyed by ``__name__``, because the config speaks policy names
(``mix_policy: uniform``), not class names.
"""

import importlib
import inspect
import pkgutil


MIX_REGISTRY = {}
_DISCOVERED = False


def register(name):
    """Register a mixing policy under an explicit config-facing name."""
    key = str(name)

    def decorator(policy):
        existing = MIX_REGISTRY.get(key)
        if existing is not None and existing is not policy:
            raise ValueError(
                f"Mix policy {key!r} is already registered by "
                f"{existing.__module__}.{existing.__qualname__}")
        MIX_REGISTRY[key] = policy
        return policy

    return decorator


def _discover_policies():
    global _DISCOVERED
    if _DISCOVERED:
        return
    import dataset

    for module_info in pkgutil.iter_modules(dataset.__path__):
        if module_info.name.startswith("mix_") and module_info.name != "mix_registry":
            importlib.import_module(f"dataset.{module_info.name}")
    _DISCOVERED = True


def available_mix_names():
    _discover_policies()
    return tuple(sorted(MIX_REGISTRY))


def get_mix_policy(name):
    _discover_policies()
    # An explicit `mix_policy: null` means "leave the mix alone", same as
    # omitting the key.
    key = "native" if name is None else str(name)
    policy = MIX_REGISTRY.get(key)
    if policy is None:
        raise ValueError(
            f"Unknown dataset.mix_policy={name!r}; options={available_mix_names()}")
    return policy


def apply_mix_policy(name, **context):
    """Run a registered policy, injecting the context arguments it declares.

    Returns ``{label: repetition count}``, one entry per source in ``counts``.
    """
    policy = get_mix_policy(name)
    parameters = inspect.signature(policy).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    arguments = dict(context) if accepts_kwargs else {
        key: value for key, value in context.items() if key in parameters
    }

    ratios = policy(**arguments)
    missing = set(context.get("counts", {})) - set(ratios)
    if missing:
        raise ValueError(
            f"Mix policy {name!r} returned no ratio for {sorted(missing)}")
    return ratios
