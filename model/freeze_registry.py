"""Registry of parameter-freezing policies, mirroring the backbone/decoder ones.

A freeze policy answers a single question: given the assembled model, which
parameters should stay trainable for this experiment? Adding one means creating
a function in a ``model/freeze_*.py`` module and decorating it with
``@register("<name>")`` - no existing file has to change, and ``train.py`` never
grows another branch.

Unlike the backbone and decoder registries, policies are plain functions keyed by
an explicit name rather than classes keyed by ``__name__``: the config already
speaks in policy names (``freeze_mode: gem_frozen``), not class names, and a
policy has no state worth putting in a class.

Policies declare only the context they actually use and are called with matching
arguments injected by name, so a new policy can ask for something new without
touching the call site. They return a short summary string (or ``None``) that the
caller logs on rank 0, which keeps distributed-printing concerns out of the
policy itself.
"""

import importlib
import inspect
import pkgutil


FREEZE_REGISTRY = {}
_DISCOVERED = False


def register(name):
    """Register a freeze policy under an explicit config-facing name."""
    key = str(name)

    def decorator(policy):
        existing = FREEZE_REGISTRY.get(key)
        if existing is not None and existing is not policy:
            raise ValueError(
                f"Freeze policy {key!r} is already registered by "
                f"{existing.__module__}.{existing.__qualname__}")
        FREEZE_REGISTRY[key] = policy
        return policy

    return decorator


def _discover_policies():
    global _DISCOVERED
    if _DISCOVERED:
        return
    import model

    for module_info in pkgutil.iter_modules(model.__path__):
        if module_info.name.startswith("freeze_"):
            importlib.import_module(f"model.{module_info.name}")
    _DISCOVERED = True


def available_freeze_names():
    _discover_policies()
    return tuple(sorted(FREEZE_REGISTRY))


def get_freeze_policy(name):
    _discover_policies()
    # An explicit `freeze_mode: null` in a config means "no freezing", same as
    # omitting the key entirely.
    key = "default" if name is None else str(name)
    policy = FREEZE_REGISTRY.get(key)
    if policy is None:
        raise ValueError(
            f"Unknown training.freeze_mode={name!r}; "
            f"options={available_freeze_names()}")
    return policy


def apply_freeze(name, **context):
    """Run a registered policy, injecting the context arguments it declares.

    Returns the policy's summary string, or ``None`` when there is nothing worth
    logging (the default policy freezes nothing).
    """
    policy = get_freeze_policy(name)
    parameters = inspect.signature(policy).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_kwargs:
        arguments = dict(context)
    else:
        arguments = {
            key: value for key, value in context.items() if key in parameters
        }
        missing = [
            key for key, parameter in parameters.items()
            if key not in arguments
            and parameter.default is inspect.Parameter.empty
            and parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        if missing:
            raise ValueError(
                f"Freeze policy {name!r} requires context {sorted(missing)} "
                f"which the caller did not provide")

    return policy(**arguments)
