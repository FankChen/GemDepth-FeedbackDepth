"""Registry for complete training objectives.

An objective owns the full contract between a model prediction and the scalar
loss used by training.  New experiments live in ``loss/objective_*.py`` and
register a builder here; ``train.py`` never grows another experiment branch.

This mirrors the decoder/freeze/mix registries already used by the project.
Builders are functions rather than classes because an objective may compose a
different module for tensor and list predictions while presenting one forward
contract to the training loop.
"""

import importlib
import inspect
import pkgutil


OBJECTIVE_REGISTRY = {}
_DISCOVERED = False


def register(*names):
    """Register an objective builder under one or more config-facing names."""
    keys = tuple(str(name) for name in names)
    if not keys:
        raise ValueError("An objective must have at least one name")

    def decorator(builder):
        for key in keys:
            existing = OBJECTIVE_REGISTRY.get(key)
            if existing is not None and existing is not builder:
                raise ValueError(
                    f"Objective {key!r} is already registered by "
                    f"{existing.__module__}.{existing.__qualname__}")
            OBJECTIVE_REGISTRY[key] = builder
        return builder

    return decorator


def _discover_objectives():
    global _DISCOVERED
    if _DISCOVERED:
        return
    import loss

    for module_info in pkgutil.iter_modules(loss.__path__):
        if (module_info.name.startswith("objective_")
                and module_info.name != "objective_registry"):
            importlib.import_module(f"loss.{module_info.name}")
    _DISCOVERED = True


def available_objective_names():
    _discover_objectives()
    return tuple(sorted(OBJECTIVE_REGISTRY))


def get_objective_builder(name):
    _discover_objectives()
    key = str(name)
    builder = OBJECTIVE_REGISTRY.get(key)
    if builder is None:
        raise ValueError(
            f"Unknown loss.objective={name!r}; "
            f"options={available_objective_names()}")
    return builder


def build_objective(name, objective_kwargs=None, **context):
    """Build a registered objective, injecting only arguments it declares."""
    builder = get_objective_builder(name)
    parameters = inspect.signature(builder).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values())

    objective_kwargs = dict(objective_kwargs or {})
    unknown = set(objective_kwargs) - set(parameters)
    if unknown and not accepts_kwargs:
        raise ValueError(
            f"Unsupported loss.kwargs for objective {name!r}: "
            f"{sorted(unknown)}")

    arguments = dict(context) if accepts_kwargs else {
        key: value for key, value in context.items()
        if key in parameters and value is not None
    }
    arguments.update(objective_kwargs)

    missing = [
        key for key, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
        and key not in arguments
    ]
    if missing:
        raise ValueError(
            f"Cannot construct objective {name!r}; "
            f"missing arguments={missing}")
    return builder(**arguments)
