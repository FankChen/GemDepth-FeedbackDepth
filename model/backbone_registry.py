import importlib
import inspect
import pkgutil


BACKBONE_REGISTRY = {}
_DISCOVERED = False


def register(cls):
    """Register a backbone under its Python class name."""
    name = cls.__name__
    existing = BACKBONE_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Backbone class name {name!r} is already registered by "
            f"{existing.__module__}.{existing.__qualname__}")
    BACKBONE_REGISTRY[name] = cls
    return cls


def _discover_backbones():
    global _DISCOVERED
    if _DISCOVERED:
        return
    import model

    for module_info in pkgutil.iter_modules(model.__path__):
        if module_info.name == "backbones" or module_info.name.startswith("backbone_"):
            importlib.import_module(f"model.{module_info.name}")
    _DISCOVERED = True


def available_backbone_names():
    _discover_backbones()
    return tuple(sorted(BACKBONE_REGISTRY))


def get_backbone_class(name):
    _discover_backbones()
    backbone_cls = BACKBONE_REGISTRY.get(str(name))
    if backbone_cls is None:
        raise ValueError(
            f"Unknown backbone class={name!r}; "
            f"options={available_backbone_names()}")
    return backbone_cls


def build_backbone(name, backbone_kwargs=None, **context):
    backbone_cls = get_backbone_class(name)
    signature = inspect.signature(backbone_cls.__init__)
    parameters = {
        key: parameter
        for key, parameter in signature.parameters.items()
        if key != "self"
    }
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    backbone_kwargs = dict(backbone_kwargs or {})
    unknown = set(backbone_kwargs) - set(parameters)
    if unknown and not accepts_kwargs:
        raise ValueError(
            f"Unsupported backbone_kwargs for {backbone_cls.__name__}: "
            f"{sorted(unknown)}")

    arguments = {
        key: value
        for key, value in context.items()
        if key in parameters and value is not None
    }
    arguments.update(backbone_kwargs)

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
            f"Cannot construct backbone {backbone_cls.__name__}; "
            f"missing constructor arguments={missing}")
    return backbone_cls(**arguments)
