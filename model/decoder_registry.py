import importlib
import inspect
import pkgutil


DECODER_REGISTRY = {}
_DISCOVERED = False


def register(cls):
    """Register a decoder under its Python class name."""
    name = cls.__name__
    existing = DECODER_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Decoder class name {name!r} is already registered by "
            f"{existing.__module__}.{existing.__qualname__}")
    DECODER_REGISTRY[name] = cls
    return cls


def _discover_decoders():
    global _DISCOVERED
    if _DISCOVERED:
        return
    import model

    for module_info in pkgutil.iter_modules(model.__path__):
        if module_info.name.startswith("dpt_"):
            importlib.import_module(f"model.{module_info.name}")
    _DISCOVERED = True


def available_decoder_names():
    _discover_decoders()
    return tuple(sorted(DECODER_REGISTRY))


def get_decoder_class(name):
    _discover_decoders()
    decoder_cls = DECODER_REGISTRY.get(str(name))
    if decoder_cls is None:
        raise ValueError(
            f"Unknown decoder class={name!r}; "
            f"options={available_decoder_names()}")
    return decoder_cls


def build_decoder(decoder_cls, decoder_kwargs=None, **context):
    """Instantiate a registered class by injecting matching constructor args."""
    signature = inspect.signature(decoder_cls.__init__)
    parameters = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
    }
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    decoder_kwargs = dict(decoder_kwargs or {})
    unknown = set(decoder_kwargs) - set(parameters)
    if unknown and not accepts_kwargs:
        raise ValueError(
            f"Unsupported decoder_kwargs for {decoder_cls.__name__}: "
            f"{sorted(unknown)}")

    arguments = {
        name: value
        for name, value in context.items()
        if name in parameters and value is not None
    }
    arguments.update(decoder_kwargs)

    missing = [
        name for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
        and name not in arguments
    ]
    if missing:
        raise ValueError(
            f"Cannot construct decoder {decoder_cls.__name__}; "
            f"missing constructor arguments={missing}")
    return decoder_cls(**arguments)
