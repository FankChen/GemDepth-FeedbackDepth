# Registry of modular GemDepth dataset loaders.
#
# To add a dataset: implement a BaseLoader subclass in its own module and append
# an instance to ``LOADERS`` below. ``DepthVideoDataset`` calls ``get_loader`` for
# every configured ``data_dir`` and routes matching ones through the loader.

from dataset.loaders.base import BaseLoader
from dataset.loaders.vkitti1 import VKitti1Loader
from dataset.loaders.mvs_synth import MVSSynthLoader
from dataset.loaders.pointodyssey import PointOdysseyLoader
from dataset.loaders.dynamic_replica import DynamicReplicaLoader

# NOTE: order matters only for disambiguating overlapping substrings; each
# ``matches`` is written to be mutually exclusive, so order is not critical here.
LOADERS = [
    VKitti1Loader(),
    MVSSynthLoader(),
    PointOdysseyLoader(),
    DynamicReplicaLoader(),
]


def get_loader(data_dir):
    """Return the loader whose ``matches`` accepts this directory, else None."""
    for loader in LOADERS:
        if loader.matches(data_dir):
            return loader
    return None


__all__ = ["BaseLoader", "LOADERS", "get_loader"]
