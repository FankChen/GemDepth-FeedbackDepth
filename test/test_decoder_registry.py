import sys
import unittest
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.decoder_registry import (
    available_decoder_names,
    build_decoder,
    get_decoder_class,
    register,
)


@register
class RegistryTestDecoder(nn.Module):
    def __init__(self, features, iterations=1):
        super().__init__()
        self.features = features
        self.iterations = iterations


class DecoderRegistryTest(unittest.TestCase):
    def test_registered_class_is_resolved_by_class_name(self):
        self.assertIs(
            get_decoder_class("RegistryTestDecoder"), RegistryTestDecoder)
        self.assertIn("DPTHeadTemporalConvNeXt", available_decoder_names())

    def test_constructor_arguments_are_injected_by_signature(self):
        decoder = build_decoder(
            RegistryTestDecoder,
            decoder_kwargs={"iterations": 4},
            features=128,
            ignored_context="not forwarded",
        )
        self.assertEqual(decoder.features, 128)
        self.assertEqual(decoder.iterations, 4)

    def test_unknown_decoder_kwarg_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported decoder_kwargs"):
            build_decoder(
                RegistryTestDecoder,
                decoder_kwargs={"unknown": True},
                features=128,
            )

    def test_duplicate_class_name_is_rejected(self):
        first = type("RegistryDuplicateDecoder", (nn.Module,), {})
        second = type("RegistryDuplicateDecoder", (nn.Module,), {})
        register(first)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register(second)

    def test_unknown_decoder_lists_options(self):
        with self.assertRaisesRegex(ValueError, "Unknown decoder"):
            get_decoder_class("MissingDecoder")


if __name__ == "__main__":
    unittest.main()
