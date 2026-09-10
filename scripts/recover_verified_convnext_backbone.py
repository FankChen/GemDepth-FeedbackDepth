"""Recover an EXACT official base from a PP-DPT export, not a finetuned model.

The original training freezes pretrained.model and stores LoRA separately.
Do not TRUST that policy alone: all 342 native tensors, names, shapes and dtypes
must match a fingerprint computed from the locally verified official export.
No adapter is merged. Any changed/missing/additional base tensor aborts before
writing weights. This is CPU-only and never modifies the source checkpoint.
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import sys

import torch


REFERENCE = {
    "source": "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "source_file_sha256": "296db49dcbd622625befd3fc23318cbbcd98049f4c4b0cc026463de6bcd24952",
    "native_tensor_count": 342,
    "native_state_sha256": "eb323d607420145dc0baa072355d232ffd9e953554e9765f3fc625324b63246c",
    "fingerprint_format": "PP-DPT native ConvNeXt state v1; little-endian bytes",
    "official_unused_export_keys": ["norms.3.weight", "norms.3.bias"],
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_state(state):
    """Hash ALL logical tensor contents; independent of .pth serialization/order."""
    if sys.byteorder != "little":
        raise RuntimeError("Reference fingerprint requires little-endian tensors")
    digest = hashlib.sha256(b"PP-DPT native ConvNeXt state v1\0")
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise TypeError(f"Expected a dense tensor for {name}")
        value = value.detach().cpu().contiguous()
        metadata = json.dumps([name, str(value.dtype), list(value.shape)],
                              separators=(",", ":"), ensure_ascii=True).encode("ascii")
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def extract_base(blob):
    """Strip only PP-DPT/DDP prefixes and known fc1/fc2 LoRA wrapper names."""
    if not isinstance(blob, Mapping):
        raise TypeError("Expected a PP-DPT state dict or model_state_dict export")
    state = blob.get("model_state_dict", blob)
    if not isinstance(state, Mapping):
        raise TypeError("model_state_dict is not a mapping")
    base = {}
    skipped_adapters = 0
    ignored_nonbackbone = 0
    for key, tensor in state.items():
        if not isinstance(key, str):
            raise TypeError("State-dict keys must be strings")
        key = key.removeprefix("module.")
        if not key.startswith("pretrained.model."):
            ignored_nonbackbone += 1
            continue
        name = key.removeprefix("pretrained.model.")
        if name.endswith((".lora_A", ".lora_B")):
            skipped_adapters += 1
            continue
        name = re.sub(r"(\.mlp\.fc[12])\.base\.(weight|bias)$", r"\1.\2", name)
        if ".base." in name:
            raise ValueError(f"Unsupported backbone wrapper: {key}")
        if name in base:
            raise ValueError(f"Duplicate canonical base tensor: {name}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Non-tensor backbone entry: {key}")
        base[name] = tensor
    if not base:
        raise ValueError("No pretrained.model tensors; not the expected PP-DPT checkpoint")
    return base, {"discarded_lora_tensors": skipped_adapters,
                  "ignored_nonbackbone_entries": ignored_nonbackbone}


def verify_base(base):
    digest = fingerprint_state(base)
    if (len(base) != REFERENCE["native_tensor_count"]
            or digest != REFERENCE["native_state_sha256"]):
        raise ValueError(
            "Base does NOT exactly match official ConvNeXt-S; no weights exported. "
            f"tensors={len(base)} (expected {REFERENCE['native_tensor_count']}), "
            f"state_sha256={digest} (expected {REFERENCE['native_state_sha256']}). "
            "Do not bypass verification or merge LoRA.")
    return digest


def recover(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).absolute()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("Output/report already exists; refusing overwrite")
    before = checkpoint.stat()
    source_hash = file_sha256(checkpoint)
    # Restricted unpickler, CPU only. Existing final_model exports are ordinary
    # state dicts; there is no weights_only=False fallback for arbitrary objects.
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    base, selection = extract_base(blob)
    digest = verify_base(base)
    after = checkpoint.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source checkpoint changed during verification; no export")
    recovered = {name: value.detach().cpu().contiguous().clone() for name, value in base.items()}
    del blob, base
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "xb") as handle:
        torch.save({"model": recovered}, handle)
    # Re-read the serialized native state as a final integrity check. Its FILE
    # hash differs from the official export; only the canonical tensor hash is
    # identical. Never claim that this new .pth has the original file hash.
    verify_base(torch.load(output, map_location="cpu", weights_only=True)["model"])
    report = {"status": "VERIFIED_OFFICIAL_BASE", "reference": REFERENCE,
              "source_checkpoint": str(checkpoint), "source_file_sha256": source_hash,
              "output": str(output), "output_file_sha256": file_sha256(output),
              "native_tensor_count": len(recovered), "native_state_sha256": digest,
              "lora_merged": False, "source_modified": False, **selection}
    with open(report_path, "x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    recover(args.checkpoint, args.output)


if __name__ == "__main__":
    main()