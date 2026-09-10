"""Recovery is allowed only for an exact reference base, not a trained substitute."""

import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recover_verified_convnext_backbone as recovery


def fixture_state():
    base = {"stem.0.weight": torch.arange(12.).reshape(3, 4),
            "stages.0.blocks.0.mlp.fc1.weight": torch.arange(8.).reshape(2, 4),
            "stages.0.blocks.0.mlp.fc1.bias": torch.tensor([1., 2.])}
    wrapped = {"module.pretrained.model.stem.0.weight": base["stem.0.weight"].clone(),
               "module.pretrained.model.stages.0.blocks.0.mlp.fc1.base.weight": base["stages.0.blocks.0.mlp.fc1.weight"].clone(),
               "module.pretrained.model.stages.0.blocks.0.mlp.fc1.base.bias": base["stages.0.blocks.0.mlp.fc1.bias"].clone(),
               "module.pretrained.model.stages.0.blocks.0.mlp.fc1.lora_A": torch.full((1, 4), 100.),
               "module.pretrained.model.stages.0.blocks.0.mlp.fc1.lora_B": torch.full((2, 1), 100.),
               "module.head.weight": torch.randn(2, 3), "module.camera_head.weight": torch.randn(2, 2)}
    return base, wrapped


def patch_reference(monkeypatch, base):
    # Production CLI has no flag to supply a different reference or skip checks.
    monkeypatch.setitem(recovery.REFERENCE, "native_tensor_count", len(base))
    monkeypatch.setitem(recovery.REFERENCE, "native_state_sha256", recovery.fingerprint_state(base))


def test_extracts_separate_base_without_merging_nonzero_lora():
    base, wrapped = fixture_state()
    result, info = recovery.extract_base({"model_state_dict": wrapped})
    assert set(result) == set(base)
    assert recovery.fingerprint_state(result) == recovery.fingerprint_state(base)
    assert info == {"discarded_lora_tensors": 2, "ignored_nonbackbone_entries": 2}
    assert recovery.fingerprint_state(dict(reversed(list(result.items())))) == recovery.fingerprint_state(base)


@pytest.mark.parametrize("change", ["value", "dtype", "shape", "missing", "extra", "name"])
def test_verification_rejects_any_base_change(monkeypatch, change):
    base, _ = fixture_state()
    patch_reference(monkeypatch, base)
    changed = copy.deepcopy(base)
    key = "stem.0.weight"
    if change == "value":
        changed[key][0, 0] += .0001
    elif change == "dtype":
        changed[key] = changed[key].half()
    elif change == "shape":
        changed[key] = changed[key].reshape(4, 3)
    elif change == "missing":
        del changed[key]
    elif change == "extra":
        changed["unexpected.weight"] = torch.zeros(1)
    elif change == "name":
        changed["renamed.weight"] = changed.pop(key)
    with pytest.raises(ValueError, match="does NOT exactly match"):
        recovery.verify_base(changed)


def test_duplicate_and_unknown_wrapper_fail_closed():
    base, wrapped = fixture_state()
    wrapped["pretrained.model.stem.0.weight"] = base["stem.0.weight"]
    with pytest.raises(ValueError, match="Duplicate"):
        recovery.extract_base(wrapped)
    with pytest.raises(ValueError, match="Unsupported backbone wrapper"):
        recovery.extract_base({"pretrained.model.unknown.base.weight": torch.ones(1)})
    with pytest.raises(ValueError, match="No pretrained.model"):
        recovery.extract_base({"head.weight": torch.ones(1)})


def test_recovery_writes_verified_native_export_without_source_mutation(tmp_path, monkeypatch):
    base, wrapped = fixture_state()
    patch_reference(monkeypatch, base)
    checkpoint, output = tmp_path / "source.pth", tmp_path / "new/native.pth"
    torch.save(wrapped, checkpoint)
    before = checkpoint.read_bytes()
    result = recovery.recover(checkpoint, output)
    exported = torch.load(output, weights_only=True)["model"]
    assert recovery.fingerprint_state(exported) == recovery.fingerprint_state(base)
    assert result["status"] == "VERIFIED_OFFICIAL_BASE" and result["lora_merged"] is False
    assert checkpoint.read_bytes() == before
    report = json.loads(output.with_suffix(".pth.json").read_text())
    assert report["native_state_sha256"] == recovery.fingerprint_state(base)
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        recovery.recover(checkpoint, output)


def test_failed_verification_never_creates_output(tmp_path, monkeypatch):
    base, wrapped = fixture_state()
    patch_reference(monkeypatch, base)
    wrapped["module.pretrained.model.stem.0.weight"][0, 0] = 999.
    checkpoint, output = tmp_path / "source.pth", tmp_path / "new/native.pth"
    torch.save(wrapped, checkpoint)
    with pytest.raises(ValueError, match="does NOT exactly match"):
        recovery.recover(checkpoint, output)
    assert not output.exists() and not output.parent.exists()