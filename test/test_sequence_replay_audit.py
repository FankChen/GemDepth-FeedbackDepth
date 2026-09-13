"""No remote GPU or stored user checkpoint required: synthetic failed-run audit."""

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "diagnostics"), str(ROOT / "test")]
import sequence_replay_audit as audit
import stereogru_sequence_experiment as experiment
from test_stereogru_sequence_workflow import completed_sources, snapshot


def test_failed_retraining_audit_replays_same_weights_without_approval(completed_sources, tmp_path, monkeypatch):
    source, repetitions, config, _cfg, _calls = completed_sources
    root = tmp_path / "sequence"
    experiment.prepare(repetitions, root, config)
    original_step = torch.optim.AdamW.step

    def simulate_changed_training_trajectory(optimizer, *args, **kwargs):
        # Fixture ONLY: manufacture an internally consistent, different final
        # state to prove cross-training inequality does NOT imply replay failure.
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.add_(1e-4)
        return result

    with monkeypatch.context() as local:
        local.setattr(torch.optim.AdamW, "step", simulate_changed_training_trajectory)
        with pytest.raises(ValueError, match="C1 final metrics did not replay"):
            experiment.train(root, 0, "B0", torch.device("cpu"))
    before = snapshot((source, repetitions, root))
    report_root = tmp_path / "audit"
    result = audit.audit(root, report_root, 0, torch.device("cpu"))
    assert result["status"] == "audit_complete"
    assert result["cross_training_difference_not_a_replay"]["passed"] is False
    assert result["all_fixed_weight_checks_passed"] and result["all_update_checks_passed"]
    assert result["formal_training_updates"] == 0
    assert result["disposable_diagnostic_updates"] == 6
    assert result["original_files_unchanged"] and not result["method_authorized"]
    assert not result["completion_certificate_written"]
    assert snapshot((source, repetitions, root)) == before
    assert not list(report_root.rglob("*.pth"))
    assert not (root / "seed0/arms/B0/formal/completed.json").exists()
    assert not (root / "seed0/arms/F1").exists()
    with pytest.raises(ValueError, match="not completed"):
        experiment.next_action(root)
    with pytest.raises(ValueError, match="outside"):
        audit.audit(root, root / "bad", 0, torch.device("cpu"))
    assert not (root / "bad").exists()
    with pytest.raises(ValueError, match="outside"):
        audit.audit(root, ROOT / "diagnostics/no-output", 0, torch.device("cpu"))
    assert not (ROOT / "diagnostics/no-output").exists()
    with pytest.raises(FileExistsError):
        audit.audit(root, report_root, 0, torch.device("cpu"))
    assert snapshot((source, repetitions, root)) == before

    context = experiment.verify(root)
    with monkeypatch.context() as local:
        def interrupted(*_args, **_kwargs):
            raise TimeoutError("simulated in-process deadline")
        local.setattr(audit.legacy, "load_cache", interrupted)
        with pytest.raises(TimeoutError):
            audit.audit(root, tmp_path / "interrupted", 0, torch.device("cpu"))
    error_report = audit.controls.read_json(tmp_path / "interrupted/audit.json")
    assert error_report["status"] == "audit_failed"
    assert error_report["source_recheck_passed"] and error_report["original_files_unchanged"]
    assert not error_report["method_authorized"]
    progress = root / "seed0/arms/B0/formal/progress.jsonl"
    original = progress.read_bytes()
    progress.write_bytes(original + b'{}\n')
    with pytest.raises((ValueError, KeyError)):
        audit.validate_failed_baseline(context, 0)
    progress.write_bytes(original)
    failure = root / "seed0/arms/B0/formal/failure.json"
    old_failure = failure.read_bytes()
    failure.write_text('{"status":"failed","next_arm_allowed":false,"error":"unrelated OOM"}')
    with pytest.raises(ValueError, match="known cross-training"):
        audit.validate_failed_baseline(context, 0)
    failure.write_bytes(old_failure)
    assert snapshot((source, repetitions, root)) == before


def test_new_diagnostic_does_not_mutate_frozen_source_inventory():
    inventory = experiment.source_inventory()
    assert all(not name.startswith(("diagnostics/", "results/", "test/")) for name in inventory)
    assert audit.RTOL == 1e-4 and audit.ATOL == 1e-6


def test_tensor_comparison_distinguishes_exact_close_and_wrong():
    a = {"x": torch.ones(2), "unused": None, "counter": torch.tensor(2)}
    assert audit.compare_tensors(a, a)["bitwise_equal"]
    b = {**a, "x": a["x"] + 1e-5}
    result = audit.compare_tensors(a, b)
    assert result["passed"] and not result["bitwise_equal"]
    b["x"] = a["x"] + .01
    assert not audit.compare_tensors(a, b)["passed"]
    assert not audit.compare_tensors(a, {**a, "unused": torch.zeros(1)})["passed"]
    with pytest.raises(FloatingPointError):
        audit.compare_tensors(a, {**a, "x": torch.full((2,), float("nan"))})
    with pytest.raises(ValueError, match="shape/dtype"):
        audit.compare_tensors(a, {**a, "x": torch.ones(3)})


def test_metric_checks_include_clip_rows_not_only_aggregate():
    import copy
    row = {"id": "clip0", "scene": "Scene01", "valid_pixels": 10,
           "index_l1": .2, "absrel": .3, "rmse": 2., "delta1": .7}
    metrics = {"train": {"overall": dict(row), "scenes": {"Scene01": dict(row)}, "clips": [dict(row)]}}
    assert audit.compare_metrics(metrics, metrics)["passed"]
    changed = copy.deepcopy(metrics)
    changed["train"]["clips"][0]["index_l1"] += .01
    result = audit.compare_metrics(changed, metrics)
    assert not result["passed"] and result["differences"][0]["group"] == "clip0"
    changed["train"]["clips"][0]["valid_pixels"] += 1
    with pytest.raises(ValueError, match="support"):
        audit.compare_metrics(changed, metrics)