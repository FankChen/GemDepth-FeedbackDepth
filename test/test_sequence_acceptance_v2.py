"""Explicit amendment: reuse once, do not edit old evidence, then real arm loops."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "diagnostics"), str(ROOT / "experiments"), str(ROOT / "test")]
import sequence_acceptance_v2 as v2
import sequence_replay_audit as audit
import stereogru_sequence_experiment as old
from test_stereogru_sequence_workflow import completed_sources, snapshot


def test_full_amended_workflow_and_failed_evidence_remains_immutable(completed_sources, tmp_path, monkeypatch):
    source, repetitions, config, cfg, _calls = completed_sources
    failed = tmp_path / "failed_sequence"
    old.prepare(repetitions, failed, config)
    original_step = torch.optim.AdamW.step

    def changed_trajectory(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            for group in optimizer.param_groups:
                for p in group["params"]:
                    p.add_(1e-4)
        return result

    with monkeypatch.context() as local:
        local.setattr(torch.optim.AdamW, "step", changed_trajectory)
        with pytest.raises(ValueError, match="C1 final metrics did not replay"):
            old.train(failed, 0, "B0", torch.device("cpu"))
    report_root = tmp_path / "audit"
    original_probe = audit.update_probe

    def synthetic_failed_update(*args, **kwargs):
        result = original_probe(*args, **kwargs)
        # CPU fixture is deterministic. Explicitly manufacture the recorded
        # updated-state failure ONLY in synthetic evidence to test its handling.
        for comparison in result["comparisons"].values():
            check = comparison["checks"]["updated_state"]
            check.update(passed=False, bitwise_equal=False, max_abs_diff=2e-5,
                         differences=[{"name": "matcher.0.weight", "max_abs_diff": 2e-5}])
            comparison["passed"] = False
        return result

    with monkeypatch.context() as local:
        local.setattr(audit, "update_probe", synthetic_failed_update)
        audit.audit(failed, report_root, 0, torch.device("cpu"))
    roots = (source, repetitions, failed, report_root)
    before = snapshot(roots)
    source_context = old.verify(failed)
    audit_file = report_root / "audit.json"
    contents = audit_file.read_bytes()
    original = v2.c.read_json(audit_file)
    for defect in ("updated_flag", "forward", "summary", "runtime", "updates", "state", "files"):
        changed = copy.deepcopy(original)
        if defect == "updated_flag":
            changed["update_probes"][0]["comparisons"]["old_vs_old_repeatability"]["checks"]["updated_state"]["passed"] = True
        elif defect == "forward":
            changed["update_probes"][0]["comparisons"]["new_vs_old_equivalence"]["checks"]["gradients"]["passed"] = False
        elif defect == "summary":
            changed["all_update_checks_passed"] = True
        elif defect == "runtime":
            changed["numerical_policy"]["cudnn_allow_tf32"] = True
        elif defect == "updates":
            changed["update_probes"][0]["checkpoint_saved"] = True
        elif defect == "state":
            changed["retained_b0_state_sha256_observed_now"] = "foreign"
        else:
            changed["observed_failed_run_files"] = {}
        audit_file.write_text(__import__("json").dumps(changed))
        with pytest.raises(ValueError):
            v2.verify_audit(source_context, report_root)
        audit_file.write_bytes(contents)
    assert snapshot(roots) == before

    revised = tmp_path / "v2"
    v2.prepare(failed, report_root, revised)
    assert snapshot(roots) == before
    context = v2.verify(revised)
    reused = v2.check_completed(context, 0, "B0")
    assert reused["new_optimizer_updates"] == 0 and reused["steps"] == cfg.steps
    assert not (failed / "seed0/arms/B0/formal/completed.json").exists()
    assert not (revised / "seed0/arms/B0/formal/final_head.pth").exists()
    assert v2.next_action(context) == "formal B0 1"
    with pytest.raises(ValueError, match="retained"):
        v2.run(revised, 0, "B0", "formal", torch.device("cpu"))
    with pytest.raises(ValueError, match="not completed"):
        v2.run(revised, 0, "F1", "gate", torch.device("cpu"))
    assert not (revised / "seed0/arms/F1").exists()
    for seed in (1, 2):
        done = v2.run(revised, seed, "B0", "formal", torch.device("cpu"))
        assert done["policy"] == v2.POLICY
    for arm in ("F1", "S1"):
        with pytest.raises(ValueError, match="gates must pass"):
            v2.run(revised, 0, arm, "formal", torch.device("cpu"))
        for seed in old.SEEDS:
            v2.run(revised, seed, arm, "gate", torch.device("cpu"))
        for seed in old.SEEDS:
            v2.run(revised, seed, arm, "formal", torch.device("cpu"))
            init = v2.c.read_json(revised / f"seed{seed}/arms/{arm}/formal/initialisation.json")
            assert init["gate_checkpoint_loaded"] is False and init["optimizer_initial_state_entries"] == 0
    context = v2.verify(revised)
    assert v2.next_action(context) == "summary - -"
    result = v2.summarize(context)
    assert result["new_formal_updates"] == 32 and result["reused_updates"] == 4
    assert result["formal_updates_total"] == 36 and len(result["rows"]) == 20
    assert v2.summarize(context) == result
    assert snapshot(roots) == before
    with pytest.raises(FileExistsError):
        v2.run(revised, 1, "B0", "formal", torch.device("cpu"))
    replay = revised / "seed2/arms/S1/formal/saved_weight_replay.json"
    replay.write_bytes(replay.read_bytes() + b" ")
    with pytest.raises(ValueError, match="replay changed"):
        v2.summarize(v2.verify(revised))


def test_tiny_gradient_variation_can_be_amplified_by_adam_without_optimizer_randomness():
    parameters = [torch.nn.Parameter(torch.tensor([.01], dtype=torch.float64)) for _ in range(2)]
    for p, gradient in zip(parameters, [1e-9, 2e-9]):
        p.grad = torch.tensor([gradient], dtype=torch.float64)
        torch.optim.AdamW([p], lr=.001, weight_decay=0.).step()
    assert abs(float(parameters[0] - parameters[1])) > 5e-5
    # An illustration of sensitivity, not proof of the specific remote kernel.