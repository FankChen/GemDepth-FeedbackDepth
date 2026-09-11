"""Corrected G1 must prove its baseline, discard gate updates, and keep the protocol.

Tiny CPU fixtures exercise the entire orchestration; they are not real-data or
official IGEV accuracy results. Full-size recurrence math has separate tests.
"""

import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "test")]

import stereogru_corrected_gru as gru
import stereogru_matched_controls as controls
import stereogru_matched_repeats as repeats
from test_stereogru_corrected_head import fixture
from test_stereogru_matched_controls import prepared_fixture


def test_eight_iteration_final_only_objective_can_learn_synthetic_depth():
    # Deliberately small unrelated tensors; no checkpoint or dataset is accessed.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        head, features, kwargs = fixture(iters=8)
        target = torch.full((1, 3, 1, 64, 64), 8.)
        mask = torch.ones_like(target)
        objective = controls.build_objective("calibrated_index_l1", {"depth_min": 3., "depth_max": 80.})

        @torch.no_grad()
        def evaluate():
            head.eval()
            q = head(features, 16, 16, 3, **kwargs).unflatten(0, (1, 3))
            return float(objective(q, target, mask)["total_loss"])

        initial = evaluate()
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=0.)
        for _ in range(60):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            q = head(features, 16, 16, 3, **kwargs).unflatten(0, (1, 3))
            loss = objective(q, target, mask)["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        final = evaluate()
        print(f"SYNTHETIC 8-iter: initial={initial}, final={final}, ratio={final / initial}")
        assert final < initial * .5
        assert len(head.last_iteration_diagnostics) == 8
    finally:
        torch.set_num_threads(previous_threads)


def test_gate_manifest_never_uses_development_clips():
    manifest = {"dataset": {"splits": {"train": {"scenes": ["Scene01", "Scene02"]}}},
                "splits": {"train": [{"id": "a", "scene": "Scene01"}, {"id": "b", "scene": "Scene01"},
                                     {"id": "c", "scene": "Scene02"}],
                           "dev": [{"id": "d", "scene": "Scene18"}]}}
    selected = gru.choose_gate_manifest(manifest)
    assert set(selected["splits"]) == {"train"}
    assert [row["id"] for row in selected["splits"]["train"]] == ["a", "c"]
    assert len(manifest["splits"]["dev"]) == 1  # Source manifest was not mutated.


def test_method_config_refuses_loss_or_gauge_change(tmp_path):
    config = OmegaConf.load(ROOT / "config/stereogru/corrected_gru.yaml")
    config.objective_contract = "new_unmatched_sequence_loss"
    path = tmp_path / "bad.yaml"
    OmegaConf.save(config, path)
    with pytest.raises(ValueError, match="do not silently change"):
        gru.method_config(path)


def test_baseline_gates_reset_and_all_seed_final_comparison(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        source, _cfg, _calls = prepared_fixture(tmp_path, monkeypatch)
        method_path = tmp_path / "method.yaml"
        config = OmegaConf.load(ROOT / "config/stereogru/corrected_gru.yaml")
        config.gate.steps = 60
        OmegaConf.save(config, method_path)
        with pytest.raises(FileNotFoundError):
            gru.prepare(tmp_path / "no_completed_baselines", tmp_path / "not_created", method_path)
        controls.train_arm(source, "C0", torch.device("cpu"))
        controls.train_arm(source, "C1", torch.device("cpu"))
        baseline_runs = tmp_path / "baseline_repeats"
        repeats.prepare_repeats(source, baseline_runs)
        for seed in (1, 2):
            path = baseline_runs / f"seed{seed}/experiment"
            controls.train_arm(path, "C0", torch.device("cpu"))
            controls.train_arm(path, "C1", torch.device("cpu"))
        repeats.summarize_repeats(baseline_runs)
        baseline_files = {str(path): controls.file_sha256(path)
                          for root in (source, baseline_runs) for path in root.rglob("*") if path.is_file()}
        destination = tmp_path / "gru"
        original_read = controls.read_json
        for filename, key, value in (("repetitions.json", "script_sha256", "foreign"),
                                      ("seed_summary.json", "scene_metrics", {})):
            def tampered(path, filename=filename, key=key, value=value):
                result = original_read(path)
                if Path(path) == baseline_runs / filename:
                    result[key] = value
                return result
            with monkeypatch.context() as patch:
                patch.setattr(controls, "read_json", tampered)
                with pytest.raises(ValueError, match="source changed|summary aggregation"):
                    gru.prepare(baseline_runs, destination, method_path)
            assert not destination.exists()
        contract = gru.prepare(baseline_runs, destination, method_path)
        for seed in gru.SEEDS:
            entry = contract["seed_entries"][str(seed)]
            assert entry["method_parameters"] > entry["baseline_parameters"]
            assert entry["added_state_keys"]
        with pytest.raises(ValueError, match="All three.*gates"):
            gru.train(destination, 0, torch.device("cpu"))
        assert not (destination / "seed0/formal").exists()
        for seed in gru.SEEDS:
            report = gru.run_gate(destination, seed, torch.device("cpu"))
            assert report["status"] == "passed" and report["gate_weights_discarded"]
            assert report["baseline_replayed"] and report["zero_iter_max_abs_diff"] < 1e-6
            assert report["loss_ratio"] < .5
        # A malformed OTHER-seed gate must reject before seed0 spends formal budget.
        for key, value in (("first_gradients", {}), ("first_gradients", {"matcher": 1.}),
                           ("zero_iter_max_abs_diff", float("nan")), ("zero_iter_max_abs_diff", -1.),
                           ("cache_sha256", "foreign"), ("loss_ratio", -1.)):
            def bad_gate(path, key=key, value=value):
                result = original_read(path)
                if Path(path) == destination / "seed2/gate/gate.json":
                    result[key] = value
                return result
            with monkeypatch.context() as patch:
                patch.setattr(controls, "read_json", bad_gate)
                with pytest.raises(ValueError):
                    gru.train(destination, 0, torch.device("cpu"))
            assert not (destination / "seed0/formal").exists()
        with pytest.raises(ValueError, match="not completed"):
            gru.summarize(destination)
        assert not (destination / "comparison.json").exists()
        for seed in gru.SEEDS:
            report = gru.train(destination, seed, torch.device("cpu"))
            assert report["status"] == "completed" and report["steps"] == 4
            evidence = controls.read_json(destination / f"seed{seed}/formal/initialisation.json")
            assert evidence["gate_checkpoint_loaded"] is False
            assert evidence["optimizer_initial_state_entries"] == 0
            assert evidence["method_initial_state_sha256"] == contract["seed_entries"][str(seed)]["method_initial_state_sha256"]
        summary = gru.summarize(destination)
        assert summary["seeds"] == [0, 1, 2] and summary["final_step"] == 4
        assert len(summary["rows"]) == 12
        assert gru.summarize(destination) == summary
        assert baseline_files == {str(path): controls.file_sha256(path)
                                  for root in (source, baseline_runs) for path in root.rglob("*") if path.is_file()}
        # A changed gate cannot silently authorize a previously completed method.
        gate_path = destination / "seed0/gate/gate.json"
        gate = json.loads(gate_path.read_text())
        gate["method_initial_state_sha256"] = "foreign"
        gate_path.write_text(json.dumps(gate))
        with pytest.raises(ValueError, match="Invalid/foreign"):
            gru.summarize(destination)
    finally:
        torch.set_num_threads(previous_threads)


def test_c1_initial_mapping_rejects_dtype_shape_and_extra_keys():
    head, _, _ = fixture()
    state = {key: value for key, value in head.state_dict().items()
             if not key.startswith(head.recurrent_state_prefixes)}
    for modification in ("dtype", "shape", "extra"):
        bad = copy.deepcopy(state)
        key = "classifier.weight"
        if modification == "dtype":
            bad[key] = bad[key].half()
        elif modification == "shape":
            bad[key] = bad[key].flatten()
        else:
            bad["new.weight"] = torch.ones(1)
        with pytest.raises(ValueError, match="initial.*differ"):
            head.load_c1_initial(bad)


@pytest.mark.parametrize("defect", ["missing_scene", "extra_scene", "bad_scene", "bad_overall",
                                  "negative_loss", "nan_loss", "support", "dev_leak"])
def test_metric_evidence_rejects_invalid_inventory_and_aggregates(defect):
    rows = [{"id": "a", "scene": "Scene01", "valid_pixels": 10,
             "index_l1": .2, "absrel": .3, "rmse": 2., "delta1": .8}]
    manifest = {"splits": {"train": copy.deepcopy(rows)}}
    item = {"clips": rows, "overall": controls.aggregate_metrics(rows),
            "scenes": {"Scene01": controls.aggregate_metrics(rows)}}
    metrics = {"train": item}
    gru.validate_metrics(metrics, manifest)
    if defect == "missing_scene":
        item["scenes"].clear()
    elif defect == "extra_scene":
        item["scenes"]["Scene18"] = copy.deepcopy(item["overall"])
    elif defect == "bad_scene":
        item["scenes"]["Scene01"]["index_l1"] = .1
    elif defect == "bad_overall":
        item["overall"]["index_l1"] = .1
    elif defect == "negative_loss":
        rows[0]["index_l1"] = -1.
    elif defect == "nan_loss":
        rows[0]["index_l1"] = float("nan")
    elif defect == "support":
        rows[0]["valid_pixels"] = 11
    else:
        metrics["dev"] = copy.deepcopy(item)
    with pytest.raises(ValueError):
        gru.validate_metrics(metrics, manifest)


def test_metric_replay_checks_scenes_not_only_overall():
    stats = {"valid_pixels": 10, "index_l1": .2, "absrel": .3, "rmse": 2., "delta1": .8}
    expected = {"train": {"overall": stats, "scenes": {"Scene01": copy.deepcopy(stats)}}}
    actual = copy.deepcopy(expected)
    actual["train"]["scenes"]["Scene01"]["absrel"] = .9
    with pytest.raises(ValueError, match="did not replay"):
        gru.assert_metric_replay(actual, expected)