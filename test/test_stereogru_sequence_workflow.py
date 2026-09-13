"""End-to-end synthetic protocol, source immutability and fail-closed orchestration."""

import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "test")]

import stereogru_matched_controls as controls
import stereogru_matched_repeats as repeats
import stereogru_sequence_experiment as experiment
from test_stereogru_matched_controls import prepared_fixture


@pytest.fixture
def completed_sources(tmp_path, monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    source, cfg, calls = prepared_fixture(tmp_path, monkeypatch)
    for arm in ("C0", "C1"):
        controls.train_arm(source, arm, torch.device("cpu"))
    repetitions = tmp_path / "source_repeats"
    repeats.prepare_repeats(source, repetitions)
    for seed in (1, 2):
        for arm in ("C0", "C1"):
            controls.train_arm(repetitions / f"seed{seed}/experiment", arm, torch.device("cpu"))
    repeats.summarize_repeats(repetitions)
    method = OmegaConf.load(experiment.DEFAULT_CONFIG)
    method.gate.steps = 60
    path = tmp_path / "sequence.yaml"
    OmegaConf.save(method, path)
    yield source, repetitions, path, cfg, calls
    torch.set_num_threads(previous)


def snapshot(roots):
    return {str(path): controls.file_sha256(path) for root in roots for path in root.rglob("*") if path.is_file()}


def test_complete_new_baseline_final_control_sequence_method_and_immutable_sources(completed_sources, tmp_path, monkeypatch):
    source, repetitions, config, cfg, calls = completed_sources
    originals = snapshot((source, repetitions))
    destination = tmp_path / "sequence"
    contract = experiment.prepare(repetitions, destination, config)
    assert snapshot((source, repetitions)) == originals
    for entry in contract["seed_entries"].values():
        assert entry["arms"]["F1"]["initial_file"] == entry["arms"]["S1"]["initial_file"]
        assert entry["arms"]["F1"]["parameters"] == entry["arms"]["S1"]["parameters"]
        assert entry["arms"]["S1"]["parameters"] > entry["arms"]["B0"]["parameters"]
        assert entry["arms"]["B0"]["weights"] == [1.]
        assert sum(entry["arms"]["S1"]["weights"]) == pytest.approx(1.)
    assert experiment.next_action(destination) == "train B0 0"
    with pytest.raises(ValueError, match="not completed"):
        experiment.run_gate(destination, 0, "F1", torch.device("cpu"))
    with pytest.raises(ValueError, match="not completed"):
        experiment.train(destination, 0, "S1", torch.device("cpu"))
    assert not (destination / "seed0/arms/S1").exists()

    for seed in experiment.SEEDS:
        # A fresh CLI process may start with these settings: never inherit prepare's policy.
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        done = experiment.train(destination, seed, "B0", torch.device("cpu"))
        assert done["status"] == "completed" and done["steps"] == cfg.steps
        new = controls.read_json(destination / f"seed{seed}/arms/B0/formal/metrics_final.json")
        pair = experiment.verify(destination)["pairs"][seed]
        experiment.legacy.assert_metric_replay(new, pair["metrics"]["C1"])
    assert experiment.next_action(destination) == "gate F1 0"

    for arm in ("F1", "S1"):
        with pytest.raises(ValueError, match="gates must pass"):
            experiment.train(destination, 0, arm, torch.device("cpu"))
        for seed in experiment.SEEDS:
            gate = experiment.run_gate(destination, seed, arm, torch.device("cpu"))
            assert gate["status"] == "passed"
            assert max(gate["ratios"].values()) <= .5
            assert not list((destination / f"seed{seed}/arms/{arm}/gate").glob("*.pth"))
        assert experiment.next_action(destination) == f"train {arm} 0"
        original_read = controls.read_json
        for key, value in (("ratios", {"weighted_sequence": -1., "final_index_l1": -1.}),
                           ("cache_sha256", "foreign"), ("steps", 1)):
            def bad_gate(path, key=key, value=value):
                result = original_read(path)
                if Path(path) == destination / f"seed2/arms/{arm}/gate/gate.json":
                    result[key] = value
                return result
            with monkeypatch.context() as local:
                local.setattr(controls, "read_json", bad_gate)
                with pytest.raises(ValueError):
                    experiment.train(destination, 0, arm, torch.device("cpu"))
            assert not (destination / f"seed0/arms/{arm}/formal").exists()
        for seed in experiment.SEEDS:
            done = experiment.train(destination, seed, arm, torch.device("cpu"))
            assert done["steps"] == cfg.steps
            assert set(done["gate_certificates"]) == {"0", "1", "2"}
            initial = controls.read_json(destination / f"seed{seed}/arms/{arm}/formal/initialisation.json")
            assert initial["gate_checkpoint_loaded"] is False and initial["optimizer_initial_state_entries"] == 0
            assert initial["initial_state_sha256"] == contract["seed_entries"][str(seed)]["arms"][arm]["initial_state_sha256"]
        with pytest.raises(FileExistsError):
            experiment.train(destination, 0, arm, torch.device("cpu"))

    assert calls["forward"] == 18  # Historical extraction only: no new arm reruns the backbone.
    assert experiment.next_action(destination) == "summarize - -"
    result = experiment.summarize(destination)
    assert result["final_step"] == cfg.steps and result["formal_updates"] == 36
    assert result["disposable_gate_updates"] == 360 and result["seeds"] == [0, 1, 2]
    assert len(result["rows"]) == 20  # train overall+2scenes, dev overall+1scene, 4 metrics.
    assert experiment.summarize(destination) == result
    assert snapshot((source, repetitions)) == originals
    trace = controls.read_json(destination / "seed0/arms/S1/formal/evaluation_final.json")
    assert len(trace["stages"]) == 9
    bad = destination / "seed1/arms/F1/formal/progress.jsonl"
    bad.write_bytes(bad.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        experiment.summarize(destination)


@pytest.mark.parametrize("defect", ["skip_baseline", "skip_final_control", "short_iters", "wrong_objective",
                                  "gamma", "update", "seeds", "loosen_gate"])
def test_config_cannot_silently_change_experiment_contract(tmp_path, defect):
    cfg = OmegaConf.to_container(OmegaConf.load(experiment.DEFAULT_CONFIG), resolve=True)
    if defect == "skip_baseline":
        cfg["arms"]["F1"]["requires"] = []
    elif defect == "skip_final_control":
        cfg["arms"]["S1"]["requires"] = ["B0"]
    elif defect == "short_iters":
        cfg["arms"]["S1"]["iterations"] = 4
    elif defect == "wrong_objective":
        cfg["arms"]["S1"]["objective"] = "calibrated_index_l1"
    elif defect == "gamma":
        cfg["arms"]["S1"]["objective_kwargs"]["gamma"] = .8
    elif defect == "update":
        cfg["index_update"] = "unbounded_additive"
    elif defect == "seeds":
        cfg["seeds"] = [0]
    else:
        cfg["gate"]["maximum_final_initial_loss_ratio"] = 1.
    path = tmp_path / "wrong.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    with pytest.raises(ValueError):
        experiment.method_config(path)


def test_failed_phase_never_marks_complete_or_resumes(completed_sources, tmp_path, monkeypatch):
    _source, repetitions, config, _cfg, _calls = completed_sources
    root = tmp_path / "sequence"
    experiment.prepare(repetitions, root, config)

    def timeout(*_args, **_kwargs):
        raise TimeoutError("test timeout before optimizer")

    with monkeypatch.context() as local:
        local.setattr(experiment, "optimisation", timeout)
        with pytest.raises(TimeoutError):
            experiment.train(root, 0, "B0", torch.device("cpu"))
    assert (root / "seed0/arms/B0/formal/failure.json").is_file()
    assert not (root / "seed0/arms/B0/formal/completed.json").exists()
    with pytest.raises(ValueError, match="not completed"):
        experiment.next_action(root)
    with pytest.raises(FileExistsError):
        experiment.train(root, 0, "B0", torch.device("cpu"))


def test_evaluator_matches_historical_formula_and_fast_final_output(completed_sources, tmp_path):
    _source, repetitions, config, _cfg, _calls = completed_sources
    root = tmp_path / "sequence"
    experiment.prepare(repetitions, root, config)
    context = experiment.verify(root)
    pair = context["pairs"][0]
    device = torch.device("cpu")
    clips, _ = experiment.legacy.load_cache(pair, device, float("inf"))
    head = experiment.instantiate(context, 0, "S1", device)
    objective = experiment.objective_for(pair, context["config"]["arms"]["S1"])
    actual = experiment.evaluate_sequence(head, clips, pair["manifest"], objective, device, float("inf"))
    expected = controls.evaluate(head, clips, pair["manifest"], objective.base, device, float("inf"))
    experiment.legacy.assert_metric_replay(actual["stages"][-1], expected)
    for split in expected:
        assert actual["stages"][-1][split]["overall"] == expected[split]["overall"]
    for defect in ("weights", "count", "aggregate", "scene", "nonfinite"):
        bad = copy.deepcopy(actual)
        if defect == "weights":
            bad["weights"][0] = 0.
        elif defect == "count":
            bad["stages"].pop()
        elif defect == "aggregate":
            bad["weighted_index_l1"]["train"] += .1
        elif defect == "scene":
            bad["stages"][0]["dev"]["scenes"].clear()
        else:
            bad["stages"][0]["train"]["clips"][0]["q_min"] = float("nan")
        with pytest.raises(ValueError):
            experiment.validate_evaluation(bad, pair["manifest"], objective.weights)


def test_prepare_refuses_source_output_and_foreign_initial(completed_sources, tmp_path):
    source, repetitions, config, _cfg, _calls = completed_sources
    with pytest.raises(ValueError, match="outside"):
        experiment.prepare(repetitions, source / "new_run", config)
    assert not (source / "new_run").exists()
    root = tmp_path / "sequence"
    contract = experiment.prepare(repetitions, root, config)
    path = root / contract["seed_entries"]["2"]["arms"]["F1"]["initial_file"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="INITIAL file changed"):
        experiment.train(root, 0, "B0", torch.device("cpu"))
    assert not (root / "seed0/arms").exists()


def test_actual_training_entry_restores_precision_before_cache(completed_sources, tmp_path, monkeypatch):
    _source, repetitions, config, _cfg, _calls = completed_sources
    root = tmp_path / "sequence"
    experiment.prepare(repetitions, root, config)
    context = experiment.verify(root)
    destination = tmp_path / "unused"
    destination.mkdir()
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    def first_boundary(*_args, **_kwargs):
        assert experiment.numeric_policy() == {"float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False,
                                               "cudnn_allow_tf32": False, "cudnn_benchmark": False}
        assert torch.initial_seed() == 1
        raise RuntimeError("policy checked before cache")

    monkeypatch.setattr(experiment.legacy, "load_cache", first_boundary)
    with pytest.raises(RuntimeError, match="policy checked"):
        experiment.optimisation(context, 1, "B0", destination, context["pairs"][1]["manifest"], [],
                                torch.device("cpu"), 2, 2, {})
    assert not (destination / "progress.jsonl").exists()


def test_whole_source_inventory_and_unlisted_dependency_are_bound(completed_sources, tmp_path, monkeypatch):
    _source, repetitions, config, _cfg, _calls = completed_sources
    root = tmp_path / "sequence"
    contract = experiment.prepare(repetitions, root, config)
    dependency = "model/motion_module/motion_module.py"
    assert dependency in contract["sources"] and dependency not in experiment.SOURCES
    original_hash = controls.file_sha256
    with monkeypatch.context() as local:
        local.setattr(controls, "file_sha256", lambda path: "changed" if Path(path) == ROOT / dependency
                      else original_hash(path))
        with pytest.raises(ValueError, match="Sequence source changed"):
            experiment.verify(root)
    inventory = experiment.source_inventory()
    monkeypatch.setattr(experiment, "source_inventory", lambda: inventory + ("model/dpt_added.py",))
    with pytest.raises(ValueError, match="source inventory changed"):
        experiment.train(root, 0, "B0", torch.device("cpu"))
    assert not (root / "seed0/arms").exists()