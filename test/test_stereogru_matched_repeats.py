"""Replications change only seed, keep the source read-only, and require all pairs."""

import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "test")]

import stereogru_matched_controls as controls
import stereogru_matched_repeats as repeats
from test_stereogru_matched_controls import prepared_fixture


def pair_fixture(seed, absrel):
    manifest = {"splits": {"dev": [{"scene": "Scene18"}]} }
    contract = {"files": {"manifest.json": "fixed_manifest"}, "runtime": "fixed",
                "source_sha256": "fixed", "backbone": "fixed", "head_parameters": 10}
    metrics = {}
    for arm, value in (("C0", absrel), ("C1", absrel + .01)):
        metrics[arm] = {"dev": {"scenes": {"Scene18": {
            "index_l1": value / 10, "absrel": value, "rmse": value * 10, "delta1": 1 - value}}}}
    return {"config": {"seed": seed, "crop_seed": 0, "steps": 1000},
            "manifest": manifest, "contract": contract, "metrics": metrics}


@pytest.mark.parametrize("change", ["budget", "crop", "data", "backbone", "runtime"])
def test_no_protocol_change_hidden_as_seed_repeat(change):
    reference, candidate = pair_fixture(0, .2), pair_fixture(1, .21)
    if change == "budget":
        candidate["config"]["steps"] = 750
    elif change == "crop":
        candidate["config"]["crop_seed"] = 1
    elif change == "data":
        candidate["manifest"]["splits"]["dev"][0]["scene"] = "Scene20"
    else:
        candidate["contract"][change] = "different"
    with pytest.raises(ValueError, match="change|changed"):
        repeats.assert_same_protocol(reference, candidate, 1)


def test_aggregation_uses_all_paired_final_seeds_and_sample_std():
    pairs = {seed: pair_fixture(seed, .2 + seed * .02) for seed in (0, 1, 2)}
    result = repeats.aggregate_pairs(pairs)["dev"]["Scene18"]["absrel"]
    assert result["C0"]["mean"] == pytest.approx(.22)
    assert result["C0"]["sample_std"] == pytest.approx(.02)
    assert result["paired_C1_minus_C0"]["mean"] == pytest.approx(.01)
    assert result["paired_C1_minus_C0"]["sample_std"] == pytest.approx(0.)
    with pytest.raises(ValueError, match="seeds 0/1/2"):
        repeats.aggregate_pairs({0: pairs[0], 2: pairs[2]})


def test_training_seeds_change_order_not_clip_membership():
    ids = [f"Scene{scene:02d}/clip{index:02d}" for scene in (1, 2) for index in range(15)]
    orders = [controls.make_training_order(ids, 1000, seed) for seed in repeats.SEEDS]
    assert len({tuple(order) for order in orders}) == 3
    assert all(len(order) == 1000 and set(order) == set(ids) for order in orders)
    for seed, order in zip(repeats.SEEDS, orders):
        assert order == controls.make_training_order(ids, 1000, seed)


def test_two_prepared_repeats_then_baseline_first_training_and_summary(tmp_path, monkeypatch):
    source, cfg, _ = prepared_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="no completed certificate"):
        repeats.prepare_repeats(source, tmp_path / "too_early")
    assert not (tmp_path / "too_early").exists()
    controls.train_arm(source, "C0", torch.device("cpu"))
    controls.train_arm(source, "C1", torch.device("cpu"))
    original = {str(path.relative_to(source)): controls.file_sha256(path)
                for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "repetitions"
    plan = repeats.prepare_repeats(source, output)
    assert plan["remaining_training_runs"] == ["seed1/C0", "seed1/C1", "seed2/C0", "seed2/C1"]
    with pytest.raises(ValueError, match="no completed certificate"):
        repeats.summarize_repeats(output)
    assert not (output / "seed_summary.json").exists()
    for seed in (1, 2):
        experiment = output / f"seed{seed}/experiment"
        source_cfg = controls.read_json(source / "config.json")
        repeated_cfg = controls.read_json(experiment / "config.json")
        expected = copy.deepcopy(source_cfg)
        expected["seed"] = seed
        assert repeated_cfg == expected
        assert (source / "manifest.json").read_bytes() == (experiment / "manifest.json").read_bytes()
        with pytest.raises(ValueError, match="no completed certificate"):
            controls.train_arm(experiment, "C1", torch.device("cpu"))
        controls.train_arm(experiment, "C0", torch.device("cpu"))
        controls.train_arm(experiment, "C1", torch.device("cpu"))
    report = repeats.summarize_repeats(output)
    assert report["seeds"] == [0, 1, 2] and report["steps_each"] == cfg.steps
    assert repeats.summarize_repeats(output) == report
    assert original == {str(path.relative_to(source)): controls.file_sha256(path)
                        for path in source.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="new empty directory"):
        repeats.prepare_repeats(source, output)