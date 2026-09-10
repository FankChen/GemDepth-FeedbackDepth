"""An explicit quota amendment must not silently loosen spacing or change arms."""

import copy
from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from dataset.stereogru_matched import select_spaced_candidates
from stereogru_matched_controls import make_training_order, validate_config


def load_config(name):
    return OmegaConf.to_container(OmegaConf.load(ROOT / "config/stereogru" / name), resolve=True)


def test_quota_v2_changes_only_protocol_id_and_balanced_train_count():
    original = load_config("matched_c0_c1.yaml")
    revised = validate_config(load_config("matched_c0_c1_30train.yaml"))
    assert revised["protocol"] == "vkitti_calibrated_depth_axis_control_v2_30train"
    assert original["dataset"]["splits"]["train"]["clips_per_scene"] == 16
    assert revised["dataset"]["splits"]["train"]["clips_per_scene"] == 15
    expected = copy.deepcopy(original)
    expected["protocol"] = revised["protocol"]
    expected["dataset"]["splits"]["train"]["clips_per_scene"] = 15
    assert revised == expected


@pytest.mark.parametrize("requested,expected_count", [(16, 0), (15, 15)])
def test_fifteen_spaced_windows_require_explicit_smaller_quota(requested, expected_count):
    # Synthetic boundary example, not a reconstruction of the real camera trajectory.
    candidates = [{"frames": list(range(start, start + 4))} for start in range(120)]
    selected, available = select_spaced_candidates(candidates, requested, min_start_gap=8)
    assert available == 15 and len(selected) == expected_count
    assert all(b["frames"][0] - a["frames"][0] >= 8 for a, b in zip(selected, selected[1:]))


def test_revised_pair_keeps_budget_dev_isolation_and_baseline_gate():
    cfg = validate_config(load_config("matched_c0_c1_30train.yaml"))
    train, dev = cfg["dataset"]["splits"]["train"], cfg["dataset"]["splits"]["dev"]
    ids = [f"{scene}/{index}" for scene in train["scenes"] for index in range(train["clips_per_scene"])]
    assert len(ids) == 30
    assert dev == {"scenes": ["Scene18"], "clips_per_scene": 16}
    order = make_training_order(ids, cfg["steps"], cfg["seed"])
    assert len(order) == 1000 and set(order) == set(ids)
    assert order == make_training_order(ids, 1000, cfg["seed"])
    assert cfg["arms"]["C1"]["requires"] == ["C0"]