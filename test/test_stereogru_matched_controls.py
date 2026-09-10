"""Fixed split, identical initial states and fail-closed baseline-first controls."""

import copy
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "test")]

from dataset.stereogru_matched import (ManifestQuotaError, build_manifest, load_manifest_clips,
                                      select_spaced_candidates, validate_manifest)
from model.decoder_registry import build_decoder, get_decoder_class
from test_stereogru_calibrated_baseline import camera_fixture, write_vkitti_scene
import stereogru_matched_controls as controls


def small_config():
    cfg = OmegaConf.load(ROOT / "config/stereogru/matched_c0_c1.yaml")
    cfg.dataset.crop_size, cfg.dataset.seq_len, cfg.dataset.min_start_gap = 64, 3, 5
    cfg.dataset.min_baseline_m, cfg.dataset.max_baseline_m = .4, .6
    for split in cfg.dataset.splits.values():
        split.clips_per_scene = 2
    cfg.decoder.kwargs.num_sample, cfg.decoder.kwargs.num_groups = 8, 2
    cfg.decoder.kwargs.match_dim, cfg.decoder.kwargs.hidden_dim = 8, 16
    cfg.steps, cfg.log_every, cfg.eval_every = 4, 2, 2
    return cfg


def fixture_manifest(tmp_path):
    root = tmp_path / "vkitti"
    for scene in ("Scene01", "Scene02", "Scene18", "Scene20"):
        write_vkitti_scene(root, scene, frames=18)
    cfg = small_config()
    manifest = build_manifest(root, OmegaConf.to_container(cfg.dataset, resolve=True), 3., 80.)
    return cfg, root, manifest


def test_flat_and_full_share_exact_initial_state_without_new_parameters():
    args = {"num_sample": 8, "num_groups": 2, "match_dim": 8, "hidden_dim": 16,
            "depth_min": 3., "depth_max": 80.}
    torch.manual_seed(9)
    full = build_decoder(get_decoder_class("DPTHeadCalibratedVolumeOnlyConvNeXt"), args,
                         in_channels_list=[8, 16, 32, 64], patch_size=4)
    torch.manual_seed(9)
    flat = build_decoder(get_decoder_class("DPTHeadCalibratedFlatVolumeConvNeXt"), args,
                         in_channels_list=[8, 16, 32, 64], patch_size=4)
    assert set(full.state_dict()) == set(flat.state_dict())
    assert all(torch.equal(value, flat.state_dict()[name]) for name, value in full.state_dict().items())
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in flat.parameters())
    images, K, E = camera_fixture()
    features = torch.randn(3, 16, 8, 8)
    for training in (True, False):
        full.train(training)
        flat.train(training)
        original = full._build_volume(features, images, E, K, 3)
        actual = flat._build_volume(features, images, E, K, 3)
        assert torch.equal(actual, original.mean(2, keepdim=True).expand_as(original))
        assert torch.allclose(actual.std(dim=2), torch.zeros_like(actual[:, :, 0]), atol=1e-8)
    actual.square().mean().backward()
    assert sum(p.grad.square().sum() for p in flat.matcher.parameters()) > 0


def test_spacing_is_maximal_then_evenly_thinned():
    candidates = [{"frames": [start, start + 1, start + 2]} for start in range(33)]
    selected, available = select_spaced_candidates(candidates, 3, 8)
    assert available == 5
    assert [row["frames"][0] for row in selected] == [0, 16, 32]
    selected, available = select_spaced_candidates(candidates, 6, 8)
    assert selected == [] and available == 5


def test_manifest_fixed_quota_scene_gap_support_and_exact_replay(tmp_path):
    cfg, root, manifest = fixture_manifest(tmp_path)
    assert len(manifest["splits"]["train"]) == 4
    assert len(manifest["splits"]["dev"]) == 2
    assert {row["scene"] for row in manifest["splits"]["train"]} == {"Scene01", "Scene02"}
    assert {row["scene"] for row in manifest["splits"]["dev"]} == {"Scene18"}
    assert "Scene20" not in manifest["selection_report"]
    repeated = build_manifest(root, OmegaConf.to_container(cfg.dataset, resolve=True), 3., 80.)
    assert repeated == manifest
    clips = load_manifest_clips(manifest)
    assert len(clips) == 6
    for rows in manifest["splits"].values():
        for row in rows:
            assert row["valid_pixels"] == 3 * 64 * 64
            assert clips[row["id"]]["intrinsics"].shape == (1, 3, 3, 3)
            assert torch.equal(clips[row["id"]]["extrinsics"][0, 0], torch.eye(4))
    saved = json.loads(json.dumps(manifest))
    assert validate_manifest(saved)


def test_insufficient_scene_count_never_relaxes_or_borrows_dev(tmp_path):
    cfg, root, _ = fixture_manifest(tmp_path)
    cfg.dataset.splits.train.clips_per_scene = 99
    with pytest.raises(ManifestQuotaError, match="no threshold/data fallback") as caught:
        build_manifest(root, OmegaConf.to_container(cfg.dataset, resolve=True), 3., 80.)
    assert caught.value.report["Scene01"]["required"] == 99
    assert caught.value.report["Scene01"]["selected"] == 0
    assert caught.value.report["Scene18"]["selected"] == 2


def test_manifest_leakage_and_modified_data_fail_closed(tmp_path):
    cfg, root, manifest = fixture_manifest(tmp_path)
    bad = copy.deepcopy(manifest)
    bad["dataset"]["splits"]["dev"]["scenes"] = ["Scene01"]
    with pytest.raises(ValueError, match="Scene leakage"):
        validate_manifest(bad)
    cfg.dataset.splits.dev.scenes = ["Scene20"]
    with pytest.raises(ValueError, match="training pool"):
        build_manifest(root, OmegaConf.to_container(cfg.dataset, resolve=True), 3., 80.)
    rgb = Path(manifest["splits"]["train"][0]["rgb"][0])
    rgb.write_bytes(rgb.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Input file changed"):
        load_manifest_clips(manifest)


def prepared_fixture(tmp_path, monkeypatch):
    cfg, data_root, _ = fixture_manifest(tmp_path)
    calls = {"forward": 0}

    class FakeBackbone(torch.nn.Module):
        embed_dims = [8, 16, 32, 64]
        feat_strides = [4, 8, 16, 32]
        patch_size = 4

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()

        def forward(self, value):
            calls["forward"] += 1
            y, x = torch.meshgrid(torch.linspace(0., 1., 64, device=value.device),
                                   torch.linspace(0., 1., 64, device=value.device), indexing="ij")
            pattern = value.mean(1, keepdim=True) + (x.square() + y)[None, None]
            return [F.adaptive_avg_pool2d(pattern, 64 // stride).repeat(1, dim, 1, 1)
                    for dim, stride in zip(self.embed_dims, self.feat_strides)]

    monkeypatch.setattr(controls, "load_clean_backbone", lambda *args: (FakeBackbone().eval(), {"synthetic": True}))
    # No real weights in this fixture; actual official-state hashing has separate tests.
    monkeypatch.setattr(controls, "verify_base", lambda state: "synthetic_base_digest")
    config_path = tmp_path / "small.yaml"
    OmegaConf.save(cfg, config_path)
    weights = tmp_path / "synthetic_backbone.pth"
    weights.write_bytes(b"synthetic backbone fixture")
    experiment = tmp_path / "experiment"
    controls.prepare(config_path, weights, data_root, experiment)
    return experiment, cfg, calls


def test_matched_pair_baseline_gate_common_init_cache_and_final_metrics(tmp_path, monkeypatch):
    experiment, cfg, calls = prepared_fixture(tmp_path, monkeypatch)
    initial_bytes = (experiment / "initial_head.pth").read_bytes()
    with pytest.raises(ValueError, match="no completed certificate"):
        controls.train_arm(experiment, "C1", torch.device("cpu"))
    assert not (experiment / "arms/C1").exists()
    with pytest.raises(ValueError, match="no completed certificate"):
        controls.compare(experiment)
    baseline = controls.train_arm(experiment, "C0", torch.device("cpu"))
    assert baseline["status"] == "completed" and baseline["steps"] == cfg.steps
    assert not (experiment / "arms/C1").exists()  # No implicit method launch.
    assert calls["forward"] == 6  # 4 train + 2 dev, computed once.
    cache_digest = controls.file_sha256(experiment / "features.pth")
    baseline_digest = controls.file_sha256(experiment / "arms/C0/final_head.pth")
    full = controls.train_arm(experiment, "C1", torch.device("cpu"))
    assert calls["forward"] == 6 and controls.file_sha256(experiment / "features.pth") == cache_digest
    for key in ("initial_state_sha256", "initial_file_sha256", "manifest_sha256", "order_sha256", "cache_sha256", "head_parameters"):
        assert baseline[key] == full[key]
    assert baseline["optimizer_initial_state_entries"] == full["optimizer_initial_state_entries"] == 0
    assert (experiment / "initial_head.pth").read_bytes() == initial_bytes
    assert controls.file_sha256(experiment / "arms/C0/final_head.pth") == baseline_digest
    for arm in ("C0", "C1"):
        evidence = json.loads((experiment / f"arms/{arm}/initialisation.json").read_text())
        assert evidence["initial_state_sha256"] == baseline["initial_state_sha256"]
    report = controls.compare(experiment)
    assert report["matched_initialisation_manifest_order_cache"]
    for split in ("train", "dev"):
        a, b = [report["results"][arm]["metrics"][split] for arm in ("C0", "C1")]
        assert a["overall"]["valid_pixels"] == b["overall"]["valid_pixels"]
        assert [row["id"] for row in a["clips"]] == [row["id"] for row in b["clips"]]
    assert set(report["results"]["C0"]["metrics"]["dev"]["scenes"]) == {"Scene18"}
    with pytest.raises(FileExistsError, match="no resume/overwrite"):
        controls.train_arm(experiment, "C0", torch.device("cpu"))


@pytest.mark.parametrize("changed", ["short_budget", "checkpoint", "feature_cache"])
def test_changed_or_short_baseline_cannot_unlock_later_arm(tmp_path, monkeypatch, changed):
    experiment, _, _ = prepared_fixture(tmp_path, monkeypatch)
    controls.train_arm(experiment, "C0", torch.device("cpu"))
    if changed == "short_budget":
        path = experiment / "arms/C0/completed.json"
        value = json.loads(path.read_text())
        value["steps"] = 1
        path.write_text(json.dumps(value))
    else:
        paths = {"checkpoint": experiment / "arms/C0/final_head.pth", "feature_cache": experiment / "features.pth"}
        path = paths[changed]
        path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="mismatch|changed"):
        controls.train_arm(experiment, "C1", torch.device("cpu"))
    assert not (experiment / "arms/C1").exists()


def test_timeout_never_marks_baseline_completed(tmp_path, monkeypatch):
    experiment, _, _ = prepared_fixture(tmp_path, monkeypatch)

    def timeout(_deadline):
        raise TimeoutError("synthetic safety limit")

    monkeypatch.setattr(controls, "check_deadline", timeout)
    with pytest.raises(TimeoutError, match="safety limit"):
        controls.train_arm(experiment, "C0", torch.device("cpu"))
    assert not (experiment / "arms/C0/completed.json").exists()
    assert (experiment / "arms/C0/failure.json").is_file()
    with pytest.raises(ValueError, match="no completed certificate"):
        controls.train_arm(experiment, "C1", torch.device("cpu"))


def test_shared_schedule_change_and_removed_prerequisite_are_rejected(tmp_path, monkeypatch):
    cfg = OmegaConf.to_container(small_config(), resolve=True)
    cfg["arms"]["C1"]["requires"] = []
    with pytest.raises(ValueError, match="baseline-first cannot be bypassed"):
        controls.validate_config(cfg)
    experiment, _, _ = prepared_fixture(tmp_path, monkeypatch)
    path = experiment / "training_order.json"
    order = json.loads(path.read_text())
    order[0] = "Scene18/not-a-training-clip"
    path.write_text(json.dumps(order))
    with pytest.raises(ValueError, match="Shared experiment artifact changed"):
        controls.train_arm(experiment, "C0", torch.device("cpu"))
    assert not (experiment / "arms/C0").exists()


def test_full_volume_cannot_be_disguised_as_the_first_baseline():
    cfg = OmegaConf.to_container(small_config(), resolve=True)
    cfg["arms"]["C0"]["decoder"] = cfg["arms"]["C1"]["decoder"]
    with pytest.raises(ValueError, match="requires ordered C0=flat"):
        controls.validate_config(cfg)
    cfg = OmegaConf.to_container(small_config(), resolve=True)
    cfg["arms"] = {"C1": {"decoder": cfg["arms"]["C1"]["decoder"], "requires": []},
                   "C0": {"decoder": cfg["arms"]["C0"]["decoder"], "requires": ["C1"]}}
    with pytest.raises(ValueError, match="requires ordered C0=flat"):
        controls.validate_config(cfg)


def test_deadline_checked_inside_caching_before_writing_shared_cache(tmp_path, monkeypatch):
    experiment, _, calls = prepared_fixture(tmp_path, monkeypatch)

    def expire_after_one_feature(_deadline):
        if calls["forward"]:
            raise TimeoutError("synthetic expiry within cache")

    monkeypatch.setattr(controls, "check_deadline", expire_after_one_feature)
    with pytest.raises(TimeoutError, match="expiry within cache"):
        controls.train_arm(experiment, "C0", torch.device("cpu"))
    assert calls["forward"] == 1
    assert not (experiment / "features.pth").exists()
    assert not (experiment / "arms/C0/completed.json").exists()