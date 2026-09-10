"""Fixed-state dependency probes must replay, preserve masks and never train."""

import copy
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import diagnose_stereogru_pilot as readout
import stereogru_calibrated_baseline as pilot
from loss.objective_registry import build_objective
from model.decoder_registry import build_decoder, get_decoder_class


def small_head_and_clip():
    torch.manual_seed(0)
    head = build_decoder(get_decoder_class("DPTHeadCalibratedVolumeOnlyConvNeXt"),
                         {"num_sample": 8, "num_groups": 2, "match_dim": 8,
                          "hidden_dim": 16, "depth_min": 3., "depth_max": 80.},
                         in_channels_list=[8, 16, 32, 64], patch_size=4).eval()
    frames, size = 3, 64
    K = torch.tensor([[64., 0., 26.], [0., 64., 31.], [0., 0., 1.]]).repeat(1, frames, 1, 1)
    E = torch.eye(4).repeat(1, frames, 1, 1)
    E[0, :, 0, 3] = torch.arange(frames) * .3
    clip = {"images": torch.rand(1, frames, 3, size, size), "intrinsics": K, "extrinsics": E,
            "depth": torch.full((1, frames, 1, size, size), 8.),
            "mask": torch.ones(1, frames, 1, size, size),
            "features": [torch.randn(frames, dim, size // (4 * 2 ** i), size // (4 * 2 ** i))
                         for i, dim in enumerate([8, 16, 32, 64])]}
    objective = build_objective("calibrated_index_l1", {"depth_min": 3., "depth_max": 80.})
    return head, clip, objective


def test_raw_probes_preserve_original_and_control_mean_or_bin_order():
    raw = torch.arange(2 * 3 * 4 * 5 * 6.).reshape(2, 3, 4, 5, 6)
    before = raw.clone()
    flat = readout.RAW_INTERVENTIONS["flat_depth_raw"](raw)
    reverse = readout.RAW_INTERVENTIONS["reverse_depth_raw"](raw)
    assert torch.equal(flat.mean(dim=2), raw.mean(dim=2))
    assert torch.count_nonzero(flat.std(dim=2)) == 0
    assert torch.equal(reverse.sort(dim=2).values, raw.sort(dim=2).values)
    assert not torch.equal(reverse, raw)
    assert torch.count_nonzero(readout.RAW_INTERVENTIONS["zero_raw"](raw)) == 0
    assert torch.equal(raw, before)


def test_readout_keeps_state_masks_and_removes_hooks():
    head, clip, objective = small_head_and_clip()
    before = {name: value.clone() for name, value in head.state_dict().items()}
    original_features = [value.clone() for value in clip["features"]]
    reference = pilot.evaluate_overfit(head, [clip], objective)
    groups, digest = readout.diagnose_groups(head, {"fitted": [clip]}, objective, reference)
    assert set(groups["fitted"]) == set(readout.RAW_INTERVENTIONS)
    for result in groups["fitted"].values():
        assert result["valid_pixels"] == reference["valid_pixels"]
    zero = groups["fitted"]["zero_raw"]["clips"][0]
    assert zero["effective_raw_input"]["zero_fraction"] == 1.
    assert zero["raw_zero_fraction"] < 1.  # Original evidence was not mutated.
    assert digest == readout.fingerprint_state(head.state_dict())
    assert all(torch.equal(before[name], value) for name, value in head.state_dict().items())
    assert all(torch.equal(a, b) for a, b in zip(original_features, clip["features"]))
    assert not head.volume_stem._forward_pre_hooks and not head._forward_hooks
    assert all(parameter.grad is None for parameter in head.parameters())


def test_hooks_removed_even_on_exception_and_training_state_rejected():
    head, _, _ = small_head_and_clip()
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with readout.observe_intervention(head, "zero_raw"):
            raise RuntimeError("synthetic failure")
    assert not head.volume_stem._forward_pre_hooks and not head._forward_hooks
    head.train()
    with pytest.raises(ValueError, match="entirely in eval"):
        with readout.observe_intervention(head, "normal"):
            pass


def test_nonfinite_raw_cannot_be_hidden_by_zeroing():
    head, _, _ = small_head_and_clip()
    raw = torch.full((1, 2, 8, 8, 8), float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite original"):
        with readout.observe_intervention(head, "zero_raw"):
            head.volume_stem(raw)
    assert not head.volume_stem._forward_pre_hooks


@pytest.mark.parametrize("key", ["index_l1", "absrel", "rmse", "delta1", "valid_pixels"])
def test_replay_rejects_changed_metric_or_support(key):
    expected = {"index_l1": .02, "absrel": .1, "rmse": 5., "delta1": .95, "valid_pixels": 100}
    actual = dict(expected)
    actual[key] += 1
    with pytest.raises(ValueError, match="did not replay"):
        readout.assert_metric_replay(actual, expected)


def test_grouping_replays_training_and_separates_unfitted_scene():
    manifests = [{"scene": scene, "rgb": [path], "frames": [frame]}
                 for scene, path, frame in [("Scene01", "a", 10), ("Scene02", "b", 20),
                                            ("Scene18", "c", 30), ("Scene01", "d", 11)]]
    clips = [{"manifest": value} for value in manifests]
    groups = readout.group_clips(clips, manifests[:2])
    assert len(groups["fitted"]) == 2
    assert groups["unfitted_training_scene"][0]["manifest"]["scene"] == "Scene18"
    assert groups["unfitted_same_scene"][0]["manifest"]["scene"] == "Scene01"
    with pytest.raises(ValueError, match="replay differs"):
        readout.group_clips(clips, list(reversed(manifests[:2])))
    overlap = copy.deepcopy(clips)
    overlap[-1]["manifest"]["rgb"] = ["a"]
    with pytest.raises(ValueError, match="overlap frames"):
        readout.group_clips(overlap, manifests[:2])


def test_saved_synthetic_pilot_readout_end_to_end(tmp_path, monkeypatch):
    _, clip, _ = small_head_and_clip()
    clips = []
    for index, scene in enumerate(("Scene01", "Scene02", "Scene18", "Scene01")):
        item = copy.deepcopy(clip)
        item.pop("features")
        item["images"] = item["images"].roll(index * 2, dims=-1)
        source = tmp_path / f"data_{index}.bin"
        source.write_bytes(f"synthetic source {index}".encode())
        item["manifest"] = {"scene": scene, "rgb": [str(source)], "depth": [], "calibration": [],
                            "frames": [index * 10, index * 10 + 1, index * 10 + 2]}
        clips.append(item)

    class FakeBackbone(torch.nn.Module):
        embed_dims = [8, 16, 32, 64]
        patch_size = 4

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()

        def forward(self, value):
            return [F.adaptive_avg_pool2d(value.mean(dim=1, keepdim=True), 64 // (4 * 2 ** i)).repeat(1, dim, 1, 1)
                    for i, dim in enumerate(self.embed_dims)]

    def fake_clips(*args, num_clips, **kwargs):
        return copy.deepcopy(clips[:num_clips])

    for module in (pilot, readout):
        monkeypatch.setattr(module, "load_training_clips", fake_clips)
        monkeypatch.setattr(module, "load_clean_backbone", lambda *args: (FakeBackbone().eval(), {"synthetic": True}))
    # Exact real-weight verification is tested separately; this fixture has no real weights.
    monkeypatch.setattr(readout, "verify_base", lambda state: "synthetic fixture only")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"synthetic weights")
    cfg = OmegaConf.load(ROOT / "config/stereogru/calibrated_volume_only.yaml")
    cfg.steps = 3
    cfg.dataset.crop_size, cfg.dataset.seq_len = 64, 3
    cfg.decoder.kwargs.num_sample, cfg.decoder.kwargs.num_groups = 8, 2
    cfg.decoder.kwargs.match_dim, cfg.decoder.kwargs.hidden_dim = 8, 16
    pilot_dir = tmp_path / "pilot"
    pilot.run(cfg, weights, tmp_path, pilot_dir, torch.device("cpu"))
    before = {p.name: pilot.sha256(p) for p in pilot_dir.iterdir()}
    report = readout.run_readout(pilot_dir, tmp_path, tmp_path / "readout", torch.device("cpu"), extra_clips=2)
    assert report["pilot_replayed"] and report["head_state_unchanged"]
    assert report["baseline_file_unchanged_during_readout"]
    assert report["baseline_file_sha256"] == before["baseline_head.pth"]
    assert "no independent final-head fingerprint" in report["baseline_identity_caveat"]
    assert report["training_steps"] == 0 and report["method_launched"] is False
    assert set(report["groups"]) == {"fitted", "unfitted_training_scene", "unfitted_same_scene"}
    assert before == {p.name: pilot.sha256(p) for p in pilot_dir.iterdir()}
    assert (tmp_path / "readout/summary.txt").is_file()
    with pytest.raises(FileExistsError, match="never overwrite"):
        readout.run_readout(pilot_dir, tmp_path, tmp_path / "readout", torch.device("cpu"), extra_clips=2)
    provenance_path = pilot_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["source_sha256"]["model/util/warp.py"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="Pilot source changed"):
        readout.verify_artifacts(pilot_dir)