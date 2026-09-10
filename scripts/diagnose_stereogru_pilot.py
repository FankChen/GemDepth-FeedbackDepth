"""Read-only follow-up to a COMPLETED calibrated volume-only overfit pilot.

First reproduce its saved final metrics on the exact training clips and hashes.
Then intervene on the raw-volume input with weights, guides and GT masks fixed,
and probe additional, non-overlapping clips from the TRAINING scene pool. These
are dependency probes, not retrained ablation arms or a generalisation benchmark.
No optimiser, checkpoint writes, BN updates, resuming or model selection.
"""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.stereogru_calibration import load_training_clips  # noqa: E402
from loss.objective_registry import build_objective  # noqa: E402
from model.decoder_registry import build_decoder, get_decoder_class  # noqa: E402
from recover_verified_convnext_backbone import fingerprint_state, verify_base  # noqa: E402
from stereogru_calibrated_baseline import (evaluate_overfit, load_clean_backbone,  # noqa: E402
                                          sha256, write_json)


RAW_INTERVENTIONS = {}


def register_raw_intervention(name):
    def register(function):
        if name in RAW_INTERVENTIONS:
            raise ValueError(f"Duplicate read-only intervention {name}")
        RAW_INTERVENTIONS[name] = function
        return function
    return register


@register_raw_intervention("normal")
def raw_identity(raw):
    return raw


@register_raw_intervention("zero_raw")
def raw_zero(raw):
    return torch.zeros_like(raw)


@register_raw_intervention("flat_depth_raw")
def raw_depth_mean(raw):
    # Retain each pixel/group mean but remove variation over metric depth bins.
    return raw.mean(dim=2, keepdim=True).expand_as(raw).contiguous()


@register_raw_intervention("reverse_depth_raw")
def raw_depth_reverse(raw):
    # Preserve the values/histogram but assign them to the wrong depth indices.
    return raw.flip(dims=(2,))


@contextmanager
def observe_intervention(head, name):
    """Replace only volume_stem input; remove both hooks even after an error."""
    if head.training or any(module.training for module in head.modules()):
        raise ValueError("Readout must run entirely in eval mode; no BN recalibration")
    transform = RAW_INTERVENTIONS[name]
    raw_stats, outputs = [], []

    def replace_raw(_module, args):
        raw = args[0]
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Nonfinite original raw volume; do not hide it with zeroing")
        replacement = transform(raw)
        if replacement.shape != raw.shape:
            raise ValueError("Raw intervention changed tensor shape")
        raw_stats.append({
            "zero_fraction": float((replacement.abs().sum(dim=(1, 2)) == 0).float().mean()),
            "depth_std": float(replacement.std(dim=2, unbiased=False).mean()),
        })
        return (replacement, *args[1:])

    def record_output(_module, _args, output):
        outputs.append(output.detach().cpu().clone())

    handles = [head.volume_stem.register_forward_pre_hook(replace_raw),
               head.register_forward_hook(record_output)]
    try:
        yield raw_stats, outputs
    finally:
        for handle in handles:
            handle.remove()


def assert_metric_replay(actual, expected):
    """Do not analyse a different crop/checkpoint as if it replayed the pilot."""
    failures = []
    for key in ("index_l1", "absrel", "rmse", "delta1"):
        if not math.isclose(actual[key], expected[key], rel_tol=1e-4, abs_tol=1e-6):
            failures.append(f"{key}: {actual[key]} != {expected[key]}")
    if actual["valid_pixels"] != expected["valid_pixels"]:
        failures.append("valid-pixel count changed")
    if failures:
        raise ValueError("Saved pilot final metrics did not replay; stopping before interventions: " + "; ".join(failures))


@torch.no_grad()
def measure(head, clips, objective, mode):
    with observe_intervention(head, mode) as (raw_stats, outputs):
        result = evaluate_overfit(head, clips, objective)
    if len(raw_stats) != len(clips) or len(outputs) != len(clips):
        raise RuntimeError("Readout did not observe one raw volume/output per clip")
    for row, effective in zip(result["clips"], raw_stats):
        # The head's raw_* fields describe its unmodified volume BEFORE our hook.
        # Expose the actual aggregation input separately; never confuse the two.
        row["effective_raw_input"] = effective
    return result, outputs


@torch.no_grad()
def diagnose_groups(head, groups, objective, expected_fit_metrics):
    head.eval()
    state_before = fingerprint_state(head.state_dict())
    results = {}
    for label, clips in groups.items():
        normal, reference_outputs = measure(head, clips, objective, "normal")
        if label == "fitted":
            assert_metric_replay(normal, expected_fit_metrics)
        modes = {"normal": normal}
        for mode in RAW_INTERVENTIONS:
            if mode == "normal":
                continue
            result, outputs = measure(head, clips, objective, mode)
            if ([row["valid_pixels"] for row in result["clips"]]
                    != [row["valid_pixels"] for row in normal["clips"]]):
                raise RuntimeError("Intervention changed GT support; comparisons must use identical pixels")
            differences = [(value - reference).abs() for value, reference in zip(outputs, reference_outputs)]
            result["all_pixel_mean_abs_delta_q"] = (
                sum(float(value.sum()) for value in differences) / sum(value.numel() for value in differences))
            result["all_pixel_max_abs_delta_q"] = max(float(value.max()) for value in differences)
            modes[mode] = result
        results[label] = modes
    if fingerprint_state(head.state_dict()) != state_before:
        raise RuntimeError("Readout modified model parameters/buffers; result rejected")
    return results, state_before


def verify_artifacts(pilot_dir):
    root = Path(pilot_dir).resolve()
    cfg_dict = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (summary["status"] != "completed_implementation_pilot"
            or summary["method_launched"] or summary["steps"] != cfg_dict["steps"]
            or summary["clips"] != len(manifest)):
        raise ValueError("Expected a completed, internally consistent volume-only pilot")
    # No model/data/evaluator changes are permitted during a numerical replay.
    for relative, digest in provenance["source_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != digest:
            raise ValueError(f"Pilot source changed: {relative}")
    expected_inputs = {p for row in manifest for key in ("rgb", "depth", "calibration") for p in row[key]}
    if set(provenance["input_sha256"]) != expected_inputs:
        raise ValueError("Pilot manifest and input fingerprint table differ")
    for path, digest in provenance["input_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Pilot input changed: {path}")
    if sha256(provenance["weights"]) != provenance["weights_sha256"]:
        raise ValueError("Saved clean backbone file changed")
    checkpoint = torch.load(root / "baseline_head.pth", map_location="cpu", weights_only=True)
    if (checkpoint["config"] != cfg_dict or checkpoint["manifest"] != manifest
            or checkpoint["steps"] != summary["steps"]):
        raise ValueError("Checkpoint metadata differs from saved pilot artifacts")
    return OmegaConf.create(cfg_dict), manifest, provenance, summary, checkpoint


def group_clips(clips, fitted_manifest):
    count = len(fitted_manifest)
    if [clip["manifest"] for clip in clips[:count]] != fitted_manifest:
        raise ValueError("Training clip/camera/crop replay differs from the stored manifest")
    fitted_paths = {path for row in fitted_manifest for path in row["rgb"]}
    fitted_scenes = {row["scene"] for row in fitted_manifest}
    if any(path in fitted_paths for clip in clips[count:] for path in clip["manifest"]["rgb"]):
        raise ValueError("Additional clips overlap frames used to fit the pilot")
    groups = {"fitted": clips[:count]}
    unseen_scene = [clip for clip in clips[count:] if clip["manifest"]["scene"] not in fitted_scenes]
    same_scene = [clip for clip in clips[count:] if clip["manifest"]["scene"] in fitted_scenes]
    if unseen_scene:
        groups["unfitted_training_scene"] = unseen_scene
    if same_scene:
        groups["unfitted_same_scene"] = same_scene
    return groups


def run_readout(pilot_dir, vkitti_root, output, device, extra_clips=6):
    if not 1 <= extra_clips <= 16:
        raise ValueError("Readout is bounded to 1..16 additional clips")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Readout output must be new/empty; never overwrite pilot artifacts")
    checkpoint_path = Path(pilot_dir) / "baseline_head.pth"
    checkpoint_file_digest = sha256(checkpoint_path)
    cfg, manifest, provenance, summary, checkpoint = verify_artifacts(pilot_dir)
    torch.manual_seed(int(cfg.seed))
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    backbone, loading = load_clean_backbone(str(cfg.backbone.name), provenance["weights"])
    verify_base(backbone.model.state_dict())
    backbone = backbone.to(device).eval()
    kwargs = dict(cfg.dataset)
    kwargs["num_clips"] = len(manifest) + extra_clips
    clips = load_training_clips(vkitti_root, **kwargs, seed=int(cfg.seed))
    groups = group_clips(clips, manifest)
    for clip in clips:
        for key in ("images", "depth", "mask", "intrinsics", "extrinsics"):
            clip[key] = clip[key].to(device)
        with torch.no_grad():
            clip["features"] = [value.detach().float() for value in backbone(clip["images"].flatten(0, 1))]
    head = build_decoder(get_decoder_class(str(cfg.decoder.name)), dict(cfg.decoder.kwargs),
                         in_channels_list=backbone.embed_dims, patch_size=backbone.patch_size).to(device).eval()
    head.load_state_dict(checkpoint["model_state_dict"], strict=True)
    objective = build_objective(str(cfg.objective.name), dict(cfg.objective.kwargs)).to(device)
    if head.iters != 0 or (head.depth_min, head.depth_max) != (objective.depth_min, objective.depth_max):
        raise ValueError("Readout only supports the calibrated volume-only baseline with matching bounds")
    del backbone, checkpoint
    results, head_state_digest = diagnose_groups(head, groups, objective, summary["final"])
    import timm
    from collections import defaultdict
    fitted_frames = defaultdict(set)
    for row in manifest:
        fitted_frames[row["scene"]].update(row["frames"])
    grouped_manifest = {}
    for label, selected in groups.items():
        grouped_manifest[label] = []
        for clip in selected:
            row = dict(clip["manifest"])
            old_frames = fitted_frames[row["scene"]]
            row["minimum_frame_index_gap_to_fitted"] = (
                min(abs(a - b) for a in row["frames"] for b in old_frames) if old_frames else None)
            grouped_manifest[label].append(row)
    inputs = {p for clip in clips for key in ("rgb", "depth", "calibration") for p in clip["manifest"][key]}
    if sha256(checkpoint_path) != checkpoint_file_digest:
        raise RuntimeError("Pilot checkpoint file changed during readout; result rejected")
    report = {
        "status": "completed_read_only_dependency_probe", "pilot": str(Path(pilot_dir).resolve()),
        "pilot_replayed": True, "training_steps": 0, "method_launched": False,
        "head_state_unchanged": True, "head_state_sha256": head_state_digest,
        "baseline_file_sha256": checkpoint_file_digest,
        "baseline_file_unchanged_during_readout": True,
        "baseline_identity_caveat": (
            "The original pilot saved no independent final-head fingerprint. This digest "
            "identifies the currently loaded file and verifies it stayed unchanged; metadata "
            "and numerical metric replay are checked, not historical byte identity."),
        "clean_backbone_file_sha256": provenance["weights_sha256"], "loading": loading,
        "torch": torch.__version__, "timm": timm.__version__,
        "pilot_torch": provenance["torch"], "pilot_timm": provenance["timm"],
        "readout_source_sha256": sha256(Path(__file__)),
        "input_sha256": {p: sha256(p) for p in sorted(inputs)},
        "groups": grouped_manifest, "results": results,
        "interpretation": [
            "All modes reuse exactly the same head, cameras, image guides, depth bounds and GT masks; only the volume_stem input is intervened on.",
            "Head raw_* statistics are pre-intervention; effective_raw_input describes the actual replaced aggregation input.",
            "No optimiser or BN updates. Intervention damage demonstrates sensitivity, not by itself correct triangulation or retrained method gains.",
            "Additional clips are non-overlapping frames from the training scene pool, motion-selected before looking at loss. Same-scene clips may be adjacent; unseen training-scene samples are listed separately, not an official test set.",
            "Tiny overfit success is not a completed full baseline or an RGB-only/SOTA result. Review dependency and unfitted performance before scaling training; no method is auto-launched.",
        ],
    }
    write_json(output / "readout.json", report)
    lines = ["PILOT_REPLAYED: true; training_steps=0; head_state_unchanged=true",
             "group / mode / clips / index_L1 / AbsRel / RMSE / delta1 / mean_abs_delta_q"]
    for label, modes in results.items():
        for mode, item in modes.items():
            lines.append(f"{label:24s} {mode:18s} {len(item['clips']):2d} "
                         f"{item['index_l1']:.6f} {item['absrel']:.6f} {item['rmse']:.4f} "
                         f"{item['delta1']:.6f} {item.get('all_pixel_mean_abs_delta_q', 0.):.6f}")
    lines += ["Not retrained ablations or a benchmark; see readout.json for fixed-support and scene-gap details.",
              f"Output directory: {output}"]
    text = "\n".join(lines)
    with open(output / "summary.txt", "x") as handle:
        handle.write(text + "\n")
    print(text, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-run", required=True)
    parser.add_argument("--vkitti-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-clips", type=int, default=6)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real checkpoint readout requires an idle GPU; CPU fixtures are separate")
    run_readout(args.pilot_run, args.vkitti_root, args.output, torch.device("cuda"), args.extra_clips)


if __name__ == "__main__":
    main()