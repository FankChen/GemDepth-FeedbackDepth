"""Matched TRAINED flat/full volume controls, not inference-only interventions.

Prepare one immutable data/order/initial-state contract. Run one registered arm
at a time; all declared predecessors must be completed and hash-verified before
the next arm starts. Never resume, train on dev, change old pilot files, select a
best-dev checkpoint, launch GRU, or substitute a different backbone.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.stereogru_matched import (ManifestQuotaError, build_manifest,  # noqa: E402
    content_sha256, file_sha256, load_manifest_clips, support_sha256,
    target_support, validate_manifest)
from loss.objective_registry import build_objective  # noqa: E402
from model.decoder_registry import build_decoder, get_decoder_class  # noqa: E402
from recover_verified_convnext_backbone import fingerprint_state, verify_base  # noqa: E402
from stereogru_calibrated_baseline import load_clean_backbone, predict, write_json  # noqa: E402

SOURCE_FILES = (
    "scripts/stereogru_matched_controls.py", "dataset/stereogru_matched.py",
    "dataset/stereogru_calibration.py", "dataset/vkitti_split.py",
    "model/dpt_calibrated_flat_volume_convnext.py", "model/dpt_calibrated_volume_only_convnext.py",
    "model/dpt_cost_volume_convnext.py", "model/util/cost_volume.py", "model/util/warp.py",
    "model/backbones.py", "model/backbone_registry.py", "model/decoder_registry.py", "model/util/lora.py",
    "loss/objective_calibrated_index.py", "loss/objective_registry.py", "loss/objective_video.py",
    "scripts/stereogru_calibrated_baseline.py", "scripts/recover_verified_convnext_backbone.py",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_state(path, state):
    with open(path, "xb") as handle:
        torch.save(state, handle)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def runtime_versions():
    import timm
    return {"torch": str(torch.__version__), "timm": timm.__version__,
            "numpy": np.__version__, "opencv": cv2.__version__}


def validate_config(cfg):
    if not 1 <= cfg["steps"] <= 1000 or not 0 < cfg["max_seconds_per_arm"] <= 3600:
        raise ValueError("This development control is bounded to 1000 updates and 3600 seconds per arm")
    if cfg["log_every"] < 1 or cfg["eval_every"] < 1 or set(cfg["dataset"]["splits"]) != {"train", "dev"}:
        raise ValueError("Need positive logging/evaluation periods and fixed train/dev splits")
    if cfg["decoder"]["kwargs"].get("iters", 0) != 0:
        raise ValueError("This entry has no GRU arms")
    if len(cfg["arms"]) != 2:
        raise ValueError("Expected exactly the preregistered baseline/control pair")
    previous = []
    for name, arm in cfg["arms"].items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or set(arm) != {"decoder", "requires"}:
            raise ValueError("An arm declares only its registered decoder and completion prerequisites")
        if arm["requires"] != previous:
            raise ValueError("Every later control requires all earlier arms; baseline-first cannot be bypassed")
        previous.append(name)
    expected_pair = {"C0": "DPTHeadCalibratedFlatVolumeConvNeXt",
                     "C1": "DPTHeadCalibratedVolumeOnlyConvNeXt"}
    if (tuple(cfg["arms"]) != tuple(expected_pair)
            or {name: arm["decoder"] for name, arm in cfg["arms"].items()} != expected_pair
            or cfg["decoder"]["initial_class"] != expected_pair["C1"]):
        raise ValueError("This protocol requires ordered C0=flat then C1=full and the shared full-head initial class")
    bounds = [cfg["decoder"]["kwargs"]["depth_min"], cfg["decoder"]["kwargs"]["depth_max"]]
    if bounds != [cfg["objective"]["kwargs"]["depth_min"], cfg["objective"]["kwargs"]["depth_max"]]:
        raise ValueError("Physical hypotheses and objective depth bounds differ")
    return cfg


def build_head(name, cfg, backbone_contract):
    head = build_decoder(get_decoder_class(name), cfg["decoder"]["kwargs"],
                         in_channels_list=backbone_contract["embed_dims"],
                         patch_size=backbone_contract["patch_size"])
    if head.iters != 0 or getattr(head, "output_space", None) != "normalized_inverse_depth_index":
        raise ValueError("Registered head violates the calibrated volume-only output contract")
    return head


def make_training_order(ids, steps, seed):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Training order needs unique fixed clip IDs")
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < steps:
        order.extend(str(ids[int(index)]) for index in rng.permutation(len(ids)))
    return order[:steps]


def prepare(config_path, weights, data_root, output):
    cfg = validate_config(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Prepare requires an empty NEW experiment directory; no overwrite or resume")
    start = time.monotonic()
    seed_everything(int(cfg["seed"]))
    write_json(output / "config.json", cfg)
    print("Preparing fixed train/dev quotas; no training has started.", flush=True)
    try:
        manifest = build_manifest(data_root, cfg["dataset"], cfg["decoder"]["kwargs"]["depth_min"],
                                  cfg["decoder"]["kwargs"]["depth_max"])
    except ManifestQuotaError as exc:
        write_json(output / "preflight_failure.json", {"status": "quota_failed", "counts": exc.report,
                                                       "training_started": False, "error": str(exc)})
        print(json.dumps(exc.report, indent=2), flush=True)
        raise
    write_json(output / "manifest.json", manifest)
    print("MANIFEST", json.dumps(manifest["selection_report"], sort_keys=True), flush=True)
    backbone, loading = load_clean_backbone(cfg["backbone"]["name"], weights)
    base_digest = verify_base(backbone.model.state_dict())
    backbone_contract = {"embed_dims": list(backbone.embed_dims), "patch_size": int(backbone.patch_size),
                         "feature_strides": list(backbone.feat_strides)}
    seed_everything(int(cfg["seed"]))
    initial = build_head(cfg["decoder"]["initial_class"], cfg, backbone_contract)
    state = {name: tensor.detach().cpu().clone() for name, tensor in initial.state_dict().items()}
    initial_digest = fingerprint_state(state)
    parameters = sum(parameter.numel() for parameter in initial.parameters())
    for arm in cfg["arms"].values():
        head = build_head(arm["decoder"], cfg, backbone_contract)
        head.load_state_dict(state, strict=True)
        if fingerprint_state(head.state_dict()) != initial_digest or sum(p.numel() for p in head.parameters()) != parameters:
            raise ValueError("Arms must share every initial parameter and BN buffer, and the same parameter count")
    save_state(output / "initial_head.pth", state)
    ids = [row["id"] for row in manifest["splits"]["train"]]
    write_json(output / "training_order.json", make_training_order(ids, int(cfg["steps"]), int(cfg["seed"])))
    contract_files = ("config.json", "manifest.json", "initial_head.pth", "training_order.json")
    contract = {
        "status": "prepared", "protocol": cfg["protocol"], "runtime": runtime_versions(),
        "files": {name: file_sha256(output / name) for name in contract_files},
        "source_sha256": {name: file_sha256(ROOT / name) for name in SOURCE_FILES},
        "backbone": {"path": str(Path(weights).resolve()), "file_sha256": file_sha256(weights),
                     "state_sha256": base_digest, "loading": loading, **backbone_contract},
        "initial_state_sha256": initial_digest, "head_parameters": parameters,
        "prepare_seconds": time.monotonic() - start,
        "interpretation": "GT-camera development subset, train scenes != dev scenes; no test-set or GRU/SOTA claim",
    }
    write_json(output / "experiment.json", contract)
    print(f"PREPARED: {output}; common initial={initial_digest}; head_parameters={parameters}", flush=True)
    return contract


def verify_experiment(experiment):
    root = Path(experiment).resolve()
    contract = read_json(root / "experiment.json")
    if contract["status"] != "prepared" or contract["runtime"] != runtime_versions():
        raise ValueError("Unprepared experiment or changed runtime versions; do not mix environments between arms")
    if set(contract["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("Source fingerprint inventory differs")
    for name, digest in contract["source_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Matched-control source changed: {name}")
    if set(contract["files"]) != {"config.json", "manifest.json", "initial_head.pth", "training_order.json"}:
        raise ValueError("Shared artifact inventory differs")
    for name, digest in contract["files"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Shared experiment artifact changed: {name}")
    cfg = validate_config(read_json(root / "config.json"))
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    bounds = [cfg["decoder"]["kwargs"]["depth_min"], cfg["decoder"]["kwargs"]["depth_max"]]
    if manifest["dataset"] != cfg["dataset"] or manifest["depth_bounds"] != bounds:
        raise ValueError("Manifest/config geometry or split mismatch")
    for path, digest in manifest["input_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"Matched input changed: {path}")
    if file_sha256(contract["backbone"]["path"]) != contract["backbone"]["file_sha256"]:
        raise ValueError("Clean backbone file changed")
    initial = torch.load(root / "initial_head.pth", map_location="cpu", weights_only=True)
    if fingerprint_state(initial) != contract["initial_state_sha256"]:
        raise ValueError("Common initial-state fingerprint differs")
    order = read_json(root / "training_order.json")
    expected = make_training_order([row["id"] for row in manifest["splits"]["train"]], cfg["steps"], cfg["seed"])
    if order != expected:
        raise ValueError("Training schedule changed or contains development clips")
    return root, cfg, manifest, contract, initial, order


def ensure_cache(root, cfg, manifest, contract, device, allow_create, deadline):
    check_deadline(deadline)
    path, metadata_path = root / "features.pth", root / "features.json"
    experiment_digest = file_sha256(root / "experiment.json")
    if not metadata_path.exists():
        if not allow_create or path.exists():
            raise ValueError("Missing/partial shared cache; do not regenerate after the baseline")
        start = time.monotonic()
        clips = load_manifest_clips(manifest, progress_check=lambda: check_deadline(deadline))
        check_deadline(deadline)
        backbone, _ = load_clean_backbone(cfg["backbone"]["name"], contract["backbone"]["path"])
        if verify_base(backbone.model.state_dict()) != contract["backbone"]["state_sha256"]:
            raise ValueError("Clean backbone state differs")
        backbone = backbone.to(device).eval()
        if backbone.training or any(p.requires_grad for p in backbone.parameters()):
            raise ValueError("Backbone must remain frozen AND eval")
        print(f"Caching {len(clips)} fixed clips once; the next arm will reuse exactly these tensors.", flush=True)
        for index, (clip_id, clip) in enumerate(clips.items(), start=1):
            check_deadline(deadline)
            with torch.no_grad():
                features = backbone(clip["images"].flatten(0, 1).to(device))
            if not all(torch.isfinite(value).all() for value in features):
                raise FloatingPointError(f"Nonfinite clean backbone features: {clip_id}")
            clip["features"] = [value.detach().cpu().float().contiguous() for value in features]
            check_deadline(deadline)
            if index % 8 == 0 or index == len(clips):
                print(f"CACHE {index}/{len(clips)}", flush=True)
        del backbone
        check_deadline(deadline)
        save_state(path, {"experiment_sha256": experiment_digest, "clips": clips})
        check_deadline(deadline)
        write_json(metadata_path, {"experiment_sha256": experiment_digest, "file_sha256": file_sha256(path),
                                   "clips": len(clips), "seconds": time.monotonic() - start})
        del clips
    check_deadline(deadline)
    metadata = read_json(metadata_path)
    if metadata["experiment_sha256"] != experiment_digest or metadata["file_sha256"] != file_sha256(path):
        raise ValueError("Shared feature-cache fingerprint mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    check_deadline(deadline)
    rows = {row["id"]: row for selected in manifest["splits"].values() for row in selected}
    if payload["experiment_sha256"] != experiment_digest or set(payload["clips"]) != set(rows):
        raise ValueError("Shared cache contains different train/dev clips")
    for clip_id, clip in payload["clips"].items():
        support = target_support(clip, *manifest["depth_bounds"])
        if support_sha256(support) != rows[clip_id]["support_sha256"]:
            raise ValueError(f"Cached GT support differs: {clip_id}")
    return payload["clips"], metadata


def on_device(clip, device):
    return {key: [value.to(device) for value in item] if key == "features" else item.to(device)
            for key, item in clip.items()}


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Per-arm safety budget exceeded; no completion certificate or next arm")


def aggregate_metrics(rows):
    count = sum(row["valid_pixels"] for row in rows)
    if count == 0:
        raise ValueError("Cannot evaluate empty support")
    return {"valid_pixels": count,
            "index_l1": sum(row["index_l1"] * row["valid_pixels"] for row in rows) / count,
            "absrel": sum(row["absrel"] * row["valid_pixels"] for row in rows) / count,
            "rmse": math.sqrt(sum(row["rmse"] ** 2 * row["valid_pixels"] for row in rows) / count),
            "delta1": sum(row["delta1"] * row["valid_pixels"] for row in rows) / count}


@torch.no_grad()
def evaluate(head, clips, manifest, objective, device, deadline):
    head.eval()
    result = {}
    for split, selected in manifest["splits"].items():
        rows = []
        for record in selected:
            check_deadline(deadline)
            clip = on_device(clips[record["id"]], device)
            q = predict(head, clip)
            if not torch.isfinite(q).all() or (q < -1e-6).any() or (q > 1. + 1e-6).any():
                raise FloatingPointError("Physical q output is nonfinite/outside [0,1]; no clamping fallback")
            loss = objective(q, clip["depth"], clip["mask"])["index_l1"]
            support = target_support(clip, objective.depth_min, objective.depth_max)
            if support_sha256(support) != record["support_sha256"]:
                raise ValueError("Evaluation support differs from fixed manifest")
            inverse = 1. / objective.depth_max + q * (1. / objective.depth_min - 1. / objective.depth_max)
            if (inverse <= 0).any():
                raise FloatingPointError("Nonpositive physical inverse-depth prediction")
            pred, gt = inverse[support].reciprocal().double(), clip["depth"][support].double()
            rows.append({"id": record["id"], "scene": record["scene"], "valid_pixels": int(support.sum()),
                         "index_l1": float(loss), "absrel": float(((pred - gt).abs() / gt).mean()),
                         "rmse": float((pred - gt).square().mean().sqrt()),
                         "delta1": float((torch.maximum(pred / gt, gt / pred) < 1.25).double().mean()),
                         "q_min": float(q.min()), "q_max": float(q.max()), **head.last_diagnostics})
        scenes = {scene: aggregate_metrics([row for row in rows if row["scene"] == scene])
                  for scene in sorted({row["scene"] for row in rows})}
        result[split] = {"overall": aggregate_metrics(rows), "scenes": scenes, "clips": rows}
    return result


def verify_completed(root, arm_id, cfg, contract, experiment_digest, certificate=None):
    """No C1 starts on a partial, foreign, changed or short-budget C0 artifact."""
    path = root / "arms" / arm_id
    completion_path = path / "completed.json"
    if certificate is None and not completion_path.is_file():
        raise ValueError(f"Required baseline {arm_id} has no completed certificate; next arm is blocked")
    completed = read_json(completion_path) if certificate is None else certificate
    if (completed.get("status") != "completed" or completed.get("arm") != arm_id
            or completed.get("steps") != cfg["steps"] or completed.get("experiment_sha256") != experiment_digest
            or completed.get("initial_state_sha256") != contract["initial_state_sha256"]
            or completed.get("initial_file_sha256") != contract["files"]["initial_head.pth"]
            or completed.get("manifest_sha256") != contract["files"]["manifest.json"]
            or completed.get("order_sha256") != contract["files"]["training_order.json"]
            or completed.get("decoder") != cfg["arms"][arm_id]["decoder"]
            or completed.get("head_parameters") != contract["head_parameters"]
            or completed.get("optimizer_initial_state_entries") != 0
            or completed.get("runtime") != contract["runtime"]):
        raise ValueError(f"Baseline completion contract mismatch: {arm_id}")
    if set(completed["files"]) != {"final_head.pth", "metrics_final.json", "progress.jsonl", "initialisation.json"}:
        raise ValueError("Incomplete baseline artifact inventory")
    for name, digest in completed["files"].items():
        if file_sha256(path / name) != digest:
            raise ValueError(f"Completed baseline artifact changed: {arm_id}/{name}")
    initialisation = read_json(path / "initialisation.json")
    for name in ("arm", "decoder", "initial_state_sha256", "initial_file_sha256",
                 "head_parameters", "optimizer_initial_state_entries"):
        if initialisation[name] != completed[name]:
            raise ValueError(f"Baseline initialisation evidence differs: {name}")
    for required in cfg["arms"][arm_id]["requires"]:
        if initialisation["prerequisite_certificate_sha256"].get(required) != file_sha256(root / "arms" / required / "completed.json"):
            raise ValueError("Prerequisite certificate changed after the later arm was started")
    cache_metadata = read_json(root / "features.json")
    if completed["cache_sha256"] != cache_metadata["file_sha256"] or file_sha256(root / "features.pth") != completed["cache_sha256"]:
        raise ValueError("Baseline feature cache changed")
    payload = torch.load(path / "final_head.pth", map_location="cpu", weights_only=True)
    if (payload["arm"] != arm_id or payload["steps"] != cfg["steps"]
            or payload["experiment_sha256"] != experiment_digest
            or payload["initial_state_sha256"] != contract["initial_state_sha256"]
            or fingerprint_state(payload["model_state_dict"]) != completed["final_state_sha256"]):
        raise ValueError("Baseline checkpoint metadata/state differs")
    head = build_head(cfg["arms"][arm_id]["decoder"], cfg, contract["backbone"])
    head.load_state_dict(payload["model_state_dict"], strict=True)
    metrics = read_json(path / "metrics_final.json")
    if metrics["steps"] != cfg["steps"] or metrics["arm"] != arm_id:
        raise ValueError("Baseline final metrics use a different budget/arm")
    manifest = read_json(root / "manifest.json")
    if set(metrics["metrics"]) != set(manifest["splits"]):
        raise ValueError("Baseline final metrics omit a split")
    for split, expected in manifest["splits"].items():
        rows = metrics["metrics"][split]["clips"]
        if ([(row["id"], row["scene"], row["valid_pixels"]) for row in rows]
                != [(row["id"], row["scene"], row["valid_pixels"]) for row in expected]):
            raise ValueError("Baseline metrics contain different clips or GT support")
        if not all(math.isfinite(row[key]) for row in rows for key in ("index_l1", "absrel", "rmse", "delta1")):
            raise ValueError("Nonfinite baseline metric")
        if aggregate_metrics(rows) != metrics["metrics"][split]["overall"]:
            raise ValueError("Baseline metric aggregation differs")
    return completed


def train_arm(experiment, arm_id, device):
    root, cfg, manifest, contract, initial, order = verify_experiment(experiment)
    if arm_id not in cfg["arms"]:
        raise ValueError(f"Unknown arm {arm_id}; options={list(cfg['arms'])}")
    experiment_digest = file_sha256(root / "experiment.json")
    specification = cfg["arms"][arm_id]
    prerequisite_hashes = {}
    for required in specification["requires"]:
        verify_completed(root, required, cfg, contract, experiment_digest)
        prerequisite_hashes[required] = file_sha256(root / "arms" / required / "completed.json")
    output = root / "arms" / arm_id
    if output.exists():
        raise FileExistsError(f"Arm output exists: {output}; no resume/overwrite/reuse of final heads")
    output.mkdir(parents=True)
    start = time.monotonic()
    deadline = start + float(cfg["max_seconds_per_arm"])
    try:
        clips, cache_metadata = ensure_cache(root, cfg, manifest, contract, device,
                                             allow_create=not specification["requires"], deadline=deadline)
        check_deadline(deadline)
        seed_everything(int(cfg["seed"]))
        head = build_head(specification["decoder"], cfg, contract["backbone"])
        head.load_state_dict(initial, strict=True)  # Always INITIAL, never C0 final.
        if fingerprint_state(head.state_dict()) != contract["initial_state_sha256"]:
            raise ValueError("Arm did not load the exact common initial state")
        head = head.to(device)
        objective = build_objective(cfg["objective"]["name"], cfg["objective"]["kwargs"]).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["optimizer"]["learning_rate"],
                                     weight_decay=cfg["optimizer"]["weight_decay"])
        initialisation = {"arm": arm_id, "decoder": specification["decoder"],
                          "initial_state_sha256": contract["initial_state_sha256"],
                          "initial_file_sha256": contract["files"]["initial_head.pth"],
                          "head_parameters": sum(p.numel() for p in head.parameters()),
                          "optimizer_initial_state_entries": len(optimizer.state),
                          "prerequisite_certificate_sha256": prerequisite_hashes}
        write_json(output / "initialisation.json", initialisation)
        initial_metrics = evaluate(head, clips, manifest, objective, device, deadline)
        write_json(output / "metrics_initial.json", {"arm": arm_id, "steps": 0, "metrics": initial_metrics})
        print(f"INITIAL {arm_id}", json.dumps({s: v["overall"] for s, v in initial_metrics.items()}), flush=True)
        loop_start = time.monotonic()
        first_matcher_grad = None
        final_metrics = None
        with open(output / "progress.jsonl", "x") as log:
            for step, clip_id in enumerate(order, start=1):
                check_deadline(deadline)
                head.train()
                clip = on_device(clips[clip_id], device)
                optimizer.zero_grad(set_to_none=True)
                q = predict(head, clip)
                loss = objective(q, clip["depth"], clip["mask"])["total_loss"]
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["optimizer"]["grad_clip"], error_if_nonfinite=True)
                if first_matcher_grad is None:
                    first_matcher_grad = sum(float(p.grad.detach().square().sum()) for p in head.matcher.parameters() if p.grad is not None) ** .5
                    if not math.isfinite(first_matcher_grad) or first_matcher_grad == 0:
                        raise ValueError("No finite nonzero matcher gradient in calibrated control")
                optimizer.step()
                elapsed = time.monotonic() - loop_start
                if step == 1 or step % cfg["log_every"] == 0 or step == cfg["steps"]:
                    row = {"arm": arm_id, "step": step, "clip_id": clip_id, "loss": float(loss.detach()),
                           "grad_norm_before_clip": float(norm.detach()), "loop_elapsed_seconds": elapsed,
                           "estimated_remaining_loop_seconds": elapsed / step * (cfg["steps"] - step),
                           **head.last_diagnostics}
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
                    final_metrics = evaluate(head, clips, manifest, objective, device, deadline)
                    write_json(output / f"metrics_step_{step:06d}.json", {"arm": arm_id, "steps": step, "metrics": final_metrics})
                    print(f"EVAL {arm_id} {step}", json.dumps({s: v["overall"] for s, v in final_metrics.items()}), flush=True)
        check_deadline(deadline)
        write_json(output / "metrics_final.json", {"arm": arm_id, "steps": cfg["steps"], "metrics": final_metrics,
                                                   "aggregation": "fixed-mask pixel-weighted metrics per scene; no alignment; final-step only"})
        final_state = {name: value.detach().cpu() for name, value in head.state_dict().items()}
        final_digest = fingerprint_state(final_state)
        save_state(output / "final_head.pth", {"arm": arm_id, "steps": cfg["steps"],
                   "experiment_sha256": experiment_digest, "initial_state_sha256": contract["initial_state_sha256"],
                   "model_state_dict": final_state})
        files = ("final_head.pth", "metrics_final.json", "progress.jsonl", "initialisation.json")
        completed = {"status": "completed", "arm": arm_id, "decoder": specification["decoder"],
                     "steps": cfg["steps"], "experiment_sha256": experiment_digest,
                     "initial_state_sha256": contract["initial_state_sha256"],
                     "initial_file_sha256": contract["files"]["initial_head.pth"],
                     "manifest_sha256": contract["files"]["manifest.json"],
                     "order_sha256": contract["files"]["training_order.json"],
                     "cache_sha256": cache_metadata["file_sha256"],
                     "head_parameters": initialisation["head_parameters"], "optimizer_initial_state_entries": 0,
                     "final_state_sha256": final_digest, "files": {name: file_sha256(output / name) for name in files},
                     "first_matcher_grad_after_clip": first_matcher_grad, "elapsed_seconds": time.monotonic() - start,
                     "runtime": runtime_versions(), "device": str(device), "gru_launched": False}
        verify_completed(root, arm_id, cfg, contract, experiment_digest, certificate=completed)
        check_deadline(deadline)
        write_json(output / "completed.json", completed)  # Written LAST, only after verification.
        print(f"COMPLETED {arm_id}: {output}; no next arm was launched.", flush=True)
        return completed
    except Exception as exc:
        if not (output / "completed.json").exists():
            write_json(output / "failure.json", {"status": "failed", "arm": arm_id, "error": str(exc),
                                                "next_arm_allowed": False})
        raise


def compare(experiment):
    root, cfg, _, contract, _, _ = verify_experiment(experiment)
    digest = file_sha256(root / "experiment.json")
    results = {}
    for arm_id in cfg["arms"]:
        certificate = verify_completed(root, arm_id, cfg, contract, digest)
        results[arm_id] = {"certificate_sha256": file_sha256(root / "arms" / arm_id / "completed.json"),
                           "steps": certificate["steps"],
                           "metrics": read_json(root / "arms" / arm_id / "metrics_final.json")["metrics"]}
    report = {"protocol": cfg["protocol"], "experiment_sha256": digest,
              "matched_initialisation_manifest_order_cache": True, "results": results,
              "interpretation": "Calibrated development comparison, final-step only; no automatic winner, GRU or RGB-only/SOTA claim"}
    if not (root / "comparison.json").exists():
        write_json(root / "comparison.json", report)
    elif read_json(root / "comparison.json") != report:
        raise ValueError("Existing comparison differs; refusing overwrite")
    print("arm / split / scene / index_L1 / AbsRel / RMSE / delta1")
    for arm_id, result in results.items():
        for split, values in result["metrics"].items():
            for scene, item in values["scenes"].items():
                print(f"{arm_id} {split} {scene} {item['index_l1']:.6f} {item['absrel']:.6f} {item['rmse']:.4f} {item['delta1']:.6f}")
    print(report["interpretation"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--config", default=str(ROOT / "config/stereogru/matched_c0_c1.yaml"))
    setup.add_argument("--backbone-weights", required=True)
    setup.add_argument("--vkitti-root", required=True)
    setup.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--experiment", required=True)
    train.add_argument("--arm", required=True)
    check = commands.add_parser("compare")
    check.add_argument("--experiment", required=True)
    args = parser.parse_args()
    actions = {"prepare": lambda: prepare(args.config, args.backbone_weights, args.vkitti_root, args.output),
               "train": lambda: train_arm(args.experiment, args.arm, torch.device("cuda")),
               "compare": lambda: compare(args.experiment)}
    if args.command == "train" and not torch.cuda.is_available():
        raise RuntimeError("Real matched training requires an idle CUDA GPU; CPU tests are separate")
    actions[args.command]()


if __name__ == "__main__":
    main()