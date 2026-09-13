"""Independent, baseline-first sequence-supervision experiment.

Only new files are written. Historical C0/C1 are read-only sources of certified
data/cache/order/INITIAL tensors, not substitutes for the new B0/F1 controls.
Model/objective selection is registry/config driven; one optimiser loop serves
every arm. Partial runs are never resumed or silently retried.
"""

import argparse
import copy
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Third-party/module-discovery notices must never contaminate CLI control output.
with redirect_stdout(sys.stderr):
    import torch
    from omegaconf import OmegaConf

    import stereogru_corrected_gru as legacy
    import stereogru_matched_controls as controls
    import stereogru_matched_repeats as repeats
    from loss.objective_registry import build_objective
    from model.decoder_registry import build_decoder, get_decoder_class


SEEDS = (0, 1, 2)
DEFAULT_CONFIG = ROOT / "config/stereogru/sequence_supervision.yaml"
NEW_SOURCES = (
    "model/dpt_calibrated_gru_sequence_convnext.py", "loss/objective_calibrated_sequence.py",
    "scripts/stereogru_sequence_experiment.py", "scripts/run_stereogru_sequence_experiment.sh",
    "config/stereogru/sequence_supervision.yaml",
)
SOURCES = tuple(sorted(set(controls.SOURCE_FILES + legacy.METHOD_SOURCES + NEW_SOURCES)))
COMMON_ARTIFACTS = {"initialisation.json", "verification.json", "evaluation_initial.json",
                    "evaluation_final.json", "metrics_final.json", "progress.jsonl", "trace.jsonl"}


def source_inventory():
    """Bind whole local code/config trees, including auto-discovered plugins.

    Re-enumeration detects added/removed modules as well as changed contents.
    Third-party packages are runtime dependencies, not claimed as hashed sources.
    Core train.py is intentionally not part of this standalone executable path.
    """
    files = set(SOURCES)
    for tree in ("model", "loss", "dataset", "scripts", "config"):
        files.update(str(path.relative_to(ROOT)) for path in (ROOT / tree).rglob("*")
                     if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".yml"})
    return tuple(sorted(files))


def method_config(path):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    expected_arms = {
        "B0": {"decoder": "DPTHeadCalibratedVolumeSequenceConvNeXt", "iterations": 0,
               "objective": "calibrated_igev_sequence_index_l1", "objective_kwargs": {"iterations": 0, "gamma": .9},
               "requires": [], "gate_required": False},
        "F1": {"decoder": "DPTHeadCalibratedGRUSequenceConvNeXt", "iterations": 8,
               "objective": "calibrated_final_sequence_index_l1", "objective_kwargs": {"iterations": 8},
               "requires": ["B0"], "gate_required": True},
        "S1": {"decoder": "DPTHeadCalibratedGRUSequenceConvNeXt", "iterations": 8,
               "objective": "calibrated_igev_sequence_index_l1", "objective_kwargs": {"iterations": 8, "gamma": .9},
               "requires": ["B0", "F1"], "gate_required": True},
    }
    if (set(cfg) != {"protocol", "seeds", "index_update", "weight_contract", "evaluation", "gate",
                     "gate_reset_required", "arms"}
            or cfg["protocol"] != "calibrated_gru_sequence_supervision_v1"
            or cfg["seeds"] != list(SEEDS) or cfg["index_update"] != "remaining_range_tanh_v1"
            or cfg["weight_contract"] != "igev_mvs_relative_gamma09_sum_normalized"
            or cfg["evaluation"] != "final_physical_q_no_alignment" or cfg["gate_reset_required"] is not True
            or tuple(cfg["arms"]) != tuple(expected_arms) or cfg["arms"] != expected_arms):
        raise ValueError("Sequence experiment contract changed; baseline/arms/weights cannot be bypassed")
    gate = cfg["gate"]
    if (set(gate) != {"steps", "clips_per_train_scene", "maximum_final_initial_loss_ratio"}
            or isinstance(gate["steps"], bool) or not isinstance(gate["steps"], int)
            or not 1 <= gate["steps"] <= 200 or gate["clips_per_train_scene"] != 1
            or gate["maximum_final_initial_loss_ratio"] != .5):
        raise ValueError("Fixed training-only gate requires 1..200 steps, one clip/scene and ratio<=0.5")
    return cfg


def load_baselines(repetitions):
    root = Path(repetitions).resolve()
    inventory = controls.read_json(root / "repetitions.json")
    summary = controls.read_json(root / "seed_summary.json")
    if (inventory["seeds"] != list(SEEDS) or set(inventory["experiments"]) != {str(s) for s in SEEDS}
            or summary["status"] != "three_paired_seeds_complete" or summary["seeds"] != list(SEEDS)):
        raise ValueError("All three certified source C0/C1 pairs are required")
    pairs = {}
    for seed in SEEDS:
        item = inventory["experiments"][str(seed)]
        pair = repeats.verify_pair(item["experiment"])
        recorded = summary["experiments"][str(seed)]
        if (pair["experiment_sha256"] != item["experiment_sha256"]
                or pair["experiment_sha256"] != recorded["contract_sha256"]
                or pair["completions"] != recorded["completion_sha256"]
                or (seed == 0 and pair["completions"] != item["completion_sha256"])):
            raise ValueError("Historical source evidence changed")
        pairs[seed] = pair
        repeats.assert_same_protocol(pairs[0], pair, seed)
    legacy.validate_baseline_summary(inventory, summary, pairs)
    if len({pair["contract"]["initial_state_sha256"] for pair in pairs.values()}) != len(SEEDS):
        raise ValueError("Different source seeds must have different common initial states")
    return pairs


def objective_for(pair, specification):
    bounds = {key: pair["config"]["decoder"]["kwargs"][key] for key in ("depth_min", "depth_max")}
    return build_objective(specification["objective"], {**bounds, **specification["objective_kwargs"]})


def new_head(pair, specification, shared):
    controls.seed_everything(int(pair["config"]["seed"]))
    kwargs = {**pair["config"]["decoder"]["kwargs"], "iters": specification["iterations"]}
    head = build_decoder(get_decoder_class(specification["decoder"]), kwargs,
                         in_channels_list=pair["contract"]["backbone"]["embed_dims"],
                         patch_size=pair["contract"]["backbone"]["patch_size"])
    head.load_c1_initial(shared)
    if (getattr(head, "output_space", None) != "normalized_inverse_depth_index"
            or head.iters != specification["iterations"]
            or controls.fingerprint_state({key: head.state_dict()[key] for key in shared})
            != pair["contract"]["initial_state_sha256"]):
        raise ValueError("Head does not preserve the declared physical output/common INITIAL tensors")
    return head


def prepare(repetitions, output, config_path=DEFAULT_CONFIG):
    cfg = method_config(config_path)
    pairs = load_baselines(repetitions)
    root = Path(output).resolve()
    protected = [Path(repetitions).resolve()] + [pair["root"] for pair in pairs.values()]
    if any(root == path or root.is_relative_to(path) for path in protected):
        raise ValueError("New experiment must be outside all historical source directories")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Prepare needs a NEW empty directory; no overwrite/resume")
    controls.write_json(root / "method.json", cfg)
    entries = {}
    for seed, pair in pairs.items():
        destination = root / f"seed{seed}"
        destination.mkdir()
        shared = torch.load(pair["root"] / "initial_head.pth", map_location="cpu", weights_only=True)
        cache = controls.read_json(pair["root"] / "features.json")
        entry = {
            "source_experiment": str(pair["root"]), "source_contract_sha256": pair["experiment_sha256"],
            "source_completion_sha256": pair["completions"],
            "manifest_sha256": pair["contract"]["files"]["manifest.json"],
            "order_sha256": pair["contract"]["files"]["training_order.json"],
            "common_initial_state_sha256": pair["contract"]["initial_state_sha256"],
            "cache_sha256": cache["file_sha256"], "cache_metadata_sha256": controls.file_sha256(pair["root"] / "features.json"),
            "arms": {},
        }
        saved = {}
        for arm, spec in cfg["arms"].items():
            head = new_head(pair, spec, shared)
            state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            state_hash = controls.fingerprint_state(state)
            # F1/S1 use the same saved file, not two vaguely "similar" seeds.
            if state_hash not in saved:
                path = destination / f"initial_{arm}.pth"
                controls.save_state(path, state)
                saved[state_hash] = str(path.relative_to(root))
            objective = objective_for(pair, spec)
            entry["arms"][arm] = {
                "initial_file": saved[state_hash], "initial_file_sha256": controls.file_sha256(root / saved[state_hash]),
                "initial_state_sha256": state_hash, "parameters": sum(p.numel() for p in head.parameters()),
                "raw_weights": list(objective.raw_weights), "weights": list(objective.weights),
                "required_gradients": list(head.required_gradient_modules),
                "added_state_keys": sorted(set(state) - set(shared)),
            }
        if (entry["arms"]["F1"]["initial_file"] != entry["arms"]["S1"]["initial_file"]
                or entry["arms"]["B0"]["parameters"] != pair["contract"]["head_parameters"]):
            raise ValueError("Controlled arms do not have the promised shared initialization/capacity")
        entries[str(seed)] = entry
    repetitions = Path(repetitions).resolve()
    contract = {"status": "prepared_sequence_experiment", "seeds": list(SEEDS),
                "method_sha256": controls.file_sha256(root / "method.json"), "runtime": controls.runtime_versions(),
                "sources": {name: controls.file_sha256(ROOT / name) for name in source_inventory()},
                "repetitions": str(repetitions),
                "repetitions_sha256": controls.file_sha256(repetitions / "repetitions.json"),
                "baseline_summary_sha256": controls.file_sha256(repetitions / "seed_summary.json"),
                "seed_entries": entries}
    controls.write_json(root / "experiment.json", contract)
    print("SEQUENCE_PREPARED", root, flush=True)
    print("B0 -> F1 -> S1; all seeds 0/1/2; gates discarded; INITIAL reset; no training yet.", flush=True)
    print("WEIGHTS", json.dumps({arm: value["weights"] for arm, value in entries["0"]["arms"].items()}), flush=True)
    return contract


def verify(output):
    root = Path(output).resolve()
    contract = controls.read_json(root / "experiment.json")
    if (contract["status"] != "prepared_sequence_experiment" or contract["seeds"] != list(SEEDS)
            or set(contract["seed_entries"]) != {str(s) for s in SEEDS}
            or contract["runtime"] != controls.runtime_versions()
            or controls.file_sha256(root / "method.json") != contract["method_sha256"]
            or set(contract["sources"]) != set(source_inventory())):
        raise ValueError("Sequence preparation/config/runtime/source inventory changed")
    for name, digest in contract["sources"].items():
        if controls.file_sha256(ROOT / name) != digest:
            raise ValueError(f"Sequence source changed: {name}")
    source = Path(contract["repetitions"])
    if (controls.file_sha256(source / "repetitions.json") != contract["repetitions_sha256"]
            or controls.file_sha256(source / "seed_summary.json") != contract["baseline_summary_sha256"]):
        raise ValueError("Source repetition inventory/summary changed")
    cfg = method_config(root / "method.json")
    pairs = load_baselines(source)
    for seed, pair in pairs.items():
        entry = contract["seed_entries"][str(seed)]
        if (entry["source_experiment"] != str(pair["root"])
                or entry["source_contract_sha256"] != pair["experiment_sha256"]
                or entry["source_completion_sha256"] != pair["completions"]
                or entry["manifest_sha256"] != pair["contract"]["files"]["manifest.json"]
                or entry["order_sha256"] != pair["contract"]["files"]["training_order.json"]
                or entry["common_initial_state_sha256"] != pair["contract"]["initial_state_sha256"]
                or entry["cache_metadata_sha256"] != controls.file_sha256(pair["root"] / "features.json")
                or entry["cache_sha256"] != controls.file_sha256(pair["root"] / "features.pth")
                or set(entry["arms"]) != set(cfg["arms"])):
            raise ValueError("Historical source binding/cache/arm inventory changed")
        shared = torch.load(pair["root"] / "initial_head.pth", map_location="cpu", weights_only=True)
        for arm, spec in cfg["arms"].items():
            item = entry["arms"][arm]
            path = (root / item["initial_file"]).resolve()
            if not path.is_relative_to(root / f"seed{seed}") or controls.file_sha256(path) != item["initial_file_sha256"]:
                raise ValueError("Saved INITIAL file changed or escaped its seed directory")
            state = torch.load(path, map_location="cpu", weights_only=True)
            head = new_head(pair, spec, shared)
            objective = objective_for(pair, spec)
            if (controls.fingerprint_state(state) != item["initial_state_sha256"]
                    or controls.fingerprint_state(head.state_dict()) != item["initial_state_sha256"]
                    or sum(p.numel() for p in head.parameters()) != item["parameters"]
                    or list(head.required_gradient_modules) != item["required_gradients"]
                    or list(objective.raw_weights) != item["raw_weights"] or list(objective.weights) != item["weights"]
                    or sorted(set(state) - set(shared)) != item["added_state_keys"]):
                raise ValueError("Saved INITIAL state/objective/model contract differs")
        if entry["arms"]["F1"]["initial_file"] != entry["arms"]["S1"]["initial_file"]:
            raise ValueError("F1/S1 must load the exact same saved GRU INITIAL file")
    return {"root": root, "config": cfg, "contract": contract, "pairs": pairs,
            "digest": controls.file_sha256(root / "experiment.json")}


def directory(context, seed, arm, stage):
    return context["root"] / f"seed{seed}" / "arms" / arm / stage


def entry_for(context, seed, arm):
    return context["contract"]["seed_entries"][str(seed)]["arms"][arm]


def instantiate(context, seed, arm, device):
    pair = context["pairs"][seed]
    shared = torch.load(pair["root"] / "initial_head.pth", map_location="cpu", weights_only=True)
    head = new_head(pair, context["config"]["arms"][arm], shared)
    item = entry_for(context, seed, arm)
    state = torch.load(context["root"] / item["initial_file"], map_location="cpu", weights_only=True)
    head.load_state_dict(state, strict=True)
    if controls.fingerprint_state(head.state_dict()) != item["initial_state_sha256"]:
        raise ValueError("Run failed to reset to its saved INITIAL state")
    return head.to(device)


def predict_sequence(head, clip):
    images = clip["images"]
    patch = head.volume_stride // 2 ** head.volume_level
    outputs = head(clip["features"], images.shape[-2] // patch, images.shape[-1] // patch,
                   images.shape[1], images=images, extrinsics=clip["extrinsics"],
                   intrinsics=clip["intrinsics"], geometry_gauge="metric", return_sequence=True)
    if not isinstance(outputs, (list, tuple)) or len(outputs) != head.iters + 1:
        raise ValueError("Head omitted/added sequence predictions")
    return tuple(value.unflatten(0, images.shape[:2]) for value in outputs)


def aggregate_rows(rows):
    return {"overall": controls.aggregate_metrics(rows), "clips": rows,
            "scenes": {scene: controls.aggregate_metrics([r for r in rows if r["scene"] == scene])
                       for scene in sorted({r["scene"] for r in rows})}}


@torch.no_grad()
def evaluate_sequence(head, clips, manifest, objective, device, deadline):
    head.eval()
    stages = [{} for _ in objective.weights]
    for split, records in manifest["splits"].items():
        rows = [[] for _ in stages]
        for record in records:
            controls.check_deadline(deadline)
            clip = controls.on_device(clips[record["id"]], device)
            predictions = predict_sequence(head, clip)
            losses = objective(predictions, clip["depth"], clip["mask"])["per_prediction_index_l1"]
            support = controls.target_support(clip, objective.depth_min, objective.depth_max)
            if controls.support_sha256(support) != record["support_sha256"]:
                raise ValueError("Sequence evaluator changed the certified GT support")
            gt = clip["depth"][support].double()
            for index, (q, loss) in enumerate(zip(predictions, losses)):
                if not torch.isfinite(q).all() or (q < -1e-6).any() or (q > 1. + 1e-6).any():
                    raise FloatingPointError("Nonfinite/outside physical q; no clamping fallback")
                inverse = 1. / objective.depth_max + q * (1. / objective.depth_min - 1. / objective.depth_max)
                if (inverse <= 0).any():
                    raise FloatingPointError("Nonpositive inverse depth")
                pred = inverse[support].reciprocal().double()
                rows[index].append({"id": record["id"], "scene": record["scene"], "valid_pixels": int(support.sum()),
                    "index_l1": float(loss), "absrel": float(((pred - gt).abs() / gt).mean()),
                    "rmse": float((pred - gt).square().mean().sqrt()),
                    "delta1": float((torch.maximum(pred / gt, gt / pred) < 1.25).double().mean()),
                    "q_min": float(q.min()), "q_max": float(q.max())})
        for stage, items in zip(stages, rows):
            stage[split] = aggregate_rows(items)
    report = {"weights": list(objective.weights), "stages": stages,
              "weighted_index_l1": {split: math.fsum(w * stage[split]["overall"]["index_l1"]
                                                     for w, stage in zip(objective.weights, stages))
                                    for split in manifest["splits"]}}
    validate_evaluation(report, manifest, objective.weights)
    return report


def validate_evaluation(report, manifest, weights):
    if (set(report) != {"weights", "stages", "weighted_index_l1"}
            or report["weights"] != list(weights) or len(report["stages"]) != len(weights)
            or set(report["weighted_index_l1"]) != set(manifest["splits"])):
        raise ValueError("Sequence evaluation weights/stage/split inventory differs")
    for stage in report["stages"]:
        legacy.validate_metrics(stage, manifest)
        for split in stage.values():
            for row in split["clips"]:
                if not -1e-6 <= row["q_min"] <= row["q_max"] <= 1. + 1e-6:
                    raise ValueError("Invalid/nonfinite evaluated q range")
    for split in manifest["splits"]:
        expected = math.fsum(w * stage[split]["overall"]["index_l1"] for w, stage in zip(weights, report["stages"]))
        if report["weighted_index_l1"][split] != expected:
            raise ValueError("Weighted sequence metric aggregation differs")


def numeric_policy():
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark}


def verify_model(head, pair, clip, objective):
    """Disposable clone: direct per-output/update grads, legacy and inference parity."""
    before = controls.fingerprint_state(head.state_dict())
    probe = copy.deepcopy(head).eval()
    base = controls.build_head(pair["config"]["arms"]["C1"]["decoder"], pair["config"], pair["contract"]["backbone"])
    base.load_state_dict({key: value for key, value in probe.state_dict().items()
                          if not key.startswith(probe.recurrent_state_prefixes)}, strict=True)
    base = base.to(clip["images"].device).eval()
    kwargs = {**pair["config"]["decoder"]["kwargs"], "iters": head.iters}
    old = build_decoder(get_decoder_class("DPTHeadCalibratedGRUConvNeXt"), kwargs,
                        in_channels_list=pair["contract"]["backbone"]["embed_dims"],
                        patch_size=pair["contract"]["backbone"]["patch_size"])
    old_state = old.state_dict()
    old_state.update(probe.state_dict())  # At zero iters only unused recurrent keys remain.
    old.load_state_dict(old_state, strict=True)
    old = old.to(clip["images"].device).eval()
    with torch.no_grad():
        sequence = predict_sequence(probe, clip)
        pairs = {"initial_vs_c1": (sequence[0], controls.predict(base, clip)),
                 "final_vs_legacy_g1": (sequence[-1], controls.predict(old, clip)),
                 "sequence_vs_fast_eval": (sequence[-1], controls.predict(probe, clip))}
        parity = {}
        for name, (actual, expected) in pairs.items():
            parity[name] = float((actual - expected).abs().max())
            if not torch.allclose(actual, expected, atol=1e-7, rtol=1e-6) or not 0 <= parity[name] <= 1e-6:
                raise ValueError(f"Forward contract failed: {name}, max_abs_diff={parity[name]}")
    del old, base
    probe.train()
    updates, handles = [], []

    def keep_update(_module, _args, result):
        result.retain_grad()
        updates.append(result)

    for name, module in probe.named_modules():
        if name == "index_head":
            handles.append(module.register_forward_hook(keep_update))
    try:
        predictions = predict_sequence(probe, clip)
        for q in predictions:
            q.retain_grad()
        objective(predictions, clip["depth"], clip["mask"])["total_loss"].backward()
        norm = lambda value: 0. if value.grad is None else float(value.grad.detach().double().square().sum().sqrt())
        output_norms = [norm(q) for q in predictions]
        update_norms = [norm(q) for q in updates]
        module_norms = {}
        for name in probe.required_gradient_modules:
            parameters = list(getattr(probe, name).parameters())
            if not parameters or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError(f"Missing/nonfinite gradient in required module {name}")
            module_norms[name] = legacy.gradient_norm(getattr(probe, name))
    finally:
        for handle in handles:
            handle.remove()
    report = {"parity": parity, "output_gradient_norms": output_norms, "update_gradient_norms": update_norms,
              "module_gradient_norms": module_norms, "numerical_policy": numeric_policy(),
              "original_state_unchanged": controls.fingerprint_state(head.state_dict()) == before}
    validate_verification(report, objective.weights, probe.required_gradient_modules)
    return report


def validate_verification(report, weights, required):
    if (report["original_state_unchanged"] is not True
            or report["numerical_policy"] != {"float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False,
                                                "cudnn_allow_tf32": False, "cudnn_benchmark": False}
            or set(report["parity"]) != {"initial_vs_c1", "final_vs_legacy_g1", "sequence_vs_fast_eval"}
            or not all(math.isfinite(v) and 0 <= v <= 1e-6 for v in report["parity"].values())
            or set(report["module_gradient_norms"]) != set(required)
            or not all(math.isfinite(v) and v > 0 for v in report["module_gradient_norms"].values())):
        raise ValueError("Failed forward/module-gradient/numerical-policy evidence")
    for values, expected in ((report["output_gradient_norms"], weights), (report["update_gradient_norms"], weights[1:])):
        if len(values) != len(expected) or not all(math.isfinite(v) and v >= 0 and (v > 0) == (w > 0)
                                                 for v, w in zip(values, expected)):
            raise ValueError("Direct initial/per-iteration gradient evidence differs from loss weights")


def prerequisite_hashes(context, arm):
    result = {}
    for required in context["config"]["arms"][arm]["requires"]:
        for seed in SEEDS:
            path = directory(context, seed, required, "formal") / "completed.json"
            if not path.is_file():
                raise ValueError(f"All seeds of prerequisite {required} must be completed before {arm}")
            result[f"seed{seed}/{required}"] = controls.file_sha256(path)
    return result


def require_predecessors(context, arm):
    for required in context["config"]["arms"][arm]["requires"]:
        for seed in SEEDS:
            check_completed(context, seed, required)
    return prerequisite_hashes(context, arm)


def gate_hashes(context, arm):
    result = {}
    if context["config"]["arms"][arm]["gate_required"]:
        for seed in SEEDS:
            verify_gate(context, seed, arm)
            result[str(seed)] = controls.file_sha256(directory(context, seed, arm, "gate") / "gate.json")
    return result


def initialisation(context, seed, arm, prerequisites):
    item = entry_for(context, seed, arm)
    return {"seed": seed, "arm": arm, "experiment_sha256": context["digest"],
            "initial_file_sha256": item["initial_file_sha256"], "initial_state_sha256": item["initial_state_sha256"],
            "common_initial_state_sha256": context["pairs"][seed]["contract"]["initial_state_sha256"],
            "optimizer_initial_state_entries": 0, "gate_checkpoint_loaded": False,
            "prerequisite_certificates": prerequisites}


def optimisation(context, seed, arm, destination, manifest, order, device, eval_every, log_every, prerequisites):
    pair = context["pairs"][seed]
    controls.seed_everything(seed)  # Each CLI stage is a fresh process; do this BEFORE cache/evaluation.
    print("SEQUENCE_NUMERICS", seed, arm, json.dumps(numeric_policy()), flush=True)
    deadline = time.monotonic() + pair["config"]["max_seconds_per_arm"]
    clips, cache = legacy.load_cache(pair, device, deadline)
    if cache["file_sha256"] != context["contract"]["seed_entries"][str(seed)]["cache_sha256"]:
        raise ValueError("Run cache differs from bound source")
    head = instantiate(context, seed, arm, device)
    spec = context["config"]["arms"][arm]
    objective = objective_for(pair, spec).to(device)
    clip = controls.on_device(clips[manifest["splits"]["train"][0]["id"]], device)
    verification = verify_model(head, pair, clip, objective)
    controls.write_json(destination / "verification.json", verification)
    # Discard the diagnostic clone/RNG consumption; formally start from saved INITIAL.
    del head
    head = instantiate(context, seed, arm, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=pair["config"]["optimizer"]["learning_rate"],
                                 weight_decay=pair["config"]["optimizer"]["weight_decay"])
    if len(optimizer.state) != 0:
        raise ValueError("Optimizer is not fresh")
    controls.write_json(destination / "initialisation.json", initialisation(context, seed, arm, prerequisites))
    initial = evaluate_sequence(head, clips, manifest, objective, device, deadline)
    controls.write_json(destination / "evaluation_initial.json", initial)
    print(f"INITIAL {arm} seed={seed}", json.dumps(initial["weighted_index_l1"]), flush=True)
    final = None
    with open(destination / "progress.jsonl", "x") as log, open(destination / "trace.jsonl", "x") as trace:
        for step, clip_id in enumerate(order, start=1):
            controls.check_deadline(deadline)
            head.train()
            clip = controls.on_device(clips[clip_id], device)
            optimizer.zero_grad(set_to_none=True)
            outputs = predict_sequence(head, clip)
            losses = objective(outputs, clip["depth"], clip["mask"])
            loss = losses["total_loss"]
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), pair["config"]["optimizer"]["grad_clip"],
                                                 error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step, "clip_id": clip_id, "loss": float(loss.detach()),
                   "per_prediction_index_l1": losses["per_prediction_index_l1"].detach().tolist(),
                   "valid_pixels": int(losses["valid_pixels"]), "grad_norm_before_clip": float(norm.detach())}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()  # Every optimizer update/order is accounted for, not just every 100th.
            if step == 1 or step % log_every == 0 or step == len(order):
                trace.write(json.dumps({"step": step, "iterations": getattr(head, "last_iteration_diagnostics", []),
                                        "diagnostics": head.last_diagnostics}, allow_nan=False) + "\n")
                trace.flush()
                print(f"TRAIN {arm} seed={seed}", json.dumps(row), flush=True)
            if step % eval_every == 0 or step == len(order):
                final = evaluate_sequence(head, clips, manifest, objective, device, deadline)
                controls.write_json(destination / f"evaluation_step_{step:06d}.json", final)
                print(f"EVAL {arm} seed={seed} step={step}",
                      json.dumps({s: v["overall"] for s, v in final["stages"][-1].items()}), flush=True)
    controls.check_deadline(deadline)
    controls.write_json(destination / "evaluation_final.json", final)
    controls.write_json(destination / "metrics_final.json", final["stages"][-1])
    return head, initial, final, deadline


def artifact_names(steps, eval_every, formal):
    evaluations = {f"evaluation_step_{step:06d}.json" for step in range(1, steps + 1)
                   if step % eval_every == 0 or step == steps}
    return COMMON_ARTIFACTS | evaluations | ({"final_head.pth"} if formal else set())


def validate_run_artifacts(context, seed, arm, stage, certificate, manifest, order, eval_every, log_every, formal):
    destination = directory(context, seed, arm, stage)
    item = entry_for(context, seed, arm)
    names = artifact_names(len(order), eval_every, formal)
    if set(certificate["files"]) != names:
        raise ValueError("Incomplete run artifact inventory")
    for name in names:
        if controls.file_sha256(destination / name) != certificate["files"][name]:
            raise ValueError(f"Run artifact changed: {seed}/{arm}/{stage}/{name}")
    init = controls.read_json(destination / "initialisation.json")
    if init != initialisation(context, seed, arm, prerequisite_hashes(context, arm)):
        raise ValueError("Run initialization/reset/prerequisite evidence differs")
    validate_verification(controls.read_json(destination / "verification.json"), item["weights"], item["required_gradients"])
    initial = controls.read_json(destination / "evaluation_initial.json")
    final = controls.read_json(destination / "evaluation_final.json")
    for report in (initial, final):
        validate_evaluation(report, manifest, item["weights"])
    for name in names:
        if name.startswith("evaluation_step_"):
            validate_evaluation(controls.read_json(destination / name), manifest, item["weights"])
    if (controls.read_json(destination / "metrics_final.json") != final["stages"][-1]
            or controls.read_json(destination / f"evaluation_step_{len(order):06d}.json") != final):
        raise ValueError("Final output/step metrics differ")
    progress = [json.loads(line) for line in (destination / "progress.jsonl").read_text().splitlines()]
    if [(row["step"], row["clip_id"]) for row in progress] != list(enumerate(order, 1)):
        raise ValueError("Actual optimizer update inventory/order differs")
    support = {row["id"]: row["valid_pixels"] for row in manifest["splits"]["train"]}
    for row in progress:
        values = row["per_prediction_index_l1"]
        if (len(values) != len(item["weights"]) or not all(math.isfinite(v) and v >= 0 for v in values)
                or not math.isfinite(row["grad_norm_before_clip"]) or row["grad_norm_before_clip"] < 0
                or row["valid_pixels"] != support[row["clip_id"]]
                or not math.isclose(row["loss"], math.fsum(w * v for w, v in zip(item["weights"], values)),
                                    rel_tol=2e-6, abs_tol=1e-7)):
            raise ValueError("Logged sequence loss/gradient/support does not implement declared objective")
    trace = [json.loads(line) for line in (destination / "trace.jsonl").read_text().splitlines()]
    if [row["step"] for row in trace] != [step for step in range(1, len(order) + 1)
                                         if step == 1 or step % log_every == 0 or step == len(order)]:
        raise ValueError("Iteration trace step inventory differs")
    iterations = context["config"]["arms"][arm]["iterations"]
    maximum = context["pairs"][seed]["config"]["decoder"]["kwargs"]["num_sample"] - 1
    for row in trace:
        if [r["iteration"] for r in row["iterations"]] != list(range(1, iterations + 1)):
            raise ValueError("Missing recurrent iteration trace")
        for r in row["iterations"]:
            if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in r.values())
                    or not 0 <= r["lookup_index_min"] <= r["lookup_index_max"] <= maximum
                    or not 0 <= r["updated_index_min"] <= r["updated_index_max"] <= maximum
                    or not 0 <= r["additive_proposal_outside_fraction"] <= 1):
                raise ValueError("Invalid/hidden recurrent update behavior")
    return initial, final


def gate_order(manifest, steps):
    ids = [row["id"] for row in manifest["splits"]["train"]]
    return [ids[index % len(ids)] for index in range(steps)]


def verify_certificate_header(context, seed, arm, certificate, status, steps):
    item = entry_for(context, seed, arm)
    entry = context["contract"]["seed_entries"][str(seed)]
    if (certificate["status"] != status or certificate["seed"] != seed or certificate["arm"] != arm
            or certificate["steps"] != steps or certificate["experiment_sha256"] != context["digest"]
            or certificate["initial_state_sha256"] != item["initial_state_sha256"]
            or certificate["cache_sha256"] != entry["cache_sha256"]
            or certificate["prerequisite_certificates"] != prerequisite_hashes(context, arm)):
        raise ValueError("Foreign/short/changed run certificate")


def verify_gate(context, seed, arm, certificate=None):
    cfg = context["config"]
    if not cfg["arms"][arm]["gate_required"]:
        raise ValueError("This arm has no method gate")
    path = directory(context, seed, arm, "gate") / "gate.json"
    if certificate is None and not path.is_file():
        raise ValueError(f"All three {arm} gates must pass before its formal training")
    report = controls.read_json(path) if certificate is None else certificate
    steps = cfg["gate"]["steps"]
    verify_certificate_header(context, seed, arm, report, "passed", steps)
    manifest = legacy.choose_gate_manifest(context["pairs"][seed]["manifest"])
    initial, final = validate_run_artifacts(context, seed, arm, "gate", report, manifest,
                                           gate_order(manifest, steps), steps, 50, formal=False)
    ratios = {"weighted_sequence": final["weighted_index_l1"]["train"] / max(initial["weighted_index_l1"]["train"], 1e-12),
              "final_index_l1": final["stages"][-1]["train"]["overall"]["index_l1"]
              / max(initial["stages"][-1]["train"]["overall"]["index_l1"], 1e-12)}
    if (report["ratios"] != ratios or report["gate_weights_discarded"] is not True
            or report["formal_must_reset"] is not True
            or not all(math.isfinite(v) and 0 <= v <= cfg["gate"]["maximum_final_initial_loss_ratio"] for v in ratios.values())
            or any(path.parent.glob("*.pth"))):
        raise ValueError("Training-only gate failed loss/reset evidence; no threshold relaxation")
    return report


def certificate_header(context, seed, arm, steps, status, files, destination, prerequisites):
    return {"status": status, "seed": seed, "arm": arm, "steps": steps,
            "experiment_sha256": context["digest"],
            "initial_state_sha256": entry_for(context, seed, arm)["initial_state_sha256"],
            "cache_sha256": context["contract"]["seed_entries"][str(seed)]["cache_sha256"],
            "prerequisite_certificates": prerequisites,
            "files": {name: controls.file_sha256(destination / name) for name in sorted(files)}}


def run_gate(output, seed, arm, device, *, _context=None):
    context = verify(output) if _context is None else _context
    cfg = context["config"]
    if not cfg["arms"][arm]["gate_required"]:
        raise ValueError("B0 must run as a fresh formal baseline, not as a method gate")
    prerequisites = require_predecessors(context, arm)
    destination = directory(context, seed, arm, "gate")
    destination.mkdir(parents=True)  # Fails on every partial/previous gate.
    try:
        manifest = legacy.choose_gate_manifest(context["pairs"][seed]["manifest"])
        steps = cfg["gate"]["steps"]
        head, initial, final, deadline = optimisation(context, seed, arm, destination, manifest,
            gate_order(manifest, steps), device, steps, 50, prerequisites)
        del head  # Gate states are deliberately never serialized.
        report = certificate_header(context, seed, arm, steps, "passed",
            artifact_names(steps, steps, False), destination, prerequisites)
        report.update({"ratios": {
            "weighted_sequence": final["weighted_index_l1"]["train"] / max(initial["weighted_index_l1"]["train"], 1e-12),
            "final_index_l1": final["stages"][-1]["train"]["overall"]["index_l1"]
            / max(initial["stages"][-1]["train"]["overall"]["index_l1"], 1e-12)},
            "gate_weights_discarded": True, "formal_must_reset": True})
        verify_gate(context, seed, arm, report)
        controls.check_deadline(deadline)
        controls.write_json(destination / "gate.json", report)
        print(f"SEQUENCE_GATE_PASSED {arm} seed={seed}", json.dumps(report["ratios"]), flush=True)
        return report
    except Exception as exc:
        controls.write_json(destination / "failure.json", {"error": str(exc), "formal_allowed": False})
        raise


def check_completed(context, seed, arm, certificate=None):
    destination = directory(context, seed, arm, "formal")
    path = destination / "completed.json"
    if certificate is None and not path.is_file():
        raise ValueError(f"Required new control {arm}/seed{seed} is not completed")
    report = controls.read_json(path) if certificate is None else certificate
    pair = context["pairs"][seed]
    cfg = pair["config"]
    verify_certificate_header(context, seed, arm, report, "completed", cfg["steps"])
    if report["gate_certificates"] != gate_hashes(context, arm):
        raise ValueError("Formal run did not bind all three unchanged method gates")
    order = controls.read_json(pair["root"] / "training_order.json")
    initial, final = validate_run_artifacts(context, seed, arm, "formal", report, pair["manifest"], order,
                                           cfg["eval_every"], cfg["log_every"], formal=True)
    payload = torch.load(destination / "final_head.pth", map_location="cpu", weights_only=True)
    if (set(payload) != {"seed", "arm", "steps", "experiment_sha256", "initial_state_sha256", "model_state_dict"}
            or any(payload[key] != report[key] for key in ("seed", "arm", "steps", "experiment_sha256", "initial_state_sha256"))
            or controls.fingerprint_state(payload["model_state_dict"]) != report["final_state_sha256"]
            or any(not torch.isfinite(value).all() for value in payload["model_state_dict"].values())):
        raise ValueError("Completed checkpoint metadata/state differs")
    head = instantiate(context, seed, arm, torch.device("cpu"))
    head.load_state_dict(payload["model_state_dict"], strict=True)
    # The fresh zero-GRU baseline is algebraically identical to historical C1;
    # require its final replay before it can certify this NEW protocol.
    if not context["config"]["arms"][arm]["requires"]:
        legacy.assert_metric_replay(final["stages"][-1], pair["metrics"]["C1"])
    return report


def train(output, seed, arm, device, *, _context=None):
    context = verify(output) if _context is None else _context
    prerequisites = require_predecessors(context, arm)
    gates = gate_hashes(context, arm)
    pair = context["pairs"][seed]
    cfg = pair["config"]
    destination = directory(context, seed, arm, "formal")
    destination.mkdir(parents=True)  # Never overwrite or load a partially trained model.
    started = time.monotonic()
    try:
        order = controls.read_json(pair["root"] / "training_order.json")
        head, _initial, _final, deadline = optimisation(context, seed, arm, destination,
            pair["manifest"], order, device, cfg["eval_every"], cfg["log_every"], prerequisites)
        state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
        payload = {"seed": seed, "arm": arm, "steps": len(order), "experiment_sha256": context["digest"],
                   "initial_state_sha256": entry_for(context, seed, arm)["initial_state_sha256"], "model_state_dict": state}
        controls.save_state(destination / "final_head.pth", payload)
        report = certificate_header(context, seed, arm, len(order), "completed",
            artifact_names(len(order), cfg["eval_every"], True), destination, prerequisites)
        report.update({"final_state_sha256": controls.fingerprint_state(state), "gate_certificates": gates,
                       "elapsed_seconds": time.monotonic() - started})
        check_completed(context, seed, arm, report)
        controls.check_deadline(deadline)
        controls.write_json(destination / "completed.json", report)  # LAST, after all evidence verifies.
        print(f"SEQUENCE_COMPLETED {arm} seed={seed} steps={len(order)}: {destination}", flush=True)
        return report
    except Exception as exc:
        controls.write_json(destination / "failure.json", {"error": str(exc), "status": "failed", "next_arm_allowed": False})
        raise


def summarize(output, *, _context=None):
    context = verify(output) if _context is None else _context
    cfg, certificates, metrics = context["config"], {}, {}
    for arm in cfg["arms"]:
        require_predecessors(context, arm)
        for seed in SEEDS:
            check_completed(context, seed, arm)
            destination = directory(context, seed, arm, "formal")
            certificates[f"seed{seed}/{arm}"] = controls.file_sha256(destination / "completed.json")
            metrics[seed, arm] = controls.read_json(destination / "metrics_final.json")
    comparisons = (("F1", "B0"), ("S1", "B0"), ("S1", "F1"))
    rows = []
    for split, records in context["pairs"][0]["manifest"]["splits"].items():
        for group in ["overall"] + sorted({row["scene"] for row in records}):
            for metric in repeats.METRICS:
                values = {}
                for arm in cfg["arms"]:
                    values[arm] = [(metrics[s, arm][split]["overall"] if group == "overall"
                                    else metrics[s, arm][split]["scenes"][group])[metric] for s in SEEDS]
                row = {"split": split, "group": group, "metric": metric,
                       **{arm: repeats.distribution(items) for arm, items in values.items()}}
                for method, baseline in comparisons:
                    row[f"{method}_minus_{baseline}"] = repeats.distribution(
                        [a - b for a, b in zip(values[method], values[baseline])])
                rows.append(row)
    steps = context["pairs"][0]["config"]["steps"]
    report = {"status": "all_sequence_controls_complete", "experiment_sha256": context["digest"],
              "seeds": list(SEEDS), "final_step": steps, "certificates": certificates, "rows": rows,
              "formal_updates": steps * len(SEEDS) * len(cfg["arms"]),
              "disposable_gate_updates": cfg["gate"]["steps"] * len(SEEDS)
              * sum(spec["gate_required"] for spec in cfg["arms"].values()),
              "head_parameters": {arm: entry_for(context, 0, arm)["parameters"] for arm in cfg["arms"]},
              "weights": {arm: entry_for(context, 0, arm)["weights"] for arm in cfg["arms"]},
              "limits": ["GT-camera fixed development subset, not RGB-only or official IGEV/SOTA.",
                         "S1 vs F1 isolates sequence weights; S1 vs B0 also adds recurrent parameters/compute.",
                         "All seeds/final steps included; seed sample SD is not independent-scene significance.",
                         "Per-iteration metrics are diagnostics, not permission to choose a better iteration."]}
    path = context["root"] / "comparison.json"
    if path.exists():
        if controls.read_json(path) != report:
            raise ValueError("Existing comparison differs; refusing overwrite")
    else:
        controls.write_json(path, report)
    print(f"SEQUENCE_FINAL_STEP={steps}; seeds=0,1,2; B0/F1/S1; formal={report['formal_updates']}; gates={report['disposable_gate_updates']}")
    for row in rows:
        print(row["split"], row["group"], row["metric"],
              *[f"{arm}={row[arm]['mean']:.6f}+/-{row[arm]['sample_std']:.6f}" for arm in cfg["arms"]],
              "S1-F1=", row["S1_minus_F1"]["values"], "S1-B0=", row["S1_minus_B0"]["values"])
    return report


def next_action(output, *, _context=None):
    """Continue orchestration only, never resume weights or retry partial phases."""
    context = verify(output) if _context is None else _context
    for arm, spec in context["config"]["arms"].items():
        phases = (["gate"] if spec["gate_required"] else []) + ["formal"]
        for stage in phases:
            for seed in SEEDS:
                destination = directory(context, seed, arm, stage)
                if not destination.exists():
                    return f"{'train' if stage == 'formal' else 'gate'} {arm} {seed}"
                validator = {"formal": check_completed, "gate": verify_gate}[stage]
                validator(context, seed, arm)  # Partial/failed -> error, never an automatic retry.
    return "summarize - -"


def production_guard(cfg, pairs):
    if (cfg["gate"] != {"steps": 200, "clips_per_train_scene": 1,
                        "maximum_final_initial_loss_ratio": .5}
            or cfg != method_config(DEFAULT_CONFIG)):
        raise ValueError("Production CLI requires the fixed 200-step/.5 sequence gate")
    reference = OmegaConf.to_container(OmegaConf.load(ROOT / "config/stereogru/matched_c0_c1_30train.yaml"), resolve=True)
    for seed, pair in pairs.items():
        expected = {**reference, "seed": seed}
        if pair["config"] != expected:
            raise ValueError("Production CLI requires original 30train/16dev/crop256/1000-step controls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--repetitions", required=True)
    setup.add_argument("--output", required=True)
    setup.add_argument("--config", default=str(DEFAULT_CONFIG))
    for name in ("gate", "train", "next", "summarize"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", required=True)
        if name in ("gate", "train"):
            sub.add_argument("--seed", required=True, type=int, choices=SEEDS)
            sub.add_argument("--arm", required=True, choices=("B0", "F1", "S1"))
    args = parser.parse_args()
    if args.command == "prepare":
        production_guard(method_config(args.config), load_baselines(args.repetitions))
        prepare(args.repetitions, args.output, args.config)
        return
    # One fresh verification per CLI invocation, reused only within this process.
    # Registry discovery may print notices even after top-level imports finished.
    with redirect_stdout(sys.stderr):
        context = verify(args.output)
        production_guard(context["config"], context["pairs"])
        if args.command == "next":
            action = next_action(args.output, _context=context)
    if args.command == "next":
        print(f"SEQUENCE_NEXT {action}")
        return
    if args.command in ("gate", "train") and not torch.cuda.is_available():
        raise RuntimeError("Real runs require an idle CUDA GPU; CPU fixtures use the API, not production CLI")
    actions = {"gate": lambda: run_gate(args.output, args.seed, args.arm, torch.device("cuda"), _context=context),
               "train": lambda: train(args.output, args.seed, args.arm, torch.device("cuda"), _context=context),
               "summarize": lambda: summarize(args.output, _context=context)}
    actions[args.command]()


if __name__ == "__main__":
    main()