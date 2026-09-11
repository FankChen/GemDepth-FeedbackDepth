"""Baseline-gated corrected GRU controls using completed C1 data/init/metrics.

The old C0/C1 sources and artifacts stay untouched. Gates optimise disposable
copies on TRAIN clips only; formal G1 always resets to its saved common-initial
mapping and an empty optimizer. All three seeds and final-step metrics are used.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

import stereogru_matched_controls as controls
from stereogru_matched_repeats import aggregate_pairs, assert_same_protocol, verify_pair
from model.decoder_registry import build_decoder, get_decoder_class


METHOD_SOURCES = ("model/dpt_calibrated_gru_convnext.py", "scripts/stereogru_corrected_gru.py",
                  "scripts/stereogru_matched_repeats.py")
SEEDS = (0, 1, 2)
REQUIRED_GRADIENTS = {"matcher", "classifier", "encoder", "gru_coarse", "gru_mid", "gru_fine", "index_head"}


def validate_metrics(metrics, manifest):
    """Recompute every aggregate from the exact certified clip/support inventory."""
    if set(metrics) != set(manifest["splits"]):
        raise ValueError("Metrics omit or add a data split")
    for split, expected in manifest["splits"].items():
        item = metrics[split]
        rows = item["clips"]
        if ([(r["id"], r["scene"], r["valid_pixels"]) for r in rows]
                != [(r["id"], r["scene"], r["valid_pixels"]) for r in expected]):
            raise ValueError("Metrics changed sample/support inventory")
        if not all(r["valid_pixels"] > 0 and all(math.isfinite(r[k]) and r[k] >= 0
                   for k in ("index_l1", "absrel", "rmse", "delta1")) and r["delta1"] <= 1 for r in rows):
            raise ValueError("Invalid/nonfinite clip metrics")
        if item["overall"] != controls.aggregate_metrics(rows):
            raise ValueError("Overall metric aggregation differs")
        scenes = {r["scene"] for r in expected}
        if set(item["scenes"]) != scenes:
            raise ValueError("Scene metric inventory differs")
        for scene in scenes:
            if item["scenes"][scene] != controls.aggregate_metrics([r for r in rows if r["scene"] == scene]):
                raise ValueError("Scene metric aggregation differs")


def validate_baseline_summary(inventory, summary, pairs):
    if inventory["script_sha256"] != controls.file_sha256(ROOT / "scripts/stereogru_matched_repeats.py"):
        raise ValueError("Repetition verifier source changed")
    for pair in pairs.values():
        for metrics in pair["metrics"].values():
            validate_metrics(metrics, pair["manifest"])
    reference = pairs[0]
    if (summary["scene_metrics"] != aggregate_pairs(pairs)
            or summary["steps_each"] != reference["config"]["steps"]
            or summary["protocol"] != reference["config"]["protocol"]
            or summary["manifest_sha256"] != reference["contract"]["files"]["manifest.json"]
            or inventory["manifest_sha256"] != summary["manifest_sha256"]):
        raise ValueError("Baseline summary aggregation/contract differs")


def method_config(path):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if (cfg["decoder"] != "DPTHeadCalibratedGRUConvNeXt" or cfg["iterations"] != 8
            or cfg["index_update"] != "remaining_range_tanh_v1"
            or cfg["objective_contract"] != "same_as_c1_final_absolute_index_l1"
            or cfg["seeds"] != list(SEEDS) or not cfg["gate_reset_required"]):
        raise ValueError("Unexpected G1 method contract; do not silently change baseline or loss")
    gate = cfg["gate"]
    if not 1 <= gate["steps"] <= 200 or gate["clips_per_train_scene"] != 1 or not 0 < gate["maximum_final_initial_loss_ratio"] <= 1:
        raise ValueError("Invalid disposable training-only gate")
    return cfg


def make_gru(base_cfg, backbone, cfg, shared_initial):
    kwargs = copy.deepcopy(base_cfg["decoder"]["kwargs"])
    kwargs["iters"] = cfg["iterations"]
    controls.seed_everything(int(base_cfg["seed"]))
    head = build_decoder(get_decoder_class(cfg["decoder"]), kwargs,
                         in_channels_list=backbone["embed_dims"], patch_size=backbone["patch_size"])
    head.load_c1_initial(shared_initial)
    common = {key: head.state_dict()[key] for key in shared_initial}
    if controls.fingerprint_state(common) != controls.fingerprint_state(shared_initial):
        raise ValueError("G1 common state does not exactly match the baseline INITIAL state")
    return head


def prepare(repetitions, output, config_path):
    cfg = method_config(config_path)
    repetitions = Path(repetitions).resolve()
    inventory = controls.read_json(repetitions / "repetitions.json")
    summary = controls.read_json(repetitions / "seed_summary.json")
    if (inventory["seeds"] != list(SEEDS) or set(inventory["experiments"]) != {str(seed) for seed in SEEDS}
            or summary["status"] != "three_paired_seeds_complete" or summary["seeds"] != list(SEEDS)):
        raise ValueError("All three original C1 baselines must be completed before preparing G1")
    pairs = {}
    for seed in SEEDS:
        entry = inventory["experiments"][str(seed)]
        pair = verify_pair(entry["experiment"])
        if (pair["experiment_sha256"] != entry["experiment_sha256"]
                or summary["experiments"][str(seed)]["contract_sha256"] != pair["experiment_sha256"]
                or summary["experiments"][str(seed)]["completion_sha256"] != pair["completions"]):
            raise ValueError("Completed baseline differs from the recorded paired experiment")
        if seed == 0 and pair["completions"] != entry["completion_sha256"]:
            raise ValueError("Seed-0 baseline changed")
        pairs[seed] = pair
        assert_same_protocol(pairs[0], pair, seed)
    validate_baseline_summary(inventory, summary, pairs)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("G1 requires a new empty output; no resume/overwrite")
    controls.write_json(output / "method.json", cfg)
    entries = {}
    for seed, pair in pairs.items():
        destination = output / f"seed{seed}"
        destination.mkdir()
        shared = torch.load(pair["root"] / "initial_head.pth", map_location="cpu", weights_only=True)
        head = make_gru(pair["config"], pair["contract"]["backbone"], cfg, shared)
        state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        controls.save_state(destination / "method_initial.pth", state)
        entries[str(seed)] = {
            "source_experiment": str(pair["root"]), "source_contract_sha256": pair["experiment_sha256"],
            "baseline_completion_sha256": pair["completions"],
            "baseline_initial_sha256": pair["contract"]["files"]["initial_head.pth"],
            "common_initial_state_sha256": pair["contract"]["initial_state_sha256"],
            "method_initial_file_sha256": controls.file_sha256(destination / "method_initial.pth"),
            "method_initial_state_sha256": controls.fingerprint_state(state),
            "baseline_parameters": pair["contract"]["head_parameters"],
            "method_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "added_state_keys": sorted(set(state) - set(shared)),
            "baseline_metrics": pair["metrics"]["C1"],
        }
    contract = {"status": "prepared_corrected_gru", "method_sha256": controls.file_sha256(output / "method.json"),
                "source_sha256": {name: controls.file_sha256(ROOT / name) for name in METHOD_SOURCES},
                "seed_entries": entries, "seeds": list(SEEDS), "runtime": controls.runtime_versions(),
                "repetitions": str(repetitions), "baseline_summary_sha256": controls.file_sha256(repetitions / "seed_summary.json"),
                "interpretation": "G1 has additional recurrent parameters/compute and bounded index update; not exact IGEV-MVS or RGB-only/SOTA."}
    controls.write_json(output / "gru_experiment.json", contract)
    print("G1_PREPARED", json.dumps({seed: {key: value[key] for key in ("baseline_parameters", "method_parameters", "common_initial_state_sha256")}
                                      for seed, value in entries.items()}), flush=True)
    print("No training started; each formal run needs its disposable train-only gate and resets to method_initial.", flush=True)
    return contract


def verify(output, seed):
    root = Path(output).resolve()
    contract = controls.read_json(root / "gru_experiment.json")
    if (contract["status"] != "prepared_corrected_gru" or contract["seeds"] != list(SEEDS)
            or contract["runtime"] != controls.runtime_versions() or seed not in SEEDS):
        raise ValueError("Invalid G1 preparation/runtime/seed")
    if controls.file_sha256(root / "method.json") != contract["method_sha256"]:
        raise ValueError("Method config changed after baseline binding")
    cfg = method_config(root / "method.json")
    if set(contract["source_sha256"]) != set(METHOD_SOURCES):
        raise ValueError("Method source inventory differs")
    for path, digest in contract["source_sha256"].items():
        if controls.file_sha256(ROOT / path) != digest:
            raise ValueError(f"G1 source changed: {path}")
    if controls.file_sha256(Path(contract["repetitions"]) / "seed_summary.json") != contract["baseline_summary_sha256"]:
        raise ValueError("Completed baseline summary changed")
    entry = contract["seed_entries"][str(seed)]
    pair = verify_pair(entry["source_experiment"])
    for metrics in pair["metrics"].values():
        validate_metrics(metrics, pair["manifest"])
    if (pair["config"]["seed"] != seed or pair["experiment_sha256"] != entry["source_contract_sha256"]
            or pair["completions"] != entry["baseline_completion_sha256"]
            or pair["metrics"]["C1"] != entry["baseline_metrics"]):
        raise ValueError("C1 prerequisite/metrics changed")
    path = root / f"seed{seed}" / "method_initial.pth"
    if controls.file_sha256(path) != entry["method_initial_file_sha256"]:
        raise ValueError("G1 saved initial file changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if controls.fingerprint_state(state) != entry["method_initial_state_sha256"]:
        raise ValueError("G1 initial tensor fingerprint changed")
    shared = torch.load(pair["root"] / "initial_head.pth", map_location="cpu", weights_only=True)
    if controls.fingerprint_state({key: state[key] for key in shared}) != pair["contract"]["initial_state_sha256"]:
        raise ValueError("G1 common tensors no longer match C1 INITIAL")
    return root, cfg, contract, entry, pair, state, shared


def instantiated(pair, cfg, state, shared, device):
    head = make_gru(pair["config"], pair["contract"]["backbone"], cfg, shared)
    head.load_state_dict(state, strict=True)
    return head.to(device)


def load_cache(pair, device, deadline):
    return controls.ensure_cache(pair["root"], pair["config"], pair["manifest"], pair["contract"],
                                 device, allow_create=False, deadline=deadline)


def assert_metric_replay(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("C1 replay split inventory changed")
    for split, item in expected.items():
        if set(actual[split]["scenes"]) != set(item["scenes"]):
            raise ValueError("C1 replay scene inventory changed")
        aggregates = [("overall", actual[split]["overall"], item["overall"])]
        aggregates.extend((s, actual[split]["scenes"][s], item["scenes"][s]) for s in item["scenes"])
        for group, a, b in aggregates:
            if a["valid_pixels"] != b["valid_pixels"]:
                raise ValueError("C1 replay support changed")
            for name in ("index_l1", "absrel", "rmse", "delta1"):
                if not math.isclose(a[name], b[name], rel_tol=1e-4, abs_tol=1e-6):
                    raise ValueError(
                        f"C1 final metrics did not replay: {split}/{group}/{name}; "
                        f"actual={a[name]:.17g}, expected={b[name]:.17g}, "
                        f"abs_diff={abs(a[name] - b[name]):.17g}; rtol=1e-4, atol=1e-6")


def choose_gate_manifest(manifest):
    chosen, scenes = [], set()
    for row in manifest["splits"]["train"]:
        if row["scene"] not in scenes:
            chosen.append(row)
            scenes.add(row["scene"])
    if len(chosen) != len(manifest["dataset"]["splits"]["train"]["scenes"]):
        raise ValueError("Gate must cover one fixed clip from each training scene")
    result = copy.deepcopy(manifest)
    result["splits"] = {"train": chosen}  # No dev data enters overfit-gate optimisation/evaluation.
    return result


def gradient_norm(module):
    values = [parameter.grad.detach() for parameter in module.parameters() if parameter.grad is not None]
    if not values or not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("Missing/nonfinite gradients in required G1 module")
    return sum(float(value.square().sum()) for value in values) ** .5


def verify_gate(root, seed, cfg, contract, entry, pair):
    path = root / f"seed{seed}" / "gate" / "gate.json"
    if not path.is_file():
        raise ValueError(f"G1 seed{seed} training-only gate must pass before formal training")
    gate = controls.read_json(path)
    gate_manifest = choose_gate_manifest(pair["manifest"])
    expected_ids = [row["id"] for row in gate_manifest["splits"]["train"]]
    if (gate["status"] != "passed" or gate["seed"] != seed or gate["steps"] != cfg["gate"]["steps"]
            or not gate["formal_must_reset"] or not gate["gate_weights_discarded"] or not gate["baseline_replayed"]
            or gate["experiment_sha256"] != controls.file_sha256(root / "gru_experiment.json")
            or gate["method_initial_state_sha256"] != entry["method_initial_state_sha256"]
            or gate["training_clip_ids"] != expected_ids
            or not math.isfinite(gate["zero_iter_max_abs_diff"])
            or not 0 <= gate["zero_iter_max_abs_diff"] <= 1e-6):
        raise ValueError("Invalid/foreign G1 gate certificate")
    validate_metrics(gate["initial"], gate_manifest)
    validate_metrics(gate["final"], gate_manifest)
    cache = controls.read_json(pair["root"] / "features.json")
    baseline = controls.read_json(pair["root"] / "arms/C1/completed.json")
    if gate["cache_sha256"] != cache["file_sha256"] or gate["cache_sha256"] != baseline["cache_sha256"]:
        raise ValueError("G1 gate cache differs from certified C1 cache")
    ratio = gate["final"]["train"]["overall"]["index_l1"] / max(gate["initial"]["train"]["overall"]["index_l1"], 1e-12)
    if (not math.isfinite(ratio) or ratio > cfg["gate"]["maximum_final_initial_loss_ratio"]
            or not math.isclose(ratio, gate["loss_ratio"], rel_tol=1e-12, abs_tol=1e-12)
            or set(gate["first_gradients"]) != REQUIRED_GRADIENTS
            or not all(math.isfinite(value) and value > 0 for value in gate["first_gradients"].values())):
        raise ValueError("G1 gate loss/gradient evidence failed")
    if controls.file_sha256(path.parent / "progress.jsonl") != gate["progress_sha256"]:
        raise ValueError("G1 gate log changed")
    return gate


def check_completed(root, seed, contract, pair, cfg, certificate=None):
    destination = root / f"seed{seed}" / "formal"
    path = destination / "completed.json"
    if certificate is None and not path.is_file():
        raise ValueError(f"G1 seed{seed} is not completed")
    certificate = controls.read_json(path) if certificate is None else certificate
    if (certificate["status"] != "completed" or certificate["seed"] != seed
            or certificate["experiment_sha256"] != controls.file_sha256(root / "gru_experiment.json")
            or certificate["method_initial_state_sha256"] != contract["seed_entries"][str(seed)]["method_initial_state_sha256"]
            or certificate["steps"] != certificate["baseline_steps"]
            or certificate["steps"] != pair["config"]["steps"]
            or certificate["baseline_completion_sha256"] != pair["completions"]
            or certificate["optimizer_initial_state_entries"] != 0):
        raise ValueError("G1 completion contract differs")
    if set(certificate["files"]) != {"final_head.pth", "metrics_final.json", "initialisation.json", "progress.jsonl", "trace.jsonl"}:
        raise ValueError("Incomplete G1 artifact inventory")
    for name, digest in certificate["files"].items():
        if controls.file_sha256(destination / name) != digest:
            raise ValueError(f"G1 completed artifact changed: {name}")
    gate = root / f"seed{seed}" / "gate" / "gate.json"
    verify_gate(root, seed, cfg, contract, contract["seed_entries"][str(seed)], pair)
    if controls.file_sha256(gate) != certificate["gate_sha256"]:
        raise ValueError("G1 gate evidence changed")
    cache = controls.read_json(pair["root"] / "features.json")
    if certificate["cache_sha256"] != cache["file_sha256"]:
        raise ValueError("Formal G1 used a different baseline cache")
    payload = torch.load(destination / "final_head.pth", map_location="cpu", weights_only=True)
    if (payload["seed"] != seed or payload["steps"] != certificate["steps"]
            or payload["method"] != cfg
            or controls.fingerprint_state(payload["model_state_dict"]) != certificate["final_state_sha256"]):
        raise ValueError("G1 checkpoint differs from completion record")
    init = controls.read_json(destination / "initialisation.json")
    if (init["method_initial_state_sha256"] != certificate["method_initial_state_sha256"]
            or init["common_c1_initial_state_sha256"] != pair["contract"]["initial_state_sha256"]
            or init["optimizer_initial_state_entries"] != 0 or init["gate_checkpoint_loaded"]
            or init["baseline_completion_sha256"] != pair["completions"]):
        raise ValueError("Formal G1 did not preserve reset/common-initial evidence")
    metrics = controls.read_json(destination / "metrics_final.json")
    validate_metrics(metrics, pair["manifest"])
    return certificate


def run_gate(output, seed, device):
    root, cfg, contract, entry, pair, state, shared = verify(output, seed)
    destination = root / f"seed{seed}" / "gate"
    destination.mkdir()  # Refuse previous/partial gates; do not tune until a gate passes.
    deadline = time.monotonic() + pair["config"]["max_seconds_per_arm"]
    try:
        # Each shell stage is a NEW process. make_gru() in prepare (or below)
        # cannot configure the earlier C1 replay in this process. Restore the
        # exact original C1 numeric policy BEFORE cache use / model evaluation.
        controls.seed_everything(int(pair["config"]["seed"]))
        numerical_policy = {
            "seed": int(pair["config"]["seed"]),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        print("C1_REPLAY_NUMERICS", json.dumps(numerical_policy), flush=True)
        clips, cache = load_cache(pair, device, deadline)
        objective = controls.build_objective(pair["config"]["objective"]["name"], pair["config"]["objective"]["kwargs"]).to(device)
        baseline = controls.build_head(pair["config"]["arms"]["C1"]["decoder"], pair["config"], pair["contract"]["backbone"]).to(device)
        final = torch.load(pair["root"] / "arms/C1/final_head.pth", map_location="cpu", weights_only=True)
        baseline.load_state_dict(final["model_state_dict"], strict=True)
        replay = controls.evaluate(baseline, clips, pair["manifest"], objective, device, deadline)
        # Preserve both sides even on replay failure; never relax the tolerance.
        controls.write_json(destination / "baseline_replay.json", {
            "numerical_policy": numerical_policy, "cache_sha256": cache["file_sha256"],
            "actual": replay, "expected": entry["baseline_metrics"],
            "relative_tolerance": 1e-4, "absolute_tolerance": 1e-6})
        validate_metrics(replay, pair["manifest"])
        assert_metric_replay(replay, entry["baseline_metrics"])
        gate_manifest = choose_gate_manifest(pair["manifest"])
        probe = instantiated(pair, cfg, state, shared, device).eval()
        probe.iters = 0
        probe.load_c1_initial(final["model_state_dict"])  # PARITY TEST ONLY, never formal/gate warm start.
        parity = 0.
        with torch.no_grad():
            for row in gate_manifest["splits"]["train"]:
                clip = controls.on_device(clips[row["id"]], device)
                expected, actual = controls.predict(baseline, clip), controls.predict(probe, clip)
                parity = max(parity, float((expected - actual).abs().max()))
                if not torch.allclose(expected, actual, atol=1e-7, rtol=1e-6):
                    raise ValueError("Zero-iteration corrected G1 did not reproduce trained C1")
        del baseline, probe, final
        head = instantiated(pair, cfg, state, shared, device)
        initial = controls.evaluate(head, clips, gate_manifest, objective, device, deadline)
        optimizer = torch.optim.AdamW(head.parameters(), lr=pair["config"]["optimizer"]["learning_rate"],
                                     weight_decay=pair["config"]["optimizer"]["weight_decay"])
        rows = gate_manifest["splits"]["train"]
        first_gradients = None
        with open(destination / "progress.jsonl", "x") as log:
            for step in range(1, cfg["gate"]["steps"] + 1):
                controls.check_deadline(deadline)
                head.train()
                clip = controls.on_device(clips[rows[(step - 1) % len(rows)]["id"]], device)
                optimizer.zero_grad(set_to_none=True)
                q = controls.predict(head, clip)
                loss = objective(q, clip["depth"], clip["mask"])["total_loss"]
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), pair["config"]["optimizer"]["grad_clip"], error_if_nonfinite=True)
                if first_gradients is None:
                    first_gradients = {name: gradient_norm(getattr(head, name)) for name in
                                       ("matcher", "classifier", "encoder", "gru_coarse", "gru_mid", "gru_fine", "index_head")}
                    if min(first_gradients.values()) <= 0:
                        raise ValueError("Zero gradient in G1 required path")
                optimizer.step()
                if step == 1 or step % 50 == 0 or step == cfg["gate"]["steps"]:
                    row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(norm.detach()),
                           **head.last_diagnostics, "iterations": head.last_iteration_diagnostics}
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    print(f"G1_GATE seed={seed}", json.dumps({k: v for k, v in row.items() if k != "iterations"}), flush=True)
        final_metrics = controls.evaluate(head, clips, gate_manifest, objective, device, deadline)
        ratio = final_metrics["train"]["overall"]["index_l1"] / max(initial["train"]["overall"]["index_l1"], 1e-12)
        if ratio > cfg["gate"]["maximum_final_initial_loss_ratio"]:
            raise ValueError(f"G1 overfit gate failed: eval loss ratio={ratio}; no formal run, do not loosen gate")
        report = {"status": "passed", "seed": seed, "steps": cfg["gate"]["steps"],
                  "experiment_sha256": controls.file_sha256(root / "gru_experiment.json"),
                  "baseline_replayed": True, "zero_iter_max_abs_diff": parity,
                  "cache_sha256": cache["file_sha256"], "method_initial_state_sha256": entry["method_initial_state_sha256"],
                  "training_clip_ids": [row["id"] for row in rows], "initial": initial, "final": final_metrics,
                  "loss_ratio": ratio, "first_gradients": first_gradients,
                  "progress_sha256": controls.file_sha256(destination / "progress.jsonl"),
                  "gate_weights_discarded": True, "formal_must_reset": True}
        controls.check_deadline(deadline)
        controls.write_json(destination / "gate.json", report)
        print(f"G1_GATE_PASSED seed={seed}; ratio={ratio:.4f}; gate weights DISCARDED before formal training.", flush=True)
        return report
    except Exception as exc:
        controls.write_json(destination / "failure.json", {"error": str(exc), "formal_allowed": False})
        raise


def train(output, seed, device):
    root, cfg, contract, entry, pair, state, shared = verify(output, seed)
    # Do not silently spend formal budget if one of the preregistered gates failed.
    for gate_seed in SEEDS:
        if not (root / f"seed{gate_seed}" / "gate" / "gate.json").is_file():
            raise ValueError("All three G1 training-only gates must pass before formal training")
        if gate_seed != seed:
            _r, _cfg, _contract, gate_entry, gate_pair, _state, _shared = verify(root, gate_seed)
            verify_gate(root, gate_seed, cfg, contract, gate_entry, gate_pair)
    gate_path = root / f"seed{seed}" / "gate" / "gate.json"
    gate = verify_gate(root, seed, cfg, contract, entry, pair)
    destination = root / f"seed{seed}" / "formal"
    destination.mkdir()  # Never resume/overwrite, never place artifacts inside C1.
    started = time.monotonic()
    deadline = started + pair["config"]["max_seconds_per_arm"]
    try:
        clips, cache = load_cache(pair, device, deadline)
        if cache["file_sha256"] != gate["cache_sha256"]:
            raise ValueError("Gate and formal feature cache differ")
        head = instantiated(pair, cfg, state, shared, device)
        if controls.fingerprint_state(head.state_dict()) != entry["method_initial_state_sha256"]:
            raise ValueError("Formal G1 failed to reset after gate")
        objective = controls.build_objective(pair["config"]["objective"]["name"], pair["config"]["objective"]["kwargs"]).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=pair["config"]["optimizer"]["learning_rate"],
                                     weight_decay=pair["config"]["optimizer"]["weight_decay"])
        controls.write_json(destination / "initialisation.json", {
            "method_initial_state_sha256": entry["method_initial_state_sha256"],
            "common_c1_initial_state_sha256": entry["common_initial_state_sha256"],
            "optimizer_initial_state_entries": len(optimizer.state), "gate_checkpoint_loaded": False,
            "baseline_completion_sha256": entry["baseline_completion_sha256"]})
        initial = controls.evaluate(head, clips, pair["manifest"], objective, device, deadline)
        controls.write_json(destination / "metrics_initial.json", initial)
        print(f"INITIAL G1 seed={seed}", json.dumps({key: value["overall"] for key, value in initial.items()}), flush=True)
        order = controls.read_json(pair["root"] / "training_order.json")
        loop_started = time.monotonic()
        final = None
        with open(destination / "progress.jsonl", "x") as log, open(destination / "trace.jsonl", "x") as trace:
            for step, clip_id in enumerate(order, start=1):
                controls.check_deadline(deadline)
                head.train()
                clip = controls.on_device(clips[clip_id], device)
                optimizer.zero_grad(set_to_none=True)
                q = controls.predict(head, clip)
                loss = objective(q, clip["depth"], clip["mask"])["total_loss"]
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), pair["config"]["optimizer"]["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                if step == 1 or step % pair["config"]["log_every"] == 0 or step == pair["config"]["steps"]:
                    elapsed = time.monotonic() - loop_started
                    row = {"seed": seed, "step": step, "clip_id": clip_id, "loss": float(loss.detach()),
                           "grad_norm": float(norm.detach()), "loop_seconds": elapsed,
                           "eta_seconds": elapsed / step * (len(order) - step), **head.last_diagnostics}
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    trace.write(json.dumps({"step": step, "iterations": head.last_iteration_diagnostics}, allow_nan=False) + "\n")
                    log.flush()
                    trace.flush()
                    print("G1", json.dumps(row), flush=True)
                if step % pair["config"]["eval_every"] == 0 or step == pair["config"]["steps"]:
                    final = controls.evaluate(head, clips, pair["manifest"], objective, device, deadline)
                    controls.write_json(destination / f"metrics_step_{step:06d}.json", final)
                    print(f"EVAL G1 seed={seed} step={step}", json.dumps({key: value["overall"] for key, value in final.items()}), flush=True)
        controls.check_deadline(deadline)
        controls.write_json(destination / "metrics_final.json", final)
        final_state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
        controls.save_state(destination / "final_head.pth", {"model_state_dict": final_state, "seed": seed,
                            "steps": pair["config"]["steps"], "method": cfg})
        files = ("final_head.pth", "metrics_final.json", "initialisation.json", "progress.jsonl", "trace.jsonl")
        certificate = {"status": "completed", "seed": seed, "steps": len(order), "baseline_steps": pair["config"]["steps"],
                       "experiment_sha256": controls.file_sha256(root / "gru_experiment.json"),
                       "baseline_completion_sha256": pair["completions"],
                       "method_initial_state_sha256": entry["method_initial_state_sha256"], "optimizer_initial_state_entries": 0,
                       "gate_sha256": controls.file_sha256(gate_path), "cache_sha256": cache["file_sha256"],
                       "final_state_sha256": controls.fingerprint_state(final_state), "elapsed_seconds": time.monotonic() - started,
                       "files": {name: controls.file_sha256(destination / name) for name in files}}
        check_completed(root, seed, contract, pair, cfg, certificate=certificate)
        controls.check_deadline(deadline)
        controls.write_json(destination / "completed.json", certificate)
        print(f"COMPLETED G1 seed={seed}: {destination}", flush=True)
        return certificate
    except Exception as exc:
        if not (destination / "completed.json").exists():
            controls.write_json(destination / "failure.json", {"error": str(exc), "status": "failed"})
        raise


def summarize(output):
    root = Path(output).resolve()
    baseline_values, method_values = {}, {}
    certificates = {}
    for seed in SEEDS:
        _, cfg, contract, entry, pair, _state, _shared = verify(root, seed)
        certificate = check_completed(root, seed, contract, pair, cfg)
        if certificate["steps"] != pair["config"]["steps"]:
            raise ValueError("Method/baseline final-step budgets differ")
        method = controls.read_json(root / f"seed{seed}" / "formal" / "metrics_final.json")
        for split, records in pair["manifest"]["splits"].items():
            if [(r["id"], r["valid_pixels"]) for r in method[split]["clips"]] != [(r["id"], r["valid_pixels"]) for r in records]:
                raise ValueError("G1 final metrics changed sample/support inventory")
        baseline_values[seed], method_values[seed] = entry["baseline_metrics"], method
        certificates[str(seed)] = controls.file_sha256(root / f"seed{seed}" / "formal" / "completed.json")
    rows = []
    for split, records in pair["manifest"]["splits"].items():
        for scene in sorted({r["scene"] for r in records}):
            for metric in ("index_l1", "absrel", "rmse", "delta1"):
                c1 = [baseline_values[s][split]["scenes"][scene][metric] for s in SEEDS]
                g1 = [method_values[s][split]["scenes"][scene][metric] for s in SEEDS]
                diff = [b - a for a, b in zip(c1, g1)]
                stats = lambda x: {"values": x, "mean": statistics.mean(x), "sample_std": statistics.stdev(x)}
                rows.append({"split": split, "scene": scene, "metric": metric,
                             "C1": stats(c1), "G1": stats(g1), "paired_G1_minus_C1": stats(diff)})
    report = {"status": "all_three_corrected_gru_seeds_complete", "method": cfg,
              "final_step": pair["config"]["steps"], "seeds": list(SEEDS), "certificates": certificates, "rows": rows,
              "head_parameters": {"C1": entry["baseline_parameters"], "G1": entry["method_parameters"]},
              "interpretation": "Same C1 data/init/optimizer/formal budget; extra GRU parameters/compute and bounded update disclosed. Gates discarded. GT-camera development only, not official IGEV/RGB-only/SOTA."}
    path = root / "comparison.json"
    if path.exists():
        if controls.read_json(path) != report:
            raise ValueError("Existing G1 comparison differs")
    else:
        controls.write_json(path, report)
    print(f"G1_FINAL_STEP={report['final_step']}; seeds=0,1,2; paired against completed C1")
    for row in rows:
        print(row["split"], row["scene"], row["metric"],
              *[f"{row[key]['mean']:.6f}+/-{row[key]['sample_std']:.6f}" for key in ("C1", "G1", "paired_G1_minus_C1")],
              row["paired_G1_minus_C1"]["values"])
    print(report["interpretation"], "Output:", path, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--repetitions", required=True)
    setup.add_argument("--output", required=True)
    setup.add_argument("--config", default=str(ROOT / "config/stereogru/corrected_gru.yaml"))
    for name in ("gate", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", required=True)
        sub.add_argument("--seed", type=int, choices=SEEDS, required=True)
    result = commands.add_parser("summarize")
    result.add_argument("--output", required=True)
    args = parser.parse_args()
    # Production CLI is fixed; shortened CPU fixtures call the Python API only.
    config_path = args.config if args.command == "prepare" else Path(args.output) / "method.json"
    production = method_config(config_path)
    if production["gate"] != {"steps": 200, "clips_per_train_scene": 1, "maximum_final_initial_loss_ratio": .5}:
        raise ValueError("Production G1 requires the fixed 200-step / 0.5 disposable gate")
    if args.command in ("gate", "train") and not torch.cuda.is_available():
        raise RuntimeError("Real G1 runs require an idle CUDA GPU; CPU contract fixtures are separate")
    actions = {"prepare": lambda: prepare(args.repetitions, args.output, args.config),
               "gate": lambda: run_gate(args.output, args.seed, torch.device("cuda")),
               "train": lambda: train(args.output, args.seed, torch.device("cuda")),
               "summarize": lambda: summarize(args.output)}
    actions[args.command]()


if __name__ == "__main__":
    main()