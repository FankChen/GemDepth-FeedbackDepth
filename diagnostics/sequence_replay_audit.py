"""Read-only audit of a completed optimisation rejected by the old replay guard.

This file lives OUTSIDE the five frozen experiment source trees deliberately.
It does not alter their inventory, issue completion certificates, resume any
training, or relax the existing replay tolerance. It evaluates SAVED weights
and performs six disposable single updates (three paths x two training clips).
Only a new external report directory is written. Failed-run hashes observed
now are not misrepresented as hashes certified before the original failure.
"""

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
with redirect_stdout(sys.stderr):
    import torch
    import stereogru_sequence_experiment as experiment

controls = experiment.controls
legacy = experiment.legacy
RTOL, ATOL = 1e-4, 1e-6


def file_inventory(root):
    return {str(path.relative_to(root)): controls.file_sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def validate_failed_baseline(context, seed):
    """Validate all retained artifacts, but never forge a completed certificate."""
    destination = experiment.directory(context, seed, "B0", "formal")
    if (destination / "completed.json").exists():
        raise ValueError("Expected failed B0 without a completion certificate")
    pair = context["pairs"][seed]
    cfg = pair["config"]
    failure = controls.read_json(destination / "failure.json")
    if (failure.get("status") != "failed" or failure.get("next_arm_allowed") is not False
            or not failure.get("error", "").startswith("C1 final metrics did not replay:")):
        raise ValueError("This diagnostic only accepts the known cross-training replay failure")
    names = experiment.artifact_names(cfg["steps"], cfg["eval_every"], True)
    if {path.name for path in destination.iterdir()} != names | {"failure.json"}:
        raise ValueError("Failed baseline artifact inventory differs")
    # No completion certificate survived. These are CURRENT observed hashes used
    # for structural validation, not fabricated historical completion evidence.
    observed = {"files": {name: controls.file_sha256(destination / name) for name in names}}
    order = controls.read_json(pair["root"] / "training_order.json")
    initial, final = experiment.validate_run_artifacts(
        context, seed, "B0", "formal", observed, pair["manifest"], order,
        cfg["eval_every"], cfg["log_every"], formal=True)
    payload = torch.load(destination / "final_head.pth", map_location="cpu", weights_only=True)
    expected = {"seed": seed, "arm": "B0", "steps": cfg["steps"],
                "experiment_sha256": context["digest"],
                "initial_state_sha256": experiment.entry_for(context, seed, "B0")["initial_state_sha256"]}
    if (set(payload) != set(expected) | {"model_state_dict"}
            or any(payload[key] != value for key, value in expected.items())
            or any(not torch.isfinite(value).all() for value in payload["model_state_dict"].values())):
        raise ValueError("Failed baseline checkpoint metadata/state differs")
    head = experiment.instantiate(context, seed, "B0", torch.device("cpu"))
    head.load_state_dict(payload["model_state_dict"], strict=True)
    try:
        legacy.assert_metric_replay(final["stages"][-1], pair["metrics"]["C1"])
    except ValueError as exc:
        if str(exc) != failure["error"]:
            raise ValueError("Stored failure no longer matches the retained metrics") from exc
    else:
        raise ValueError("Retained metrics do not reproduce the claimed cross-training failure")
    return payload["model_state_dict"], initial, final


def compare_metrics(actual, expected):
    """Fixed weights only: aggregates AND individual clips, using unchanged tolerances."""
    differences, maximum = [], 0.
    if set(actual) != set(expected):
        raise ValueError("Replay split inventory differs")
    for split in expected:
        left, right = actual[split], expected[split]
        if (set(left["scenes"]) != set(right["scenes"])
                or [(r["id"], r["scene"]) for r in left["clips"]]
                != [(r["id"], r["scene"]) for r in right["clips"]]):
            raise ValueError("Replay scene/clip inventory differs")
        groups = [("overall", left["overall"], right["overall"])]
        groups += [(scene, left["scenes"][scene], right["scenes"][scene]) for scene in right["scenes"]]
        groups += [(a["id"], a, b) for a, b in zip(left["clips"], right["clips"])]
        for group, a, b in groups:
            if a["valid_pixels"] != b["valid_pixels"]:
                raise ValueError("Replay GT support changed")
            for key in ("index_l1", "absrel", "rmse", "delta1"):
                x, y = a[key], b[key]
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("Nonfinite replay metric")
                maximum = max(maximum, abs(x - y))
                if not math.isclose(x, y, rel_tol=RTOL, abs_tol=ATOL):
                    differences.append({"split": split, "group": group, "metric": key,
                                        "actual": x, "expected": y, "abs_diff": abs(x - y)})
    return {"passed": not differences, "max_abs_diff": maximum, "differences": differences}


def compare_tensors(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("Tensor inventory differs")
    differences, exact, maximum = [], True, 0.
    for name, a in actual.items():
        b = expected[name]
        if a is None or b is None:
            if a is not None or b is not None:
                differences.append({"name": name, "reason": "gradient presence differs"})
                exact = False
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Tensor shape/dtype differs: {name}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise FloatingPointError(f"Nonfinite audited tensor: {name}")
        error = float((a.double() - b.double()).abs().max())
        maximum = max(maximum, error)
        exact = exact and torch.equal(a, b)
        if not torch.allclose(a, b, rtol=RTOL, atol=ATOL):
            differences.append({"name": name, "max_abs_diff": error})
    return {"passed": not differences, "bitwise_equal": exact,
            "max_abs_diff": maximum, "differences": differences}


def old_head(pair, state, device):
    head = controls.build_head(pair["config"]["arms"]["C1"]["decoder"],
                               pair["config"], pair["contract"]["backbone"])
    head.load_state_dict(state, strict=True)
    return head.to(device)


def single_update(head, clip, objective, predict, seed, optimizer_config):
    """Disposable, never serialized; matches formal AdamW + clipping settings."""
    controls.seed_everything(seed)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=optimizer_config["learning_rate"],
                                 weight_decay=optimizer_config["weight_decay"])
    if optimizer.state:
        raise ValueError("Diagnostic optimizer is not fresh")
    outputs = predict(head, clip)
    loss = objective(outputs, clip["depth"], clip["mask"])["total_loss"]
    loss.backward()
    gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                 for name, p in head.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), optimizer_config["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    final = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
    return {"forward_and_loss": {"q": final.detach().cpu(), "loss": loss.detach().cpu()},
            "gradients": gradients,
            "updated_state": {name: value.detach().cpu().clone() for name, value in head.state_dict().items()},
            "grad_norm_before_clip": float(norm)}


def update_probe(context, seed, clip, device):
    pair = context["pairs"][seed]
    initial = torch.load(pair["root"] / "initial_head.pth", map_location="cpu", weights_only=True)
    heads = [old_head(pair, initial, device), old_head(pair, initial, device),
             experiment.instantiate(context, seed, "B0", device)]
    objective = experiment.objective_for(pair, context["config"]["arms"]["B0"]).to(device)
    results = []
    for head, loss, predict in zip(heads, [objective.base, objective.base, objective],
                                   [controls.predict, controls.predict, experiment.predict_sequence]):
        if controls.fingerprint_state(head.state_dict()) != pair["contract"]["initial_state_sha256"]:
            raise ValueError("Probe paths do not share identical INITIAL tensors")
        results.append(single_update(head, clip, loss, predict, seed, pair["config"]["optimizer"]))
    comparisons = {}
    for label, index in (("old_vs_old_repeatability", 1), ("new_vs_old_equivalence", 2)):
        checks = {key: compare_tensors(results[index][key], results[0][key])
                  for key in ("forward_and_loss", "gradients", "updated_state")}
        comparisons[label] = {"passed": all(v["passed"] for v in checks.values()), "checks": checks}
    return {"comparisons": comparisons, "grad_norm_before_clip": [r["grad_norm_before_clip"] for r in results],
            "disposable_updates": 3, "checkpoint_saved": False}


def audit(output, report_root, seed, device, *, _context=None):
    context = experiment.verify(output) if _context is None else _context
    controls.seed_everything(seed)
    pair = context["pairs"][seed]
    destination = Path(report_root).resolve()
    protected = [ROOT, context["root"], Path(context["contract"]["repetitions"])] + [p["root"] for p in context["pairs"].values()]
    if any(destination == p or destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected):
        raise ValueError("Audit report must be outside all code/experiment/source directories")
    destination.mkdir(parents=True, exist_ok=False)
    before = file_inventory(context["root"])
    report = {"status": "incomplete", "seed": seed, "experiment": str(context["root"]),
              "experiment_sha256": context["digest"], "audit_source_sha256": controls.file_sha256(Path(__file__)),
              "observed_failed_run_files": before, "rtol": RTOL, "atol": ATOL,
              "method_authorized": False, "completion_certificate_written": False,
              "formal_training_updates": 0, "runtime": controls.runtime_versions(),
              "numerical_policy": experiment.numeric_policy(), "device": str(device),
              "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}
    try:
        state, saved_initial, saved_final = validate_failed_baseline(context, seed)
        report["retained_b0_state_sha256_observed_now"] = controls.fingerprint_state(state)
        report["cross_training_difference_not_a_replay"] = compare_metrics(saved_final["stages"][-1], pair["metrics"]["C1"])
        deadline = time.monotonic() + pair["config"]["max_seconds_per_arm"]
        clips, cache = legacy.load_cache(pair, device, deadline)
        report["cache_sha256"] = cache["file_sha256"]
        historical_path = pair["root"] / "arms/C1/final_head.pth"
        historical = torch.load(historical_path, map_location="cpu", weights_only=True)["model_state_dict"]
        shared = torch.load(pair["root"] / "initial_head.pth", map_location="cpu", weights_only=True)
        objective = experiment.objective_for(pair, context["config"]["arms"]["B0"]).to(device)
        checks, metrics = {}, {}
        for label, weights, expected in (("historical_c1", historical, pair["metrics"]["C1"]),
                                         ("retained_b0", state, saved_final["stages"][-1]),
                                         ("common_initial", shared, saved_initial["stages"][-1])):
            controls.check_deadline(deadline)
            head = old_head(pair, weights, device)
            original = controls.fingerprint_state(head.state_dict())
            old_metrics = controls.evaluate(head, clips, pair["manifest"], objective.base, device, deadline)
            again = controls.evaluate(head, clips, pair["manifest"], objective.base, device, deadline)
            if controls.fingerprint_state(head.state_dict()) != original:
                raise ValueError("Fixed-checkpoint eval mutated parameters/BN")
            del head
            head = experiment.instantiate(context, seed, "B0", device)
            head.load_state_dict(weights, strict=True)
            current = experiment.evaluate_sequence(head, clips, pair["manifest"], objective, device, deadline)["stages"][0]
            if controls.fingerprint_state(head.state_dict()) != original:
                raise ValueError("Sequence eval mutated parameters/BN")
            del head
            metrics[label] = {"saved": expected, "old_path": old_metrics, "old_repeat": again, "sequence_path": current}
            checks[label] = {"old_vs_saved": compare_metrics(old_metrics, expected),
                             "old_repeat": compare_metrics(again, old_metrics),
                             "new_vs_old": compare_metrics(current, old_metrics),
                             "new_vs_saved": compare_metrics(current, expected)}
            print("FIXED_WEIGHT_REPLAY", label, json.dumps({k: v["passed"] for k, v in checks[label].items()}), flush=True)
        report["fixed_weight_replay"] = checks
        controls.write_json(destination / "replayed_metrics.json", metrics)
        report["replayed_metrics_sha256"] = controls.file_sha256(destination / "replayed_metrics.json")
        probes = []
        gate = legacy.choose_gate_manifest(pair["manifest"])
        for row in gate["splits"]["train"]:
            controls.check_deadline(deadline)
            item = update_probe(context, seed, controls.on_device(clips[row["id"]], device), device)
            item["clip_id"] = row["id"]
            probes.append(item)
            print("ONE_UPDATE_PROBE", row["id"], json.dumps({k: v["passed"] for k, v in item["comparisons"].items()}), flush=True)
        report["update_probes"] = probes
        report["disposable_diagnostic_updates"] = sum(p["disposable_updates"] for p in probes)
        if file_inventory(context["root"]) != before:
            raise ValueError("Original failed experiment changed during audit")
        experiment.verify(output)  # Recheck all frozen sources, historical pairs/cache/initial files.
        controls.check_deadline(deadline)
        report["original_files_unchanged"] = True
        report["all_fixed_weight_checks_passed"] = all(v["passed"] for c in checks.values() for v in c.values())
        report["all_update_checks_passed"] = all(v["passed"] for p in probes for v in p["comparisons"].values())
        report["status"] = "audit_complete"
        report["interpretation"] = (
            "Even all checks passing establishes only fixed-weight replay and local one-update equivalence. "
            "It does not identify the historical 1000-step divergence cause, guarantee CUDA determinism, "
            "approve the failed baseline, or authorize F1/S1. No thresholds or old certificates were changed.")
    except Exception as exc:
        report["status"] = "audit_failed"
        report["error"] = str(exc)
        try:
            report["original_files_unchanged"] = file_inventory(context["root"]) == before
            experiment.verify(output)
            report["source_recheck_passed"] = True
        except Exception as check_error:
            report["source_recheck_passed"] = False
            report["source_recheck_error"] = str(check_error)
        controls.write_json(destination / "audit.json", report)
        raise
    controls.write_json(destination / "audit.json", report)
    print("AUDIT_COMPLETE", json.dumps({key: report[key] for key in
          ("all_fixed_weight_checks_passed", "all_update_checks_passed", "original_files_unchanged", "method_authorized")}), flush=True)
    print("AUDIT_REPORT", destination, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, choices=experiment.SEEDS, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Remote diagnostic requires CUDA; CPU tests call the API explicitly")
    print("AUDIT_VERIFY: frozen sources and all baseline certificates/cache; no training", flush=True)
    context = experiment.verify(args.experiment)
    experiment.production_guard(context["config"], context["pairs"])
    audit(args.experiment, args.report, args.seed, torch.device("cuda"), _context=context)


if __name__ == "__main__":
    main()