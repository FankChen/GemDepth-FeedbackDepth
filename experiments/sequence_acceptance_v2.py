"""Explicit acceptance amendment, NOT a change to optimisation or old evidence.

v1 wrongly compared independently trained terminal states as checkpoint replay.
v2 validates full update provenance and replays each run's OWN saved checkpoint.
Seed0 B0 may be referenced from the completed audit; its old failure is untouched.
All new stages still execute the frozen v1 optimisation, config, gates and reset.
"""

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "diagnostics")]
with redirect_stdout(sys.stderr):
    import torch
    import stereogru_sequence_experiment as old
    import sequence_replay_audit as audit

c = old.controls
POLICY = "own_saved_checkpoint_replay_v2_not_cross_training_equality"
EXTENSIONS = ("experiments/sequence_acceptance_v2.py", "experiments/run_sequence_v2.sh",
              "experiments/start_sequence_v2.sh", "diagnostics/sequence_replay_audit.py")


def verify_audit(source, report_root):
    """Use the existing evidence, no new optimisation or GPU diagnostics."""
    report_root = Path(report_root).resolve()
    report = c.read_json(report_root / "audit.json")
    if (report["status"] != "audit_complete" or report["seed"] != 0
            or report["experiment"] != str(source["root"])
            or report["experiment_sha256"] != source["digest"]
            or report["runtime"] != source["contract"]["runtime"]
            or report["audit_source_sha256"] != c.file_sha256(ROOT / "diagnostics/sequence_replay_audit.py")
            or report["rtol"] != audit.RTOL or report["atol"] != audit.ATOL
            or report["original_files_unchanged"] is not True
            or report["method_authorized"] is not False
            or report["completion_certificate_written"] is not False
            or report["formal_training_updates"] != 0
            or report["disposable_diagnostic_updates"] != 6
            or report["all_fixed_weight_checks_passed"] is not True
            or report["all_update_checks_passed"] is not False
            or report["numerical_policy"] != {"float32_matmul_precision": "highest",
                "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "cudnn_benchmark": False}
            or report["observed_failed_run_files"] != audit.file_inventory(source["root"])
            or report["replayed_metrics_sha256"] != c.file_sha256(report_root / "replayed_metrics.json")):
        raise ValueError("Audit does not bind the unchanged failed experiment and fixed replay policy")
    state, initial, final = audit.validate_failed_baseline(source, 0)
    if (report["retained_b0_state_sha256_observed_now"] != c.fingerprint_state(state)
            or report["cache_sha256"] != source["contract"]["seed_entries"]["0"]["cache_sha256"]):
        raise ValueError("Audit saved-state/cache binding differs")
    metrics = c.read_json(report_root / "replayed_metrics.json")
    expected = {"historical_c1": source["pairs"][0]["metrics"]["C1"],
                "retained_b0": final["stages"][-1], "common_initial": initial["stages"][-1]}
    if set(metrics) != set(expected) or set(report["fixed_weight_replay"]) != set(expected):
        raise ValueError("Audit omitted fixed-weight replay paths")
    for label, saved in expected.items():
        values = metrics[label]
        if set(values) != {"saved", "old_path", "old_repeat", "sequence_path"} or values["saved"] != saved:
            raise ValueError("Audit saved metrics differ from original artifacts")
        for value in values.values():
            old.legacy.validate_metrics(value, source["pairs"][0]["manifest"])
        checks = {"old_vs_saved": audit.compare_metrics(values["old_path"], saved),
                  "old_repeat": audit.compare_metrics(values["old_repeat"], values["old_path"]),
                  "new_vs_old": audit.compare_metrics(values["sequence_path"], values["old_path"]),
                  "new_vs_saved": audit.compare_metrics(values["sequence_path"], saved)}
        if checks != report["fixed_weight_replay"][label] or not all(v["passed"] for v in checks.values()):
            raise ValueError("Own-checkpoint replay did not pass unchanged tolerances")
    rows = old.legacy.choose_gate_manifest(source["pairs"][0]["manifest"])["splits"]["train"]
    if [p["clip_id"] for p in report["update_probes"]] != [r["id"] for r in rows]:
        raise ValueError("Audit probe clip inventory differs")
    for probe in report["update_probes"]:
        if probe["disposable_updates"] != 3 or probe["checkpoint_saved"] is not False:
            raise ValueError("Probe update accounting differs")
        if set(probe["comparisons"]) != {"old_vs_old_repeatability", "new_vs_old_equivalence"}:
            raise ValueError("Missing old-old/new-old controls")
        for comparison in probe["comparisons"].values():
            if set(comparison["checks"]) != {"forward_and_loss", "gradients", "updated_state"}:
                raise ValueError("Missing probe evidence")
            for name, check in comparison["checks"].items():
                if not math.isfinite(check["max_abs_diff"]) or check["max_abs_diff"] < 0:
                    raise ValueError("Invalid probe differences")
                if name != "updated_state" and (check["passed"] is not True or check["differences"]):
                    raise ValueError("Forward/gradient equivalence not established")
                if name == "updated_state" and (check["passed"] is not False or not check["differences"]
                                                or check["bitwise_equal"] is not False):
                    raise ValueError("Recorded updated-state failure must not be relabeled as passed")
            if comparison["passed"] is not False:
                raise ValueError("Probe aggregate failure must remain recorded")
            # Update equality is recorded, never relabeled as passed. No optimizer
            # eps/LR change and no new numerical threshold are introduced here.
    return {"audit_sha256": c.file_sha256(report_root / "audit.json"),
            "metrics_sha256": c.file_sha256(report_root / "replayed_metrics.json"),
            "original_inventory": report["observed_failed_run_files"],
            "retained_state_sha256": c.fingerprint_state(state)}, final["stages"][-1]


def prepare(source_root, audit_root, output):
    source = old.verify(source_root)
    evidence, metrics = verify_audit(source, audit_root)
    root = Path(output).resolve()
    protected = [ROOT, source["root"], Path(audit_root).resolve(), Path(source["contract"]["repetitions"])]
    protected += [pair["root"] for pair in source["pairs"].values()]
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError("Revision must be outside all old evidence/code directories")
    # Explicitly limited to the sole already-run seed0 B0, not selective result import.
    for seed in old.SEEDS:
        for arm in source["config"]["arms"]:
            if (seed, arm) != (0, "B0") and (source["root"] / f"seed{seed}/arms/{arm}").exists():
                raise ValueError("Unexpected old runs; do not selectively reuse a result")
    root.mkdir(parents=True, exist_ok=False)
    names = {"experiment.json", "method.json"}
    names.update(item["initial_file"] for entry in source["contract"]["seed_entries"].values()
                 for item in entry["arms"].values())
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source["root"] / name, target)
    revision = {"policy": POLICY, "source": str(source["root"]), "audit": str(Path(audit_root).resolve()),
                "original_contract_sha256": source["digest"], "evidence": evidence,
                "extensions": {name: c.file_sha256(ROOT / name) for name in EXTENSIONS},
                "retained_run": {"seed": 0, "arm": "B0", "steps": source["pairs"][0]["config"]["steps"]},
                "certification_scope": "Audit hashes first observed at v2 prepare, not authenticated historical signatures; "
                                       "structural/metric recomputation detects inconsistent changes, not coordinated forged reports.",
                "amendment": "Validation only, after seeing baseline-only evidence; all training configs/weights/budget unchanged. "
                             "Old failed certification remains failed; reuse is separately certified under v2. "
                             "Historical repeated-training terminal equality is not a valid fixed-checkpoint replay test."}
    c.write_json(root / "acceptance_revision.json", revision)
    context = verify(root)
    destination = old.directory(context, 0, "B0", "formal")
    destination.mkdir(parents=True)
    c.write_json(destination / "metrics_final.json", metrics)
    c.write_json(destination / "completed.json", {"kind": "audited_existing_run", "policy": POLICY,
        "revision_sha256": context["digest"], "source": str(source["root"]), "evidence": evidence,
        "steps": revision["retained_run"]["steps"], "new_optimizer_updates": 0,
        "metrics_sha256": c.file_sha256(destination / "metrics_final.json")})
    check_completed(context, 0, "B0")
    print("B0_SEED0_REUSED", f"steps={revision['retained_run']['steps']}; new_updates=0; old failure untouched", flush=True)
    print("V2_PREPARED", root, "NEXT: real B0 seed1 training", flush=True)


def verify(output):
    root = Path(output).resolve()
    revision = c.read_json(root / "acceptance_revision.json")
    if revision["policy"] != POLICY or set(revision["extensions"]) != set(EXTENSIONS):
        raise ValueError("Unknown acceptance revision")
    for name, digest in revision["extensions"].items():
        if c.file_sha256(ROOT / name) != digest:
            raise ValueError(f"Acceptance source changed: {name}")
    source = old.verify(revision["source"])
    evidence, _ = verify_audit(source, revision["audit"])
    context = old.verify(root)
    if (evidence != revision["evidence"] or context["digest"] != source["digest"]
            or context["digest"] != revision["original_contract_sha256"]):
        raise ValueError("Amended experiment changed original inputs/initials/config or audit")
    context.update(digest=c.file_sha256(root / "acceptance_revision.json"), revision=revision)
    return context


def check_completed(context, seed, arm, certificate=None):
    destination = old.directory(context, seed, arm, "formal")
    path = destination / "completed.json"
    if certificate is None and not path.is_file():
        raise ValueError(f"Required baseline/control {arm} seed{seed} not completed")
    report = c.read_json(path) if certificate is None else certificate
    if (seed, arm) == (0, "B0"):
        revision = context["revision"]
        expected = {"kind": "audited_existing_run", "policy": POLICY, "revision_sha256": context["digest"],
                    "source": revision["source"], "evidence": revision["evidence"],
                    "steps": context["pairs"][0]["config"]["steps"], "new_optimizer_updates": 0,
                    "metrics_sha256": c.file_sha256(destination / "metrics_final.json")}
        if report != expected or c.read_json(destination / "metrics_final.json") != c.read_json(
                Path(revision["source"]) / "seed0/arms/B0/formal/metrics_final.json"):
            raise ValueError("Reused baseline evidence differs")
        return report
    pair = context["pairs"][seed]
    cfg = pair["config"]
    old.verify_certificate_header(context, seed, arm, report, "completed", cfg["steps"])
    if report["policy"] != POLICY or report["gate_certificates"] != old.gate_hashes(context, arm):
        raise ValueError("Completion lacks acceptance policy/all method gates")
    order = c.read_json(pair["root"] / "training_order.json")
    _, final = old.validate_run_artifacts(context, seed, arm, "formal", report, pair["manifest"], order,
                                         cfg["eval_every"], cfg["log_every"], formal=True)
    payload = torch.load(destination / "final_head.pth", map_location="cpu", weights_only=True)
    if (set(payload) != {"seed", "arm", "steps", "experiment_sha256", "initial_state_sha256", "model_state_dict"}
            or any(payload[key] != report[key] for key in payload if key != "model_state_dict")
            or c.fingerprint_state(payload["model_state_dict"]) != report["final_state_sha256"]
            or any(not torch.isfinite(v).all() for v in payload["model_state_dict"].values())):
        raise ValueError("Saved model differs from completion certificate")
    head = old.instantiate(context, seed, arm, torch.device("cpu"))
    head.load_state_dict(payload["model_state_dict"], strict=True)
    if c.file_sha256(destination / "saved_weight_replay.json") != report["replay_sha256"]:
        raise ValueError("Saved-weight replay changed")
    replay = c.read_json(destination / "saved_weight_replay.json")
    if (replay["checkpoint_sha256"] != report["files"]["final_head.pth"]
            or replay["rtol"] != audit.RTOL or replay["atol"] != audit.ATOL
            or replay["state_unchanged"] is not True):
        raise ValueError("Replay not bound to actual saved checkpoint")
    old.validate_evaluation(replay["evaluation"], pair["manifest"], old.entry_for(context, seed, arm)["weights"])
    for actual, expected in zip(replay["evaluation"]["stages"], final["stages"]):
        if not audit.compare_metrics(actual, expected)["passed"]:
            raise ValueError("Own saved checkpoint did not replay at unchanged tolerance")
    return report


def require_predecessors(context, arm):
    for required in context["config"]["arms"][arm]["requires"]:
        for seed in old.SEEDS:
            check_completed(context, seed, required)
    return old.prerequisite_hashes(context, arm)


def run(output, seed, arm, stage, device, *, _context=None):
    context = verify(output) if _context is None else _context
    prerequisites = require_predecessors(context, arm)
    pair, spec = context["pairs"][seed], context["config"]["arms"][arm]
    cfg = pair["config"]
    if stage == "gate" and not spec["gate_required"]:
        raise ValueError("No B0 gate")
    gates = old.gate_hashes(context, arm) if stage == "formal" else {}
    if (seed, arm, stage) == (0, "B0", "formal"):
        raise ValueError("Seed0 B0 is retained, never retrained or used as method initialization")
    destination = old.directory(context, seed, arm, stage)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = old.legacy.choose_gate_manifest(pair["manifest"]) if stage == "gate" else pair["manifest"]
    order = old.gate_order(manifest, context["config"]["gate"]["steps"]) if stage == "gate" else c.read_json(pair["root"] / "training_order.json")
    eval_every, log_every = (len(order), 50) if stage == "gate" else (cfg["eval_every"], cfg["log_every"])
    print(f"STAGE_RUNNING stage={stage} arm={arm} seed={seed} target_updates={len(order)}; "
          "verification first; TRAIN step=... confirms optimizer updates", flush=True)
    try:
        head, initial, final, deadline = old.optimisation(context, seed, arm, destination, manifest, order,
                                                          device, eval_every, log_every, prerequisites)
        report = None
        if stage == "gate":
            del head
            report = old.certificate_header(context, seed, arm, len(order), "passed",
                old.artifact_names(len(order), eval_every, False), destination, prerequisites)
            report.update(ratios={"weighted_sequence": final["weighted_index_l1"]["train"] / max(initial["weighted_index_l1"]["train"], 1e-12),
                "final_index_l1": final["stages"][-1]["train"]["overall"]["index_l1"] / max(initial["stages"][-1]["train"]["overall"]["index_l1"], 1e-12)},
                gate_weights_discarded=True, formal_must_reset=True)
            old.verify_gate(context, seed, arm, report)
            c.check_deadline(deadline)
            c.write_json(destination / "gate.json", report)
            print(f"V2_GATE_PASSED arm={arm} seed={seed} updates={len(order)}; weights discarded", flush=True)
        else:
            state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            c.save_state(destination / "final_head.pth", {"seed": seed, "arm": arm, "steps": len(order),
                "experiment_sha256": context["digest"], "initial_state_sha256": old.entry_for(context, seed, arm)["initial_state_sha256"],
                "model_state_dict": state})
            del head
            c.seed_everything(seed)
            clips, _ = old.legacy.load_cache(pair, device, deadline)
            head = old.instantiate(context, seed, arm, device)
            payload = torch.load(destination / "final_head.pth", map_location="cpu", weights_only=True)
            head.load_state_dict(payload["model_state_dict"], strict=True)
            before = c.fingerprint_state(head.state_dict())
            objective = old.objective_for(pair, spec).to(device)
            replay = old.evaluate_sequence(head, clips, manifest, objective, device, deadline)
            c.write_json(destination / "saved_weight_replay.json", {"evaluation": replay,
                "checkpoint_sha256": c.file_sha256(destination / "final_head.pth"), "rtol": audit.RTOL,
                "atol": audit.ATOL, "state_unchanged": before == c.fingerprint_state(head.state_dict())})
            report = old.certificate_header(context, seed, arm, len(order), "completed",
                old.artifact_names(len(order), eval_every, True), destination, prerequisites)
            report.update(policy=POLICY, gate_certificates=gates, final_state_sha256=c.fingerprint_state(state),
                          replay_sha256=c.file_sha256(destination / "saved_weight_replay.json"))
            check_completed(context, seed, arm, report)
            c.check_deadline(deadline)
            c.write_json(destination / "completed.json", report)
            print(f"V2_COMPLETED arm={arm} seed={seed} updates={len(order)} own_checkpoint_replay=passed", flush=True)
        return report
    except Exception as exc:
        c.write_json(destination / "failure.json", {"status": "failed", "error": str(exc), "next_arm_allowed": False})
        raise


def next_action(context):
    for arm, spec in context["config"]["arms"].items():
        for stage in (["gate"] if spec["gate_required"] else []) + ["formal"]:
            for seed in old.SEEDS:
                if not old.directory(context, seed, arm, stage).exists():
                    return f"{stage} {arm} {seed}"
                (old.verify_gate if stage == "gate" else check_completed)(context, seed, arm)
    return "summary - -"


def summarize(context):
    rows, metrics, certificates = [], {}, {}
    for arm in context["config"]["arms"]:
        require_predecessors(context, arm)
        for seed in old.SEEDS:
            check_completed(context, seed, arm)
            path = old.directory(context, seed, arm, "formal")
            metrics[seed, arm] = c.read_json(path / "metrics_final.json")
            certificates[f"{seed}/{arm}"] = c.file_sha256(path / "completed.json")
    for split, records in context["pairs"][0]["manifest"]["splits"].items():
        for group in ["overall"] + sorted({r["scene"] for r in records}):
            for metric in old.repeats.METRICS:
                values = {arm: [(metrics[s, arm][split]["overall"] if group == "overall" else
                                metrics[s, arm][split]["scenes"][group])[metric] for s in old.SEEDS]
                          for arm in context["config"]["arms"]}
                rows.append({"split": split, "group": group, "metric": metric,
                    **{arm: old.repeats.distribution(v) for arm, v in values.items()},
                    **{f"{a}_minus_{b}": old.repeats.distribution([x-y for x,y in zip(values[a], values[b])])
                       for a,b in (("F1", "B0"), ("S1", "B0"), ("S1", "F1"))}})
    steps = context["pairs"][0]["config"]["steps"]
    result = {"policy": POLICY, "revision_sha256": context["digest"], "final_step": steps,
              "certificates": certificates, "rows": rows, "formal_updates_total": 9 * steps,
              "reused_updates": steps, "new_formal_updates": 8 * steps,
              "discarded_gate_updates": 6 * context["config"]["gate"]["steps"],
              "limits": "Acceptance amended after baseline-only evidence, no method results seen; no SOTA or guaranteed gain; final step/all seeds only."}
    path = context["root"] / "comparison.json"
    if path.exists():
        if c.read_json(path) != result:
            raise ValueError("Summary changed")
    else:
        c.write_json(path, result)
    print(f"V2_ALL_COMPLETED final_step={steps} seeds=0,1,2 B0/F1/S1 new_updates={8*steps} reused={steps}", flush=True)
    for row in rows:
        print(json.dumps(row), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "next", "formal", "gate", "summary"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--source")
    parser.add_argument("--audit")
    parser.add_argument("--arm", choices=("B0", "F1", "S1"))
    parser.add_argument("--seed", type=int, choices=old.SEEDS)
    args = parser.parse_args()
    if args.command == "prepare":
        source = old.verify(args.source)
        old.production_guard(source["config"], source["pairs"])
        prepare(args.source, args.audit, args.output)
        return
    with redirect_stdout(sys.stderr):
        context = verify(args.output)
        old.production_guard(context["config"], context["pairs"])
        action = next_action(context) if args.command == "next" else None
    if args.command == "next":
        print(f"V2_NEXT {action}")
    elif args.command == "summary":
        summarize(context)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Production training requires idle CUDA GPU")
        run(args.output, args.seed, args.arm, args.command, torch.device("cuda"), _context=context)


if __name__ == "__main__":
    main()