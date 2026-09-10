"""Repeat the completed C0/C1 protocol at seeds 1/2 without changing anything else.

No optimiser here: preparation and validation delegate to the unchanged matched
trainer. The shell entry runs C0 then C1 for each seed, using its baseline gate.
The seed-0 experiment remains read-only. Summaries use final1000, never best-dev.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
import stereogru_matched_controls as controls


SEEDS = (0, 1, 2)
METRICS = ("index_l1", "absrel", "rmse", "delta1")


def verify_pair(experiment, require_complete=True):
    root, cfg, manifest, contract, _initial, _order = controls.verify_experiment(experiment)
    digest = controls.file_sha256(root / "experiment.json")
    completions, metrics = {}, {}
    if require_complete:
        for arm in cfg["arms"]:
            controls.verify_completed(root, arm, cfg, contract, digest)
            completions[arm] = controls.file_sha256(root / "arms" / arm / "completed.json")
            metrics[arm] = controls.read_json(root / "arms" / arm / "metrics_final.json")["metrics"]
    return {"root": root, "config": cfg, "manifest": manifest, "contract": contract,
            "experiment_sha256": digest, "completions": completions, "metrics": metrics}


def assert_same_protocol(reference, candidate, seed):
    expected = copy.deepcopy(reference["config"])
    expected["seed"] = seed
    if candidate["config"] != expected:
        raise ValueError("Repetition may change ONLY the training seed, not loss/budget/crop/split/model")
    if (candidate["manifest"] != reference["manifest"]
            or candidate["contract"]["files"]["manifest.json"] != reference["contract"]["files"]["manifest.json"]):
        raise ValueError("Repetition changed manifest, crop, camera or GT support")
    for key in ("runtime", "source_sha256", "backbone", "head_parameters"):
        if candidate["contract"][key] != reference["contract"][key]:
            raise ValueError(f"Repetition changed common contract: {key}")


def prepare_repeats(source_experiment, output):
    reference = verify_pair(source_experiment)
    if reference["config"]["seed"] != 0:
        raise ValueError("Expected the completed seed-0 pair as the fixed reference")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Repetitions require a new empty directory; do not overwrite or resume")
    entries = {"0": {"experiment": str(reference["root"]),
                     "experiment_sha256": reference["experiment_sha256"],
                     "completion_sha256": reference["completions"]}}
    initial_states = {reference["contract"]["initial_state_sha256"]}
    for seed in SEEDS[1:]:
        cfg = copy.deepcopy(reference["config"])
        cfg["seed"] = seed
        config_path = output / f"seed{seed}.yaml"
        with open(config_path, "x") as handle:
            handle.write(OmegaConf.to_yaml(OmegaConf.create(cfg)))
        destination = output / f"seed{seed}" / "experiment"
        controls.prepare(config_path, reference["contract"]["backbone"]["path"],
                         reference["manifest"]["root"], destination)
        candidate = verify_pair(destination, require_complete=False)
        assert_same_protocol(reference, candidate, seed)
        initial = candidate["contract"]["initial_state_sha256"]
        if initial in initial_states:
            raise ValueError("Different seeds did not produce distinct initial states")
        initial_states.add(initial)
        entries[str(seed)] = {"experiment": str(destination),
                              "experiment_sha256": candidate["experiment_sha256"]}
    plan = {"status": "prepared_paired_repetitions", "seeds": list(SEEDS), "experiments": entries,
            "manifest_sha256": reference["contract"]["files"]["manifest.json"],
            "script_sha256": controls.file_sha256(Path(__file__)),
            "remaining_training_runs": ["seed1/C0", "seed1/C1", "seed2/C0", "seed2/C1"],
            "only_changed_setting": "seed (head initialization and training order); crop_seed stays fixed",
            "checkpoint_rule": "prespecified final step for every seed, not best development checkpoint"}
    controls.write_json(output / "repetitions.json", plan)
    print("PAIRED_REPEATS_PREPARED: seeds 1/2; identical manifest/crop/model/loss/budget; no training started.", flush=True)
    return plan


def distribution(values):
    return {"values": values, "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values)}


def aggregate_pairs(pairs):
    if sorted(pairs) != list(SEEDS):
        raise ValueError("Exactly the prespecified seeds 0/1/2 must all be complete; no cherry-picked seeds")
    reference = pairs[0]
    result = {}
    for seed, pair in pairs.items():
        assert_same_protocol(reference, pair, seed)
    for split in reference["manifest"]["splits"]:
        result[split] = {}
        scenes = sorted({row["scene"] for row in reference["manifest"]["splits"][split]})
        for scene in scenes:
            row = {}
            for metric in METRICS:
                c0 = [pairs[seed]["metrics"]["C0"][split]["scenes"][scene][metric] for seed in SEEDS]
                c1 = [pairs[seed]["metrics"]["C1"][split]["scenes"][scene][metric] for seed in SEEDS]
                row[metric] = {"C0": distribution(c0), "C1": distribution(c1),
                               "paired_C1_minus_C0": distribution([b - a for a, b in zip(c0, c1)])}
            result[split][scene] = row
    return result


def summarize_repeats(output):
    output = Path(output).resolve()
    plan = controls.read_json(output / "repetitions.json")
    if plan["seeds"] != list(SEEDS) or set(plan["experiments"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("Repetition inventory changed")
    if controls.file_sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("Repetition script changed; do not mix aggregation rules")
    pairs = {}
    for seed in SEEDS:
        entry = plan["experiments"][str(seed)]
        pair = verify_pair(entry["experiment"])
        if pair["experiment_sha256"] != entry["experiment_sha256"]:
            raise ValueError(f"Prepared seed-{seed} contract changed")
        if seed == 0 and pair["completions"] != entry["completion_sha256"]:
            raise ValueError("Original seed-0 completion artifacts changed")
        pairs[seed] = pair
    metrics = aggregate_pairs(pairs)
    report = {"status": "three_paired_seeds_complete", "seeds": list(SEEDS),
              "steps_each": pairs[0]["config"]["steps"], "protocol": pairs[0]["config"]["protocol"],
              "manifest_sha256": plan["manifest_sha256"],
              "experiments": {str(seed): {"path": str(pair["root"]),
                  "contract_sha256": pair["experiment_sha256"], "completion_sha256": pair["completions"]}
                  for seed, pair in pairs.items()},
              "scene_metrics": metrics,
              "limits": ["All values are prespecified final-step metrics on the SAME development subset.",
                         "Sample std and paired differences describe seed variability, not independent-scene uncertainty or statistical significance.",
                         "GT-camera control only; no automatic winner, schedule change, GRU or RGB-only/SOTA claim."]}
    path = output / "seed_summary.json"
    if path.exists():
        if controls.read_json(path) != report:
            raise ValueError("Existing seed summary differs; refusing overwrite")
    else:
        controls.write_json(path, report)
    print(f"FINAL_STEP={report['steps_each']}; seeds=0,1,2; same manifest and protocol", flush=True)
    print("split / scene / metric / C0 mean+/-sd / C1 mean+/-sd / paired C1-C0 mean+/-sd / paired values")
    for split, scenes in metrics.items():
        for scene, values in scenes.items():
            for metric, stats in values.items():
                groups = [stats[name] for name in ("C0", "C1", "paired_C1_minus_C0")]
                print(split, scene, metric, *[f"{row['mean']:.6f}+/-{row['sample_std']:.6f}" for row in groups],
                      json.dumps(groups[-1]["values"]))
    print("No GRU or additional training launched. Summary:", path, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--source-experiment", required=True)
    setup.add_argument("--output", required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--output", required=True)
    args = parser.parse_args()
    actions = {"prepare": lambda: prepare_repeats(args.source_experiment, args.output),
               "summarize": lambda: summarize_repeats(args.output)}
    actions[args.command]()


if __name__ == "__main__":
    main()