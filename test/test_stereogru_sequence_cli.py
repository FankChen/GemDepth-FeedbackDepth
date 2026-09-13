"""Fresh-process control-channel and shell dispatch tests; never launch a GPU job."""

import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import stereogru_sequence_experiment as experiment


def test_fresh_next_process_routes_missing_xformers_notices_to_stderr():
    code = r'''
import importlib.abc
import sys
class NoXformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "xformers" or fullname.startswith("xformers."):
            raise ModuleNotFoundError("deliberately absent xformers")
sys.meta_path.insert(0, NoXformers())
sys.path.insert(0, "scripts")
import stereogru_sequence_experiment as experiment
calls = []
context = {"config": {}, "pairs": {}}
def verify(output):
    calls.append(output)
    experiment.get_decoder_class("DPTHeadCalibratedGRUSequenceConvNeXt")
    print("verification diagnostic")
    return context
def action(output, *, _context):
    assert _context is context
    assert len(calls) == 1
    print("discovery diagnostic")
    return "train B0 0"
experiment.verify = verify
experiment.production_guard = lambda *args: None
experiment.next_action = action
sys.argv = ["sequence", "next", "--output", "synthetic-no-GPU"]
experiment.main()
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True,
                            capture_output=True, timeout=60, check=True)
    assert result.stdout == "SEQUENCE_NEXT train B0 0\n"
    assert "xFormers not available" in result.stderr
    assert "verification diagnostic" in result.stderr and "discovery diagnostic" in result.stderr


def test_production_gate_not_redefined_by_shortened_default(tmp_path, monkeypatch):
    cfg = experiment.method_config(experiment.DEFAULT_CONFIG)
    reference = OmegaConf.to_container(OmegaConf.load(
        ROOT / "config/stereogru/matched_c0_c1_30train.yaml"), resolve=True)
    pairs = {seed: {"config": {**reference, "seed": seed}} for seed in experiment.SEEDS}
    experiment.production_guard(cfg, pairs)
    cfg["gate"]["steps"] = 1
    path = tmp_path / "short-default.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    monkeypatch.setattr(experiment, "DEFAULT_CONFIG", path)
    assert experiment.method_config(path) == cfg  # Short fixtures remain allowed only in API tests.
    with pytest.raises(ValueError, match="fixed 200-step"):
        experiment.production_guard(cfg, pairs)


def test_source_inventory_covers_discovery_dependencies_and_new_plugins(tmp_path, monkeypatch):
    names = set(experiment.source_inventory())
    assert {"model/motion_module/motion_module.py", "model/decoder_registry.py",
            "loss/objective_registry.py", "config/stereogru/matched_c0_c1_30train.yaml"} <= names
    assert "train.py" not in names
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    plugin = tmp_path / "model/dpt_new_plugin.py"
    plugin.parent.mkdir()
    before = experiment.source_inventory()
    plugin.write_text("# New auto-discovered code must invalidate prepared evidence.\n")
    assert set(experiment.source_inventory()) - set(before) == {"model/dpt_new_plugin.py"}
    plugin.unlink()
    assert experiment.source_inventory() == before


@pytest.mark.parametrize("reply,code,action", [
    ("SEQUENCE_NEXT train B0 0", 17, "train"),
    ("SEQUENCE_NEXT gate S1 2", 17, "gate"),
    ("SEQUENCE_NEXT summarize - -", 0, "summarize"),
    ("xFormers not available\nSEQUENCE_NEXT train B0 0", 2, None),
    ("SEQUENCE_NEXT train B0 0\nextra", 2, None),
    ("SEQUENCE_NEXT train B0 9", 2, None),
    ("SEQUENCE_NEXT train BAD 0", 2, None),
])
def test_launcher_strict_protocol_and_child_failure(tmp_path, reply, code, action):
    executable = tmp_path / "fake-python"
    executable.write_text('''#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == '-u' ]]; then shift; fi
case "$2" in
  next) printf '%s\n' "$NEXT_REPLY" ;;
  train|gate) printf '%s\n' "$2" >> "$ACTION_LOG"; exit 17 ;;
  summarize) printf '%s\n' "$2" >> "$ACTION_LOG" ;;
  *) exit 99 ;;
esac
''')
    executable.chmod(0o700)
    smi = tmp_path / "nvidia-smi"
    smi.write_text('''#!/usr/bin/env bash
case "$*" in
  *query-gpu=uuid*) echo GPU-synthetic ;;
  *query-gpu=memory.used*) echo 0 ;;
  *query-compute-apps*) : ;;
  *) exit 98 ;;
esac
''')
    smi.chmod(0o700)
    root = tmp_path / "run"
    root.mkdir()
    (root / "experiment.json").write_text("{}")
    log = tmp_path / "actions"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "PYTHON": str(executable),
           "SEQUENCE_RUN": str(root), "NEXT_REPLY": reply, "ACTION_LOG": str(log)}
    result = subprocess.run(["bash", str(ROOT / "scripts/run_stereogru_sequence_experiment.sh")],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == code, result.stdout + result.stderr
    assert (log.read_text().splitlines() if log.exists() else []) == ([] if action is None else [action])