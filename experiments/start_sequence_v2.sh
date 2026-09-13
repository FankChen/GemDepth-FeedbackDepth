#!/usr/bin/env bash
# Invoked as: git show FIXED_COMMIT:experiments/start_sequence_v2.sh | bash -s -- FIXED_COMMIT
set -euo pipefail
REV=${1:?Pass the fixed release commit}
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected full pinned commit'; exit 2; }
RUN_ROOT=${RUN_ROOT:-/mnt/data/PROJECT_CHEN/code/PP-DPT}
cd "$RUN_ROOT"
command -v setsid >/dev/null
command -v flock >/dev/null
[[ -x /usr/local/bin/python && -f stereogru_sequence_wcIm3t5e/run/experiment.json && -f sequence_audit_XR5svFMg/report/audit.json ]]
# Freeze full source snapshot without checkout/pull/reset or checkpoint changes.
D=$(mktemp -d /tmp/sequence-v2.XXXXXXXX)
git archive "$REV" model loss dataset config scripts diagnostics experiments | tar -x -C "$D"
L=$(mktemp "$RUN_ROOT/sequence_v2_XXXXXXXX.log")
RUN_ROOT="$RUN_ROOT" GPU=${GPU:-5} setsid nohup bash "$D/experiments/run_sequence_v2.sh" >"$L" 2>&1 </dev/null &
echo "LAUNCHER_PID=$! (not proof of training); LOG=$L"
echo 'WILL reuse B0 seed0; train B0 seeds1/2 then gated F1/S1. TRAIN step=... confirms real updates.'
echo 'Ctrl+C stops log viewing only. Duplicate launches are locked; failures stop all subsequent phases.'
tail -n 40 -f "$L"