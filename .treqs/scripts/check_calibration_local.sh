#!/usr/bin/env bash
# Restricted operator checks: standard library only, no package downloads.
# Full supervisor/CI CPU suite: bash .treqs/scripts/check_calibration_cpu.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p artifacts/operator-checks/tmp
export TMPDIR="$PWD/artifacts/operator-checks/tmp"
python3 -S -c 'from scripts.calibration_robocasa import load_inputs; load_inputs(".treqs/calibration/plan.json", ".treqs/calibration/resolved-config.json"); print("PASS: actual runtime recipe and pinned input/launcher checks")'
python3 -S -m unittest tests.treqs.test_calibration_process
