#!/usr/bin/env bash
# Supervisor/CI only; may download packages. Cache-managed host checks: no GPU requirements or candidate-source mutations.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
uv run --no-project --python 3.12.12 \
  --with-requirements .treqs/calibration/local-check-requirements.txt \
  python -m pytest -q tests/treqs
