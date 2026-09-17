#!/usr/bin/env bash
# The workflow bounds this whole sequence to 6300 seconds, including setup.
set -euo pipefail
python3 -c 'import json,time; from pathlib import Path; p=Path("artifacts/robocasa-calibration"); p.mkdir(parents=True,exist_ok=True); (p/"setup-clock.json").write_text(json.dumps({"startedUnixSeconds":time.time()}))'
export PATH="$(python3 -m site --user-base)/bin:${PATH}"
nvidia-smi
test "$(nvidia-smi --list-gpus | wc -l | tr -d ' ')" = "8"
timeout --signal=TERM --kill-after=30 180 python3 -m pip install --user --disable-pip-version-check uv==0.12.3
timeout --signal=TERM --kill-after=30 180 uv venv --python 3.11 .venv
timeout --signal=TERM --kill-after=30 2400 uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt
timeout --signal=TERM --kill-after=30 180 uv pip install --python .venv/bin/python pytest==8.4.2 pyyaml==6.0.3
timeout --signal=TERM --kill-after=30 180 uv pip install --python .venv/bin/python --no-deps -e .
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" HF_HOME=/tmp/lda-calibration-hf HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
timeout --signal=TERM --kill-after=30 180 python -m pytest -q tests/treqs/test_calibration_timing.py tests/treqs/test_calibration_process.py tests/treqs/test_calibration_package.py tests/treqs/test_vlm_factory.py
timeout --signal=TERM --kill-after=30 3600 python .treqs/scripts/prepare_robocasa_calibration.py
timeout --signal=TERM --kill-after=30 7200 python -m scripts.calibration_robocasa run --plan .treqs/calibration/plan.json --config .treqs/calibration/resolved-config.json --output artifacts/robocasa-calibration/points
timeout --signal=TERM --kill-after=30 900 python .treqs/scripts/report_robocasa_calibration.py
