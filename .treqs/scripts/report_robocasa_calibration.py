"""Export compact timing evidence; all model weights remain on temporary compute."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from package_robocasa_calibration import measured_points, verify_checkpoint

from scripts.calibration_robocasa import encoded, load_inputs, sha256_file

ROOT = Path("artifacts/robocasa-calibration")


def main():
    plan_path = Path(".treqs/calibration/plan.json")
    config_path = plan_path.with_name("resolved-config.json")
    plan, _, _ = load_inputs(plan_path, config_path)
    points_root = ROOT / "points"
    final = plan["protocol"]["points"][-1]
    run = points_root / final["id"]
    checkpoint = run / "checkpoints" / f"steps_{final['steps']}_pytorch_model.pt"
    validation = verify_checkpoint(
        checkpoint, ROOT / "inputs/base/LDA-robocasa.pt", run / "summary.jsonl", final["steps"]
    )
    manifest_path = ROOT / "input-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    identity = {
        "configurationSha256": sha256_file(config_path),
        "seed": plan["protocol"]["seed"],
        "baseFiles": [
            {key: entry[key] for key in ("input", "sha256", "sizeBytes")}
            for entry in manifest["files"]
            if entry["input"] in {"base", "backbone"}
        ],
    }
    events = [json.loads(line) for line in (points_root / "processes.jsonl").read_text().splitlines()]
    points = measured_points(
        plan, events,
        initial_state_sha=hashlib.sha256(encoded(identity)).hexdigest(),
        final_sha=sha256_file(checkpoint),
    )
    report = {
        "schema": "reproai.cost-worker-report/v1",
        "candidateCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "planSha256": sha256_file(plan_path),
        "configSha256": sha256_file(config_path),
        "plan": plan,
        "points": points,
        "processEvents": events,
        "trainerEvents": {
            point["id"]: [
                json.loads(line)
                for line in (points_root / f"{point['id']}.json.jsonl").read_text().splitlines()
            ]
            for point in plan["protocol"]["points"]
        },
        "initialStateBasis": identity,
        "checkpointValidation": validation,
        "checkpointBytes": checkpoint.stat().st_size,
        "inputManifestSha256": sha256_file(manifest_path),
        "inputFileCount": len(manifest["files"]),
        "inputBytes": sum(entry["sizeBytes"] for entry in manifest["files"]),
        "inputPreparation": {key: manifest[key] for key in ("startedUnixSeconds", "finishedUnixSeconds")},
        "setupClock": json.loads((ROOT / "setup-clock.json").read_bytes()),
        "reportedUnixSeconds": time.time(),
        "scope": "Timing report with worker-side checkpoint validation; no model publication.",
    }
    payload = encoded(report)
    if len(payload) > 1024 * 1024:
        raise ValueError("cost report exceeds 1 MiB; refusing to truncate evidence")
    (ROOT / "cost-report.json").write_bytes(payload)
    print("COST_REPORT=" + json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    print("Cost report complete; checkpoints remain on the temporary worker.", flush=True)


if __name__ == "__main__":
    main()
