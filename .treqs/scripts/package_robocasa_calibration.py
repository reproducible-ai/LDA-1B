"""Package real calibration measurements and the last load-verified checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

from scripts.calibration_robocasa import encoded, load_inputs, recipe_digest, sha256_file

ROOT = Path("artifacts/robocasa-calibration")
RELEASE = ROOT / "release"
POINTS = ROOT / "points"


def measured_points(plan, events, *, initial_state_sha, final_sha):
    completed = [event for event in events if event["event"] == "point-complete"]
    expected = plan["protocol"]["points"]
    if any(event["event"] == "point-failed" for event in events):
        raise ValueError("failed points cannot be packaged as a successful calibration")
    if [event["pointId"] for event in completed] != [point["id"] for point in expected]:
        raise ValueError("missing, duplicate or out-of-order calibration point completion")
    result = []
    for point, event in zip(expected, completed, strict=True):
        timing = event["timing"]
        if timing["completedSteps"] != point["steps"] or timing["fullSteps"] != plan["recipe"]["fullSteps"]:
            raise ValueError("point used another update count or full schedule")
        start, finish = event["startedMonotonicSeconds"], timing["trainingFinishedMonotonicSeconds"]
        if (
            not math.isfinite(start)
            or not math.isfinite(finish)
            or not start < finish < event["processExitedMonotonicSeconds"]
        ):
            raise ValueError("invalid synchronized point interval")
        result.append(
            {
                "id": point["id"],
                "requestedSteps": point["steps"],
                "completedSteps": timing["completedSteps"],
                "outcome": "completed",
                "exitCode": 0,
                "recipeSha256": recipe_digest(plan["recipe"]),
                "hardware": plan["recipe"]["hardware"],
                "seed": plan["protocol"]["seed"],
                "cachePolicy": plan["protocol"]["cachePolicy"],
                "initialStateSha256": initial_state_sha,
                "startedMonotonicSeconds": start,
                "finishedMonotonicSeconds": finish,
                "steadyUpdates": timing["steadyUpdates"],
                "steadySeconds": timing["steadySeconds"],
                "checkpointSha256": final_sha if point["id"] == plan["protocol"]["finalPointId"] else None,
            }
        )
    return result


def verify_checkpoint(checkpoint, base_checkpoint, summary_path, steps):
    import torch

    from lda.model.checkpoint_loading import load_tensor_state_dict

    summaries = [json.loads(line) for line in Path(summary_path).read_text().splitlines()]
    if len(summaries) != 1 or summaries[0]["steps"] != steps or summaries[0]["optimizer_steps_completed"] != steps:
        raise ValueError("checkpoint does not prove the requested optimizer updates")
    trainable = set(summaries[0]["trainable_parameters"])
    if (
        not any(name.startswith("qwen_vl_interface.") for name in trainable)
        or not any(name.startswith("action_model.") for name in trainable)
        or any(name.startswith("action_model.vision_encoder.") for name in trainable)
    ):
        raise ValueError("trainable modules differ from the calibration recipe")
    base, trained = load_tensor_state_dict(base_checkpoint), load_tensor_state_dict(checkpoint)
    if not trained or base.keys() != trained.keys() or not trainable <= trained.keys():
        raise ValueError("checkpoint keys differ from the released model")
    changed = {"qwen": False, "action": False}
    for name, tensor in trained.items():
        reference = base[name]
        if tensor.shape != reference.shape or not torch.isfinite(tensor).all().item():
            raise ValueError("invalid checkpoint tensor: " + name)
        equal = torch.equal(tensor, reference.to(dtype=tensor.dtype))
        if name.startswith("action_model.vision_encoder.") and not equal:
            raise ValueError("frozen DINO weights changed")
        if name in trainable and not equal:
            changed["qwen" if name.startswith("qwen_vl_interface.") else "action"] = True
    if not all(changed.values()):
        raise ValueError("checkpoint lacks both Qwen and action-model updates")
    return {
        "status": "passed",
        "optimizerSteps": steps,
        "tensorCount": len(trained),
        "changedModules": changed,
        "loadVerified": True,
        "scope": "all tensor keys/shapes/finiteness and frozen DINO; no policy-quality claim",
    }


def main():
    started = time.time()
    plan_path = Path(".treqs/calibration/plan.json")
    config_path = Path(".treqs/calibration/resolved-config.json")
    plan, config, config_bytes = load_inputs(plan_path, config_path)
    if RELEASE.exists():
        raise RuntimeError("refusing to replace a packaged calibration")
    final = plan["protocol"]["points"][-1]
    run = POINTS / final["id"]
    checkpoint = run / "checkpoints" / f"steps_{final['steps']}_pytorch_model.pt"
    evaluation = verify_checkpoint(
        checkpoint, ROOT / "inputs/base/LDA-robocasa.pt", run / "summary.jsonl", final["steps"]
    )
    RELEASE.mkdir(parents=True)
    index = RELEASE / "LDA-robocasa-calibration.pt"
    shutil.copy2(checkpoint, index)
    final_sha = sha256_file(index)
    manifest = json.loads((ROOT / "input-manifest.json").read_bytes())
    identity = {
        "baseFiles": [
            {key: entry[key] for key in ("input", "sha256", "sizeBytes")}
            for entry in manifest["files"]
            if entry["input"] in {"base", "backbone"}
        ],
        "seed": plan["protocol"]["seed"],
        "optimizer": "fresh AdamW with ZeRO-2; no checkpoint resume",
        "configurationSha256": sha256_file(config_path),
    }
    initial_sha = hashlib.sha256(encoded(identity)).hexdigest()
    events = [json.loads(line) for line in (POINTS / "processes.jsonl").read_text().splitlines()]
    points = measured_points(plan, events, initial_state_sha=initial_sha, final_sha=final_sha)
    candidate = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    common = {"planSha256": sha256_file(plan_path), "candidateCommit": candidate}
    raw = {
        "schema": "reproai.calibration-timings/v1",
        **common,
        "points": points,
        "processEvents": events,
        "initialStateBasis": identity,
        "trainerEvents": {},
    }
    for point in plan["protocol"]["points"]:
        raw["trainerEvents"][point["id"]] = [
            json.loads(line) for line in (POINTS / f"{point['id']}.json.jsonl").read_text().splitlines()
        ]
    (RELEASE / "timings.json").write_bytes(encoded(raw))
    # Late upload and allocation shutdown cannot truthfully be known in this payload.
    # Preserve clock anchors for later host reconciliation instead of inventing zeros.
    components = {
        "coldSetupSeconds": None,
        "finalizationSeconds": None,
        "checkpointSeconds": sum(
            event["timing"]["checkpointSeconds"] for event in events if event["event"] == "point-complete"
        )
        / len(points),
        "evaluationSeconds": next(
            event["timing"]["evaluationSeconds"] for event in events if event["event"] == "point-complete"
        ),
        "otherCostUsd": None,
    }
    overhead = {
        "schema": "reproai.calibration-overhead/v1",
        **common,
        "components": components,
        "inputPreparation": {key: manifest[key] for key in ("startedUnixSeconds", "finishedUnixSeconds")},
        "packageStartedUnixSeconds": started,
        "setupClock": json.loads((ROOT / "setup-clock.json").read_bytes()),
    }
    (RELEASE / "overhead.json").write_bytes(encoded(overhead))
    final_checkpoint = {
        "pointId": final["id"],
        "sha256": final_sha,
        "optimizerSteps": final["steps"],
        "loadVerified": True,
    }
    measurements = {
        "schema": "reproai.calibration-measurements/v1",
        "planSha256": common["planSha256"],
        "rawLogSha256": sha256_file(RELEASE / "timings.json"),
        "coverage": {"fullRecipe": True, "representativeData": True, "fixedWork": False},
        "overhead": {**components, "evidenceSha256": sha256_file(RELEASE / "overhead.json")},
        "points": points,
        "finalCheckpoint": final_checkpoint,
    }
    (RELEASE / "calibration.json").write_bytes(encoded(measurements))
    (RELEASE / "resolved-config.json").write_bytes(config_bytes)
    (RELEASE / "trainer-state.json").write_bytes(
        encoded(
            {
                "schema": "reproai.calibration-trainer-state/v1",
                **common,
                "finalCheckpoint": final_checkpoint,
                "initialStateSha256": initial_sha,
                "trainerStateSha256": sha256_file(run / "summary.jsonl"),
                "initialStateBasis": (
                    "Pinned model input bytes and reset configuration; source and process journal "
                    "establish fresh loading and optimizer construction."
                ),
            }
        )
    )
    shutil.copy2(ROOT / "input-manifest.json", RELEASE / "input-manifest.json")
    (RELEASE / "evaluation.json").write_bytes(encoded(evaluation))
    shutil.copy2(run / "summary.jsonl", RELEASE / "summary.jsonl")
    shutil.copy2(plan_path, RELEASE / "calibration-plan.json")
    shutil.copy2("LICENSE", RELEASE / "CC-BY-NC-4.0.md")
    (RELEASE / "LICENSE").write_text(
        "Component license index\n\n"
        "LDA source: CC BY-NC 4.0; see CC-BY-NC-4.0.md.\n"
        "Released LDA and Qwen model metadata: Apache 2.0; see APACHE-2.0.txt.\n"
        "Embedded DINOv3 materials: DINOv3 License; see DINOv3-LICENSE.md.\n"
        "No single license supersedes all component terms. Built with DINOv3.\n"
    )
    shutil.copy2(".treqs/assets/APACHE-2.0.txt", RELEASE / "APACHE-2.0.txt")
    shutil.copy2(ROOT / "inputs/pretrained/dinov3-vits16-pretrain-lvd1689m/LICENSE.md", RELEASE / "DINOv3-LICENSE.md")
    (RELEASE / "NOTICE").write_text(
        "Derived from LDA RoboCasa, Qwen3-VL and DINOv3. Component-specific licenses apply.\n"
    )
    (RELEASE / "README.md").write_text(
        "# LDA-1B RoboCasa calibration checkpoint\n\n"
        f"Private {final['steps']}-update calibration with the full 300000-update scheduler. "
        "Continuation from released post-trained weights on all 24 GR1 tasks, SDPA attention, eight A100 GPUs. "
        "No original-pretraining, certification or held-out policy-quality claim. "
        "Checkpoint contains model weights only. Use the pinned source and resolved configuration "
        "with the pinned input preparation script to restore processor and dataset resources.\n"
        f"Source: {candidate}\n"
    )
    files = [
        {
            "path": str(path.relative_to(RELEASE)),
            "sha256": sha256_file(path),
            "sizeBytes": path.stat().st_size,
        }
        for path in sorted(RELEASE.rglob("*"))
        if path.is_file()
    ]
    artifact_manifest = {
        "schema": "reproai.artifact-manifest/v1",
        "format": "lda-robocasa-torch-state-dict",
        "loadVerified": True,
        "files": files,
    }
    (RELEASE / "artifact-manifest.json").write_bytes(encoded(artifact_manifest))
    artifact = {
        **artifact_manifest,
        "schema": "reproai.artifact/v1",
        "path": str(index),
        "sha256": final_sha,
        "sizeBytes": index.stat().st_size,
        "files": [{**entry, "path": str(RELEASE / entry["path"])} for entry in files],
    }
    result = {
        "schema": "reproai.result/v1",
        "checkpoint": str(index),
        "artifactSha256": final_sha,
        "artifactSizeBytes": index.stat().st_size,
        "loadVerified": True,
        "optimizerSteps": final["steps"],
        "taskMetric": {"metric": "finiteLoss", "minimum": 1},
        "finiteLoss": 1,
        "finalLoss": events[-1]["timing"]["finalLoss"],
    }
    (RELEASE / "result.json").write_bytes(encoded(result))
    print("E2E_ARTIFACT=" + json.dumps(artifact, sort_keys=True, allow_nan=False), flush=True)
    print("E2E_RESULT=" + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
