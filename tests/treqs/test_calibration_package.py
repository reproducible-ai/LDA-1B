from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import torch

from scripts.calibration_robocasa import load_inputs, sha256_file

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "calibration_package", SOURCE / ".treqs/scripts/package_robocasa_calibration.py"
)
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    shutil.copytree(SOURCE / ".treqs/calibration", tmp_path / ".treqs/calibration")
    monkeypatch.chdir(tmp_path)
    root = package.ROOT
    (root / "inputs/base").mkdir(parents=True)
    (root / "setup-clock.json").write_text('{"startedUnixSeconds":1}')
    base = {
        "qwen_vl_interface.weight": torch.ones(2),
        "action_model.weight": torch.ones(2),
        "action_model.vision_encoder.weight": torch.ones(2),
    }
    torch.save(base, root / "inputs/base/LDA-robocasa.pt")
    (root / "input-manifest.json").write_text(
        json.dumps(
            {
                "startedUnixSeconds": 2,
                "finishedUnixSeconds": 3,
                "files": [
                    {
                        "input": "base",
                        "path": str(root / "inputs/base/LDA-robocasa.pt"),
                        "sizeBytes": 10,
                        "sha256": "a" * 64,
                    }
                ],
            }
        )
    )
    for filename in (
        "LICENSE",
        ".treqs/assets/APACHE-2.0.txt",
        str(root / "inputs/pretrained/dinov3-vits16-pretrain-lvd1689m/LICENSE.md"),
    ):
        p = Path(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic license")
    run = package.POINTS / "p3"
    (run / "checkpoints").mkdir(parents=True)
    trained = {k: (v + 0.1 if "vision_encoder" not in k else v) for k, v in base.items()}
    checkpoint = run / "checkpoints/steps_400_pytorch_model.pt"
    torch.save(trained, checkpoint)
    (run / "summary.jsonl").write_text(
        json.dumps(
            {
                "steps": 400,
                "optimizer_steps_completed": 400,
                "trainable_parameters": ["qwen_vl_interface.weight", "action_model.weight"],
            }
        )
        + "\n"
    )
    plan = json.loads(Path(".treqs/calibration/plan.json").read_text())
    events = []
    for i, p in enumerate(plan["protocol"]["points"]):
        events.append(
            {
                "event": "point-complete",
                "pointId": p["id"],
                "startedUnixSeconds": i * 1000 + 10,
                "startedMonotonicSeconds": i * 1000 + 10,
                "processExitedMonotonicSeconds": i * 1000 + p["steps"] + 100,
                "finishedUnixSeconds": i * 1000 + p["steps"] + 100,
                "timing": {
                    "completedSteps": p["steps"],
                    "fullSteps": 300000,
                    "trainingFinishedMonotonicSeconds": i * 1000 + p["steps"] + 20,
                    "steadyUpdates": p["steps"] - 20,
                    "steadySeconds": p["steps"] - 20,
                    "checkpointSeconds": 20,
                    "evaluationSeconds": 5 if i == 0 else None,
                    "finalLoss": 1.2,
                },
            }
        )
        (package.POINTS / f"{p['id']}.json.jsonl").write_text("{}\n")
    (package.POINTS / "processes.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    monkeypatch.setattr(package.subprocess, "check_output", lambda *a, **k: "f" * 40)
    return checkpoint, root / "inputs/base/LDA-robocasa.pt", run / "summary.jsonl"


def test_real_tensor_readback_and_packaging(prepared, capsys):
    package.main()
    output = capsys.readouterr().out
    assert output.count("E2E_RESULT=") == output.count("E2E_ARTIFACT=") == 1
    measurements = json.loads((package.RELEASE / "calibration.json").read_text())
    assert measurements["coverage"]["fixedWork"] is False
    assert measurements["overhead"]["coldSetupSeconds"] is None
    assert measurements["overhead"]["evaluationSeconds"] == 5
    assert measurements["overhead"]["checkpointSeconds"] == 20
    assert measurements["finalCheckpoint"]["optimizerSteps"] == 400
    manifest = json.loads((package.RELEASE / "artifact-manifest.json").read_text())
    for entry in manifest["files"]:
        assert sha256_file(package.RELEASE / entry["path"]) == entry["sha256"]


@pytest.mark.parametrize("failure", ["frozen", "missing", "nan", "unchanged", "step", "trainable"])
def test_rejects_invalid_checkpoint(prepared, failure):
    checkpoint, base, summary = prepared
    weights = torch.load(checkpoint, weights_only=True)
    if failure == "frozen":
        weights["action_model.vision_encoder.weight"] += 1
    elif failure == "missing":
        weights.pop("qwen_vl_interface.weight")
    elif failure == "nan":
        weights["action_model.weight"][0] = float("nan")
    elif failure == "unchanged":
        weights = torch.load(base, weights_only=True)
    elif failure in ("step", "trainable"):
        record = json.loads(summary.read_text())
        if failure == "step":
            record["optimizer_steps_completed"] = 1
        else:
            record["trainable_parameters"].append("action_model.vision_encoder.weight")
        summary.write_text(json.dumps(record))
    torch.save(weights, checkpoint)
    with pytest.raises(ValueError):
        package.verify_checkpoint(checkpoint, base, summary, 400)


def test_configuration_and_launcher_tampering_are_rejected(prepared):
    plan = Path(".treqs/calibration/plan.json")
    config = plan.with_name("resolved-config.json")
    load_inputs(plan, config)
    launcher = plan.with_name("deepspeed.json")
    launcher.write_text(launcher.read_text() + " ")
    with pytest.raises(ValueError, match="launcher"):
        load_inputs(plan, config)


def test_cost_report_retains_measurements_without_copying_weights(prepared, capsys, monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(SOURCE / ".treqs/scripts"))
    import report_robocasa_calibration as report

    monkeypatch.setattr(report.subprocess, "check_output", lambda *a, **k: "f" * 40)
    report.main()
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.startswith("COST_REPORT=")]
    assert len(lines) == 1
    result = json.loads(lines[0].split("=", 1)[1])
    assert [p["completedSteps"] for p in result["points"]] == [100, 200, 400]
    assert result["checkpointValidation"]["status"] == "passed"
    assert len(result["trainerEvents"]) == 3
    assert result["checkpointBytes"] == prepared[0].stat().st_size
    assert not package.RELEASE.exists()
    assert "E2E_ARTIFACT=" not in output
    assert (report.ROOT / "cost-report.json").stat().st_size < 1024 * 1024
    sys.modules.pop("report_robocasa_calibration", None)


def test_points_must_be_complete_and_ordered(prepared):
    plan = json.loads(Path(".treqs/calibration/plan.json").read_text())
    events = [json.loads(line) for line in (package.POINTS / "processes.jsonl").read_text().splitlines()]
    for invalid in (events[:-1], events[::-1], [*events, {"event": "point-failed"}]):
        with pytest.raises(ValueError):
            package.measured_points(plan, invalid, initial_state_sha="a" * 64, final_sha="b" * 64)


def test_storage_allowance_requires_evidence_and_excludes_instance_disks(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "calibration_prepare", SOURCE / ".treqs/scripts/prepare_robocasa_calibration.py"
    )
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    with pytest.raises(RuntimeError):
        prepare.storage_inventory(tmp_path)
    for name, model, size in [
        ("nvme0n1", "Amazon Elastic Block Store", 200 * 1024**3),
        ("nvme1n1", "Amazon EC2 NVMe Instance Storage", 8 * 1024**4),
    ]:
        device = tmp_path / name
        (device / "device").mkdir(parents=True)
        (device / "device/model").write_text(model)
        (device / "size").write_text(str(size // 512))
    assert len(prepare.storage_inventory(tmp_path)) == 1
    (tmp_path / "nvme0n1/size").write_text(str(2 * 1024**4 // 512))
    with pytest.raises(RuntimeError):
        prepare.storage_inventory(tmp_path)
