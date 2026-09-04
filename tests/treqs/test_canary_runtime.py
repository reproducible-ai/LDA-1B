from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".treqs" / "scripts"


def load_script(name: str, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"treqs_test_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_records(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def test_verifier_accepts_a_tiny_one_step_checkpoint_and_records_the_change(
    monkeypatch,
    tmp_path,
):
    verifier = load_script("verify_robocasa_canary", monkeypatch)
    artifact_root = tmp_path / "artifact"
    base_snapshot = artifact_root / "base"
    qwen_snapshot = artifact_root / "qwen"
    dino_snapshot = artifact_root / "dino"
    dataset_path = artifact_root / "dataset"
    for root, filename in (
        (base_snapshot, "config.yaml"),
        (qwen_snapshot, "config.json"),
        (dino_snapshot, "config.json"),
        (dataset_path, "episode.hdf5"),
    ):
        root.mkdir(parents=True)
        (root / filename).write_text(root.name)

    base_checkpoint = base_snapshot / "checkpoints" / "base.pt"
    base_checkpoint.parent.mkdir()
    torch.save(
        {
            "action_model.weight": torch.tensor([1.0]),
            "frozen.weight": torch.tensor([2.0]),
        },
        base_checkpoint,
    )
    run_dir = artifact_root / "run"
    trained_checkpoint = run_dir / "checkpoints" / "steps_1_pytorch_model.pt"
    trained_checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "action_model.weight": torch.tensor([1.25]),
            "frozen.weight": torch.tensor([2.0]),
        },
        trained_checkpoint,
    )
    (run_dir / "summary.jsonl").write_text(
        json.dumps(
            {
                "steps": 1,
                "optimizer_steps_completed": 1,
                "trainable_parameters": ["action_model.weight"],
            }
        )
        + "\n"
    )
    manifest_path = artifact_root / "input-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": {
                    "base_model": manifest_records(base_snapshot),
                    "qwen_model": manifest_records(qwen_snapshot),
                    "dino_model": manifest_records(dino_snapshot),
                    "dataset": manifest_records(dataset_path),
                }
            }
        )
    )
    evaluation_path = run_dir / "evaluation.json"
    for name, value in {
        "INPUT_MANIFEST_PATH": manifest_path,
        "BASE_CHECKPOINT": base_checkpoint,
        "TRAINED_CHECKPOINT": trained_checkpoint,
        "BASE_SNAPSHOT": base_snapshot,
        "QWEN_SNAPSHOT": qwen_snapshot,
        "DINO_SNAPSHOT": dino_snapshot,
        "DATASET_PATH": dataset_path,
        "RUN_DIR": run_dir,
        "EVALUATION_PATH": evaluation_path,
    }.items():
        monkeypatch.setattr(verifier, name, value)

    verifier.main()

    result = json.loads(evaluation_path.read_text())
    assert result["status"] == "passed"
    assert result["global_step"] == 1
    assert result["optimizer_steps_completed"] == 1
    assert result["checkpoint"]["changed_tensor"] == "action_model.weight"
    assert result["base_checkpoint"]["sha256"] != result["checkpoint"]["sha256"]


def test_hf_preflight_rejects_a_public_destination_before_preupload(monkeypatch):
    preflight = load_script("check_hf_access", monkeypatch)
    monkeypatch.setattr(preflight, "request_json", lambda *_args: {"private": False})

    def unexpected_preupload(*_args, **_kwargs):
        raise AssertionError("public destinations must fail before the preupload request")

    monkeypatch.setattr(preflight, "urlopen", unexpected_preupload)

    with pytest.raises(RuntimeError, match="must be private"):
        preflight.check_private_writable_repo(
            "reproducible-ai/harness-test-lda-robocasa-issue-5",
            "not-a-real-token",
        )


def test_runner_launches_one_exact_four_gpu_one_step_command(monkeypatch, tmp_path):
    runner = load_script("run_robocasa_canary", monkeypatch)
    base_snapshot = tmp_path / "base"
    qwen_snapshot = tmp_path / "qwen"
    pretrained_root = tmp_path / "pretrained"
    base_checkpoint = base_snapshot / "checkpoints" / "base.pt"
    for path in (
        base_snapshot / "config.yaml",
        qwen_snapshot / "config.json",
        pretrained_root / "dinov3-vits16-pretrain-lvd1689m" / "config.json",
        base_checkpoint,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "BASE_SNAPSHOT", base_snapshot)
    monkeypatch.setattr(runner, "QWEN_SNAPSHOT", qwen_snapshot)
    monkeypatch.setattr(runner, "PRETRAINED_ROOT", pretrained_root)
    monkeypatch.setattr(runner, "BASE_CHECKPOINT", base_checkpoint)
    monkeypatch.setattr(runner, "RUN_ROOT", tmp_path / "run")
    monkeypatch.setattr(runner, "RUN_ID", "test-run")
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        runner.torch.cuda,
        "get_device_properties",
        lambda _index: type("Properties", (), {"total_memory": 48 * 1024**3})(),
    )
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    runner.main()

    assert len(calls) == 1
    command = calls[0][0][0]
    kwargs = calls[0][1]
    assert command[command.index("--num_processes") + 1] == "4"
    assert command[command.index("--trainer.max_train_steps") + 1] == "1"
    assert command[command.index("--trainer.save_interval") + 1] == "1"
    assert kwargs["cwd"] == tmp_path
    assert kwargs["check"] is True
    assert kwargs["env"]["WANDB_MODE"] == "disabled"
