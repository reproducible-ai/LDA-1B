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


def test_runner_launches_one_exact_one_gpu_one_step_command(monkeypatch, tmp_path):
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
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        runner.torch.cuda,
        "get_device_properties",
        lambda _index: type("Properties", (), {"total_memory": 96 * 1024**3, "name": "RTX PRO 6000 Blackwell"})(),
    )
    monkeypatch.setattr(runner.torch, "__version__", "2.9.0+cu128")
    monkeypatch.setattr(runner.torch.version, "cuda", "12.8")
    monkeypatch.setattr(runner.torch.cuda, "get_device_capability", lambda _: (12, 0))
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    runner.main()

    assert len(calls) == 1
    command = calls[0][0][0]
    kwargs = calls[0][1]
    assert command[command.index("--num_processes") + 1] == "1"
    assert command[command.index("--trainer.max_train_steps") + 1] == "1"
    assert command[command.index("--trainer.save_interval") + 1] == "1"
    assert kwargs["cwd"] == tmp_path
    assert kwargs["check"] is True
    assert kwargs["env"]["WANDB_MODE"] == "disabled"


def configure_packager(monkeypatch, tmp_path, *, include_dino_license: bool):
    packager = load_script("package_robocasa_canary", monkeypatch)
    root = tmp_path / "source"
    base_snapshot = tmp_path / "base"
    qwen_snapshot = tmp_path / "qwen"
    dino_snapshot = tmp_path / "dino"
    run_dir = tmp_path / "run"
    release_root = tmp_path / "release"
    for path in (root, base_snapshot, qwen_snapshot, dino_snapshot, run_dir):
        path.mkdir(parents=True)

    source_license = b"CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/\n"
    (root / "LICENSE").write_bytes(source_license)
    asset_root = root / ".treqs" / "assets"
    asset_root.mkdir(parents=True)
    apache_license = b"Apache License 2.0 fixture\n"
    (asset_root / "APACHE-2.0.txt").write_bytes(apache_license)
    (asset_root / "robocasa-demo-canary-model-card.md").write_bytes(
        (ROOT / ".treqs" / "assets" / "robocasa-demo-canary-model-card.md").read_bytes()
    )
    (base_snapshot / "config.yaml").write_text(
        "framework:\n  qwenvl: {}\n  action_model: {}\ntrainer: {}\ndatasets:\n  vla_data: {}\n"
    )
    (base_snapshot / "README.md").write_text("LDA base model card\n")
    (qwen_snapshot / "README.md").write_text("Qwen model card\n")
    (dino_snapshot / "config.json").write_text("{}\n")
    (dino_snapshot / ".cache" / "huggingface").mkdir(parents=True)
    (dino_snapshot / ".cache" / "huggingface" / "download.metadata").write_text("cache metadata\n")
    dino_license = b"pinned DINOv3 license fixture\n"
    if include_dino_license:
        (dino_snapshot / "LICENSE.md").write_bytes(dino_license)

    trained_checkpoint = run_dir / "checkpoint.pt"
    trained_checkpoint.write_bytes(b"trained-checkpoint")
    (run_dir / "dataset_statistics.json").write_text("{}\n")
    evaluation_path = run_dir / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "global_step": 1,
                "optimizer_steps_completed": 1,
                "loadVerified": True,
                "checkpoint": {"sha256": sha256_file(trained_checkpoint)},
            }
        )
    )
    input_manifest = tmp_path / "input-manifest.json"
    input_manifest.write_text("{}\n")

    values = {
        "ROOT": root,
        "ASSET_ROOT": asset_root,
        "BASE_SNAPSHOT": base_snapshot,
        "QWEN_SNAPSHOT": qwen_snapshot,
        "DINO_SNAPSHOT": dino_snapshot,
        "RUN_DIR": run_dir,
        "TRAINED_CHECKPOINT": trained_checkpoint,
        "EVALUATION_PATH": evaluation_path,
        "INPUT_MANIFEST_PATH": input_manifest,
        "RELEASE_ROOT": release_root,
        "RELEASE_CHECKPOINT": release_root / "checkpoints" / "canary.pt",
        "DINO_LICENSE_SHA256": hashlib.sha256(dino_license).hexdigest(),
        "SOURCE_LICENSE_SHA256": hashlib.sha256(source_license).hexdigest(),
        "APACHE_LICENSE_SHA256": hashlib.sha256(apache_license).hexdigest(),
    }
    for name, value in values.items():
        monkeypatch.setattr(packager, name, value, raising=False)
    monkeypatch.setattr(packager, "source_commit", lambda: "a" * 40)
    return packager, release_root


def test_packager_fails_closed_without_the_pinned_dinov3_license(monkeypatch, tmp_path):
    packager, release_root = configure_packager(
        monkeypatch,
        tmp_path,
        include_dino_license=False,
    )

    with pytest.raises(RuntimeError, match="Missing DINOv3 license"):
        packager.main()

    assert not release_root.exists()


@pytest.mark.parametrize(
    ("license_name", "error"),
    (
        ("source", "LDA source license hash mismatch"),
        ("apache", "Apache 2.0 license hash mismatch"),
    ),
)
def test_packager_fails_closed_for_a_tampered_static_license(
    monkeypatch,
    tmp_path,
    license_name,
    error,
):
    packager, release_root = configure_packager(
        monkeypatch,
        tmp_path,
        include_dino_license=True,
    )
    license_path = packager.ROOT / "LICENSE" if license_name == "source" else packager.ASSET_ROOT / "APACHE-2.0.txt"
    license_path.write_text("tampered license\n")

    with pytest.raises(RuntimeError, match=error):
        packager.main()

    assert not release_root.exists()


def test_packager_bundles_and_discloses_component_licenses(monkeypatch, tmp_path, capsys):
    packager, release_root = configure_packager(
        monkeypatch,
        tmp_path,
        include_dino_license=True,
    )

    packager.main()

    output = capsys.readouterr().out
    receipts = {line.split("=", 1)[0]: json.loads(line.split("=", 1)[1]) for line in output.splitlines() if line.startswith("E2E_")}
    assert receipts["E2E_RESULT"] == json.loads((release_root / "checkpoints/result.json").read_text())
    assert receipts["E2E_ARTIFACT"]["loadVerified"] is True
    assert (release_root / "LICENSE").is_file()
    assert (release_root / "CC-BY-NC-4.0.md").is_file()
    assert (release_root / "APACHE-2.0.txt").is_file()
    assert (release_root / "DINOv3-LICENSE.md").read_bytes() == (
        release_root / "pretrained" / "dino" / "LICENSE.md"
    ).read_bytes()
    assert not (release_root / "pretrained" / "dino" / ".cache").exists()
    licenses = {
        record["id"]: record for record in json.loads((release_root / "publication.json").read_text())["licenses"]
    }
    assert licenses["CC-BY-NC-4.0"]["sha256"] == packager.SOURCE_LICENSE_SHA256
    assert licenses["Apache-2.0"]["sha256"] == packager.APACHE_LICENSE_SHA256
    assert licenses["LicenseRef-DINOv3"]["sha256"] == packager.DINO_LICENSE_SHA256
    card = (release_root / "README.md").read_text()
    assert "Built with DINOv3" in card
    assert "component-specific" in card.lower()
    publication = json.loads((release_root / "publication.json").read_text())
    assert {license_record["id"] for license_record in publication["licenses"]} == {
        "CC-BY-NC-4.0",
        "Apache-2.0",
        "LicenseRef-DINOv3",
    }

    result = json.loads((release_root / "checkpoints/result.json").read_text())
    assert result["optimizerSteps"] == 1
    manifest = json.loads((release_root / "checkpoints/artifact-manifest.json").read_text())
    directory = release_root / "checkpoints"
    expected = {str(p.relative_to(directory)) for p in directory.rglob("*")
                if p.is_file() and p.name not in {"artifact-manifest.json", "result.json"}}
    assert {record["path"] for record in manifest["files"]} == expected
    for record in manifest["files"]:
        path = directory / record["path"]
        assert record["sha256"] == sha256_file(path)
        assert record["sizeBytes"] == path.stat().st_size
