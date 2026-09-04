"""Package the verified checkpoint in LDA's loader-compatible layout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from lda_canary_contract import (
    BASE_MODEL_REVISION,
    BASE_SNAPSHOT,
    DINO_MODEL_REVISION,
    DINO_SNAPSHOT,
    EVALUATION_PATH,
    INPUT_MANIFEST_PATH,
    PUBLICATION_REPO_ID,
    PUBLICATION_VERSION,
    QWEN_MODEL_REVISION,
    QWEN_SNAPSHOT,
    RELEASE_CHECKPOINT,
    RELEASE_ROOT,
    ROOT,
    RUN_DIR,
    TRAINED_CHECKPOINT,
)

ASSET_ROOT = ROOT / ".treqs" / "assets"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def render_card(commit: str) -> str:
    card = (ASSET_ROOT / "robocasa-demo-canary-model-card.md").read_text()
    replacements = {
        "{{SOURCE_COMMIT}}": commit,
        "{{BASE_MODEL_REVISION}}": BASE_MODEL_REVISION,
        "{{QWEN_MODEL_REVISION}}": QWEN_MODEL_REVISION,
        "{{DINO_MODEL_REVISION}}": DINO_MODEL_REVISION,
        "{{PUBLICATION_VERSION}}": PUBLICATION_VERSION,
    }
    for marker, value in replacements.items():
        card = card.replace(marker, value)
    if "{{" in card or "}}" in card:
        raise RuntimeError("Unresolved model-card template marker")
    return card


def copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def main() -> None:
    if RELEASE_CHECKPOINT.parent.name != "checkpoints":
        raise RuntimeError("Release checkpoint must be nested under checkpoints/")
    evaluation = json.loads(EVALUATION_PATH.read_text())
    if evaluation.get("status") != "passed" or evaluation.get("global_step") != 1:
        raise RuntimeError("Refusing to package an unverified checkpoint")
    if not INPUT_MANIFEST_PATH.is_file() or not TRAINED_CHECKPOINT.is_file():
        raise RuntimeError("Missing manifest or trained checkpoint")

    if RELEASE_ROOT.exists():
        shutil.rmtree(RELEASE_ROOT)
    RELEASE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TRAINED_CHECKPOINT, RELEASE_CHECKPOINT)
    if sha256_file(RELEASE_CHECKPOINT) != evaluation["checkpoint"]["sha256"]:
        raise RuntimeError("Packaged checkpoint hash differs from evaluation")

    config = yaml.safe_load((BASE_SNAPSHOT / "config.yaml").read_text())
    config["framework"]["qwenvl"]["base_vlm"] = "Qwen/Qwen3-VL-4B-Instruct"
    config["framework"]["qwenvl"]["revision"] = QWEN_MODEL_REVISION
    config["framework"]["qwenvl"]["attn_implementation"] = "sdpa"
    config["framework"]["action_model"]["vision_encoder_path"] = "pretrained"
    config["framework"]["action_model"]["vision_encoder_load_weights"] = False
    config["framework"]["action_model"]["only_policy"] = True
    config["framework"]["action_model"]["policy_and_video_gen"] = False
    config["framework"]["action_model"]["only_wo_video_gen"] = False
    config["trainer"]["pretrained_checkpoint"] = None
    config["trainer"]["max_train_steps"] = 1
    config["trainer"]["freeze_modules"] = "action_model.vision_encoder,qwen_vl_interface"
    config["datasets"]["vla_data"]["data_root_dir"] = "playground/demo_data"
    config["datasets"]["vla_data"]["data_mix"] = "demo_data"
    config["datasets"]["vla_data"]["per_device_batch_size"] = 1
    config["datasets"]["vla_data"]["training_tasks"] = ["policy"]
    config["datasets"]["vla_data"]["training_task_weights"] = [1.0]
    (RELEASE_ROOT / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    stats = RUN_DIR / "dataset_statistics.json"
    if not stats.is_file():
        raise RuntimeError(f"Missing fine-tuning dataset statistics: {stats}")
    shutil.copyfile(stats, RELEASE_ROOT / "dataset_statistics.json")
    shutil.copyfile(EVALUATION_PATH, RELEASE_ROOT / "evaluation.json")
    shutil.copyfile(INPUT_MANIFEST_PATH, RELEASE_ROOT / "input-manifest.json")
    shutil.copyfile(ROOT / "LICENSE", RELEASE_ROOT / "LICENSE")
    shutil.copytree(
        DINO_SNAPSHOT,
        RELEASE_ROOT / "pretrained" / DINO_SNAPSHOT.name,
    )
    (RELEASE_ROOT / "README.md").write_text(render_card(source_commit()))
    (RELEASE_ROOT / "NOTICE").write_text(
        "LDA RoboCasa TReqs reproducibility canary\n\n"
        "Derived from Wayer2/LDA-robocasa and reproducible-ai/LDA-1B.\n"
        "The source repository states CC BY-NC 4.0 for the work and dataset.\n"
    )

    copy_if_present(BASE_SNAPSHOT / "README.md", RELEASE_ROOT / "upstream" / "LDA-robocasa" / "README.md")
    copy_if_present(QWEN_SNAPSHOT / "README.md", RELEASE_ROOT / "upstream" / "Qwen3-VL-4B-Instruct" / "README.md")

    publication = {
        "schema_version": 1,
        "repository": PUBLICATION_REPO_ID,
        "version": PUBLICATION_VERSION,
        "source_commit": source_commit(),
        "base_model_revision": BASE_MODEL_REVISION,
        "qwen_model_revision": QWEN_MODEL_REVISION,
        "dino_model_revision": DINO_MODEL_REVISION,
        "evaluation": evaluation,
    }
    (RELEASE_ROOT / "publication.json").write_text(
        json.dumps(publication, indent=2, sort_keys=True) + "\n"
    )
    print(f"Packaged release for hf://{PUBLICATION_REPO_ID}/{PUBLICATION_VERSION}")


if __name__ == "__main__":
    main()
