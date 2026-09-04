"""Package the verified checkpoint in LDA's loader-compatible layout."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml
from lda_canary_contract import (
    APACHE_LICENSE_SHA256,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BASE_SNAPSHOT,
    DINO_LICENSE_NAME,
    DINO_LICENSE_SHA256,
    DINO_MODEL_ID,
    DINO_MODEL_REVISION,
    DINO_SNAPSHOT,
    EVALUATION_PATH,
    INPUT_MANIFEST_PATH,
    PUBLICATION_REPO_ID,
    PUBLICATION_VERSION,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
    QWEN_SNAPSHOT,
    RELEASE_CHECKPOINT,
    RELEASE_ROOT,
    ROOT,
    RUN_DIR,
    SOURCE_LICENSE_SHA256,
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


def require_file(path: Path, description: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing {description}: {path}")
    return path


def license_index() -> str:
    return (
        "# Component license index\n\n"
        "No single license supersedes the terms for every component in this assembled checkpoint.\n\n"
        "- LDA source and the bundled demo dataset: CC BY-NC 4.0; see `CC-BY-NC-4.0.md`.\n"
        "- Wayer2/LDA-robocasa and Qwen/Qwen3-VL-4B-Instruct upstream model metadata: "
        "Apache 2.0; see `APACHE-2.0.txt`.\n"
        "- Embedded DINOv3 materials: DINOv3 License; see `DINOv3-LICENSE.md`.\n\n"
        "Use and redistribution must comply with all applicable component-specific terms.\n\n"
        "Built with DINOv3.\n"
    )


def main() -> None:
    if RELEASE_CHECKPOINT.parent.name != "checkpoints":
        raise RuntimeError("Release checkpoint must be nested under checkpoints/")
    evaluation = json.loads(EVALUATION_PATH.read_text())
    if evaluation.get("status") != "passed" or evaluation.get("global_step") != 1:
        raise RuntimeError("Refusing to package an unverified checkpoint")
    if not INPUT_MANIFEST_PATH.is_file() or not TRAINED_CHECKPOINT.is_file():
        raise RuntimeError("Missing manifest or trained checkpoint")

    source_license = require_file(ROOT / "LICENSE", "LDA source license notice")
    apache_license = require_file(ASSET_ROOT / "APACHE-2.0.txt", "Apache 2.0 license")
    dino_license = require_file(
        DINO_SNAPSHOT / DINO_LICENSE_NAME,
        "DINOv3 license",
    )
    if sha256_file(source_license) != SOURCE_LICENSE_SHA256:
        raise RuntimeError("LDA source license hash mismatch")
    if sha256_file(apache_license) != APACHE_LICENSE_SHA256:
        raise RuntimeError("Apache 2.0 license hash mismatch")
    if sha256_file(dino_license) != DINO_LICENSE_SHA256:
        raise RuntimeError("Pinned DINOv3 license hash mismatch")

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
    (RELEASE_ROOT / "LICENSE").write_text(license_index())
    shutil.copyfile(source_license, RELEASE_ROOT / "CC-BY-NC-4.0.md")
    shutil.copyfile(apache_license, RELEASE_ROOT / "APACHE-2.0.txt")
    shutil.copyfile(dino_license, RELEASE_ROOT / "DINOv3-LICENSE.md")
    shutil.copytree(
        DINO_SNAPSHOT,
        RELEASE_ROOT / "pretrained" / DINO_SNAPSHOT.name,
        ignore=shutil.ignore_patterns(".cache"),
    )
    (RELEASE_ROOT / "README.md").write_text(render_card(source_commit()))
    (RELEASE_ROOT / "NOTICE").write_text(
        "LDA RoboCasa TReqs reproducibility canary\n\n"
        "Derived from Wayer2/LDA-robocasa and reproducible-ai/LDA-1B.\n"
        "Includes Qwen3-VL and DINOv3 components embedded in the assembled checkpoint.\n"
        "Built with DINOv3.\n\n"
        "Licensing is component-specific; see LICENSE, CC-BY-NC-4.0.md, "
        "APACHE-2.0.txt, and DINOv3-LICENSE.md.\n"
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
        "licenses": [
            {
                "component": "LDA source and bundled demo dataset",
                "id": "CC-BY-NC-4.0",
                "file": "CC-BY-NC-4.0.md",
                "sha256": SOURCE_LICENSE_SHA256,
            },
            {
                "component": f"{BASE_MODEL_ID} and {QWEN_MODEL_ID} upstream model metadata",
                "id": "Apache-2.0",
                "file": "APACHE-2.0.txt",
                "sha256": APACHE_LICENSE_SHA256,
            },
            {
                "component": DINO_MODEL_ID,
                "id": "LicenseRef-DINOv3",
                "file": "DINOv3-LICENSE.md",
                "revision": DINO_MODEL_REVISION,
                "sha256": DINO_LICENSE_SHA256,
            },
        ],
        "evaluation": evaluation,
    }
    (RELEASE_ROOT / "publication.json").write_text(json.dumps(publication, indent=2, sort_keys=True) + "\n")
    print(f"Packaged release for hf://{PUBLICATION_REPO_ID}/{PUBLICATION_VERSION}")


if __name__ == "__main__":
    main()
