"""Download pinned model inputs and hash the committed demo dataset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from lda_canary_contract import (
    ARTIFACT_ROOT,
    BASE_CHECKPOINT,
    BASE_CHECKPOINT_NAME,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BASE_SNAPSHOT,
    DATASET_PATH,
    DINO_CONFIG,
    DINO_LICENSE,
    DINO_LICENSE_NAME,
    DINO_LICENSE_SHA256,
    DINO_MODEL_ID,
    DINO_MODEL_REVISION,
    DINO_PREPROCESSOR_CONFIG,
    DINO_SNAPSHOT,
    EXPECTED_EPISODES,
    INPUT_MANIFEST_PATH,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
    QWEN_SNAPSHOT,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if ".cache" in path.parts:
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def download(repo_id: str, revision: str, destination: Path, token: str, **kwargs) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=destination,
        **kwargs,
    )
    return Path(resolved)


def main() -> None:
    if INPUT_MANIFEST_PATH.name != "input-manifest.json":
        raise RuntimeError("Input manifest must be named input-manifest.json")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if INPUT_MANIFEST_PATH.exists():
        raise RuntimeError(f"Refusing to reuse an existing input manifest: {INPUT_MANIFEST_PATH}")

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    download(
        repo_id=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        destination=BASE_SNAPSHOT,
        token=token,
        allow_patterns=[
            ".gitattributes",
            "README.md",
            "config.yaml",
            "dataset_statistics.json",
            BASE_CHECKPOINT_NAME,
        ],
    )
    source_checkpoint = BASE_SNAPSHOT / BASE_CHECKPOINT_NAME
    BASE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if not source_checkpoint.is_file():
        raise RuntimeError(f"Missing downloaded base checkpoint: {source_checkpoint}")
    shutil.move(source_checkpoint, BASE_CHECKPOINT)

    download(
        repo_id=QWEN_MODEL_ID,
        revision=QWEN_MODEL_REVISION,
        destination=QWEN_SNAPSHOT,
        token=token,
    )
    download(
        repo_id=DINO_MODEL_ID,
        revision=DINO_MODEL_REVISION,
        destination=DINO_SNAPSHOT,
        token=token,
        allow_patterns=[DINO_LICENSE_NAME],
    )
    if sha256_file(DINO_LICENSE) != DINO_LICENSE_SHA256:
        raise RuntimeError("Pinned DINOv3 license hash mismatch")
    DINO_SNAPSHOT.mkdir(parents=True, exist_ok=True)
    (DINO_SNAPSHOT / "config.json").write_text(json.dumps(DINO_CONFIG, indent=2, sort_keys=True) + "\n")
    (DINO_SNAPSHOT / "preprocessor_config.json").write_text(
        json.dumps(DINO_PREPROCESSOR_CONFIG, indent=2, sort_keys=True) + "\n"
    )

    episodes = sorted(DATASET_PATH.glob("data/chunk-*/episode_*.parquet"))
    if len(episodes) != EXPECTED_EPISODES:
        raise RuntimeError(f"Expected {EXPECTED_EPISODES} demo episodes; found {len(episodes)}")

    required = (
        BASE_SNAPSHOT / "config.yaml",
        BASE_SNAPSHOT / "dataset_statistics.json",
        BASE_CHECKPOINT,
        QWEN_SNAPSHOT / "config.json",
        DINO_SNAPSHOT / "config.json",
        DINO_LICENSE,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing pinned input files: {missing}")

    manifest = {
        "schema_version": 1,
        "inputs": {
            "base_model": {"repo_id": BASE_MODEL_ID, "revision": BASE_MODEL_REVISION},
            "qwen_model": {"repo_id": QWEN_MODEL_ID, "revision": QWEN_MODEL_REVISION},
            "dino_model": {
                "repo_id": DINO_MODEL_ID,
                "revision": DINO_MODEL_REVISION,
                "weights_source": "embedded in strict-loaded base checkpoint",
                "architecture_config_source": "vendored DINOv3ViTConfig defaults",
            },
            "dataset": {
                "path": str(DATASET_PATH.relative_to(DATASET_PATH.parents[2])),
                "episodes": len(episodes),
            },
        },
        "files": {
            "base_model": hash_tree(BASE_SNAPSHOT),
            "qwen_model": hash_tree(QWEN_SNAPSHOT),
            "dino_model": hash_tree(DINO_SNAPSHOT),
            "dataset": hash_tree(DATASET_PATH),
        },
    }
    INPUT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Prepared and hashed pinned inputs: {INPUT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
