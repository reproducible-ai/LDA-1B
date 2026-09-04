"""Verify the one-step checkpoint and prove an action tensor changed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from lda_canary_contract import (
    BASE_CHECKPOINT,
    BASE_SNAPSHOT,
    DATASET_PATH,
    DINO_SNAPSHOT,
    EVALUATION_PATH,
    INPUT_MANIFEST_PATH,
    QWEN_SNAPSHOT,
    RUN_DIR,
    TRAINED_CHECKPOINT,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def verify_manifest_files(records: list[dict[str, object]], root: Path) -> None:
    expected = {str(record["path"]): record for record in records}
    actual = {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"Input manifest path mismatch under {root}; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for relative_path, record in expected.items():
        path = actual[relative_path]
        if path.stat().st_size != record["size"]:
            raise RuntimeError(f"Input size changed after fetch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Input hash changed after fetch: {path}")


def main() -> None:
    if not INPUT_MANIFEST_PATH.is_file():
        raise RuntimeError(f"Missing input manifest: {INPUT_MANIFEST_PATH}")
    if not BASE_CHECKPOINT.is_file():
        raise RuntimeError(f"Missing base checkpoint: {BASE_CHECKPOINT}")
    if not TRAINED_CHECKPOINT.is_file():
        raise RuntimeError(f"Missing trained checkpoint: {TRAINED_CHECKPOINT}")

    manifest = json.loads(INPUT_MANIFEST_PATH.read_text())
    manifest_roots = {
        "base_model": BASE_SNAPSHOT,
        "qwen_model": QWEN_SNAPSHOT,
        "dino_model": DINO_SNAPSHOT,
        "dataset": DATASET_PATH,
    }
    for name, root in manifest_roots.items():
        try:
            records = manifest["files"][name]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Input manifest is missing file records for {name}") from exc
        verify_manifest_files(records, root)

    summary_path = RUN_DIR / "summary.jsonl"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing training summary: {summary_path}")
    summaries = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
    global_step = summaries[-1].get("steps") if summaries else None
    if global_step != 1:
        raise RuntimeError(f"Expected global_step=1; found {global_step}")
    optimizer_steps_completed = summaries[-1].get("optimizer_steps_completed")
    if optimizer_steps_completed != 1:
        raise RuntimeError(
            f"Expected one completed optimizer step; found {optimizer_steps_completed}"
        )
    trainable_parameters = set(summaries[-1].get("trainable_parameters") or [])
    if not trainable_parameters:
        raise RuntimeError("Training summary contains no trainable parameter names")

    base_sha256 = sha256_file(BASE_CHECKPOINT)
    checkpoint_sha256 = sha256_file(TRAINED_CHECKPOINT)
    if base_sha256 == checkpoint_sha256:
        raise RuntimeError("Fine-tuned checkpoint is byte-identical to the base checkpoint")

    base_state = load_state(BASE_CHECKPOINT)
    trained_state = load_state(TRAINED_CHECKPOINT)
    if set(base_state) != set(trained_state):
        missing = sorted(set(base_state) - set(trained_state))
        unexpected = sorted(set(trained_state) - set(base_state))
        raise RuntimeError(
            f"Checkpoint key mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    for name, tensor in trained_state.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite floating tensor in trained checkpoint: {name}")

    changed_tensor = None
    for name in sorted(key for key in trained_state if key.startswith("action_model.")):
        if name not in trainable_parameters:
            continue
        before = base_state[name]
        after = trained_state[name]
        if (
            before.is_floating_point()
            and torch.isfinite(after).all()
            and not torch.allclose(before, after, rtol=0, atol=0, equal_nan=True)
        ):
            changed_tensor = name
            break
    if changed_tensor is None:
        raise RuntimeError("No finite trainable action_model parameter changed after the optimizer step")
    if changed_tensor not in trainable_parameters:
        raise RuntimeError("Changed tensor was not recorded as a trainable parameter")

    first_tensor = next(iter(sorted(trained_state)))
    result = {
        "schema_version": 1,
        "status": "passed",
        "inputs_verified": True,
        "global_step": global_step,
        "optimizer_steps_completed": optimizer_steps_completed,
        "base_checkpoint": {
            "path": str(BASE_CHECKPOINT),
            "size": BASE_CHECKPOINT.stat().st_size,
            "sha256": base_sha256,
        },
        "checkpoint": {
            "path": str(TRAINED_CHECKPOINT),
            "size": TRAINED_CHECKPOINT.stat().st_size,
            "sha256": checkpoint_sha256,
            "tensor_count": len(trained_state),
            "first_tensor": {
                "name": first_tensor,
                "shape": list(trained_state[first_tensor].shape),
                "dtype": str(trained_state[first_tensor].dtype),
            },
            "changed_tensor": changed_tensor,
        },
    }
    EVALUATION_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"LDA canary passed at global_step=1; changed_tensor={changed_tensor}")


if __name__ == "__main__":
    main()
