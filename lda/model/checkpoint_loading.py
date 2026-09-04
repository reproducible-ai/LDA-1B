"""Safe checkpoint loading and packaged-resource path helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_tensor_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Load a tensor-only state dictionary without enabling pickle execution."""
    path = Path(checkpoint_path)
    try:
        checkpoint: Any = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (RuntimeError, TypeError):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    if not isinstance(checkpoint, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in checkpoint.items()
    ):
        raise TypeError("Checkpoint must be a string-keyed dictionary of tensors")
    return checkpoint


def resolve_checkpoint_resource_path(
    resource_path: str | Path,
    checkpoint_path: str | Path,
) -> str:
    """Resolve a packaged relative resource path from the release root."""
    path = Path(resource_path)
    if path.is_absolute():
        return str(path)
    release_root = Path(checkpoint_path).absolute().parents[1]
    return str((release_root / path).resolve())
