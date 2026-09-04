from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HELPERS_PATH = ROOT / "lda" / "model" / "checkpoint_loading.py"


def load_helpers():
    assert HELPERS_PATH.is_file(), "checkpoint loading helpers are missing"
    spec = importlib.util.spec_from_file_location("checkpoint_loading", HELPERS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_dict_loading_keeps_weights_only_on_mmap_fallback(monkeypatch, tmp_path):
    helpers = load_helpers()
    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        if kwargs.get("mmap"):
            raise RuntimeError("mmap unavailable")
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(helpers.torch, "load", fake_load)
    checkpoint = tmp_path / "checkpoint.pt"

    state_dict = helpers.load_tensor_state_dict(checkpoint)

    assert list(state_dict) == ["weight"]
    assert calls == [
        (
            checkpoint,
            {"map_location": "cpu", "weights_only": True, "mmap": True},
        ),
        (
            checkpoint,
            {"map_location": "cpu", "weights_only": True},
        ),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        [torch.ones(1)],
        {1: torch.ones(1)},
        {"weight": "not a tensor"},
    ],
)
def test_state_dict_loading_rejects_non_tensor_mappings(monkeypatch, tmp_path, payload):
    helpers = load_helpers()
    monkeypatch.setattr(helpers.torch, "load", lambda *args, **kwargs: payload)

    with pytest.raises(TypeError, match="string-keyed dictionary of tensors"):
        helpers.load_tensor_state_dict(tmp_path / "checkpoint.pt")


def test_relative_resource_paths_resolve_from_release_root(tmp_path, monkeypatch):
    helpers = load_helpers()
    release_root = tmp_path / "release"
    checkpoint = release_root / "checkpoints" / "steps_1_pytorch_model.pt"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = helpers.resolve_checkpoint_resource_path("pretrained", checkpoint)

    assert resolved == str((release_root / "pretrained").resolve())


def test_checkpoint_symlink_does_not_move_the_release_root(tmp_path):
    helpers = load_helpers()
    release = tmp_path / "release"
    checkpoints = release / "checkpoints"
    checkpoints.mkdir(parents=True)
    (release / "pretrained").mkdir()
    blob = tmp_path / "blobs" / "checkpoint.pt"
    blob.parent.mkdir()
    blob.write_bytes(b"checkpoint")
    checkpoint = checkpoints / "model.pt"
    checkpoint.symlink_to(blob)

    resolved = helpers.resolve_checkpoint_resource_path("pretrained", checkpoint)

    assert resolved == str((release / "pretrained").resolve())


def test_runtime_loaders_use_shared_safe_helpers():
    trainer = (ROOT / "lda" / "training" / "trainer_utils" / "trainer_tools.py").read_text()
    framework = (ROOT / "lda" / "model" / "framework" / "base_framework.py").read_text()

    assert "load_tensor_state_dict(checkpoint_path)" in trainer
    assert "load_tensor_state_dict(pretrained_checkpoint)" in framework
    assert "Path(pretrained_checkpoint).resolve()" not in framework
    assert "Path(pretrained_checkpoint).absolute()" in framework
    assert "resolve_checkpoint_resource_path(" in framework
    assert "model_config.framework.action_model.vision_encoder_path" in framework
