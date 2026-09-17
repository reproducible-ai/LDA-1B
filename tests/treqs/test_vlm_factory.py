"""Model family selection must survive relocating pinned snapshots."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def factory(monkeypatch):
    # Load the real factory without importing GPU-only interface dependencies.
    name = "test_vlm_factory_package"
    path = ROOT / "lda/model/modules/vlm/__init__.py"
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    for child, interface in (
        ("QWen3", "_QWen3_VL_Interface"),
        ("QWen2_5", "_QWen_VL_Interface"),
        ("Florence2", "_Florence_Interface"),
    ):
        stub = ModuleType(f"{name}.{child}")
        setattr(stub, interface, lambda config, family=child: (family, config))
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    return module.get_vlm_model


def config(path):
    return SimpleNamespace(framework=SimpleNamespace(qwenvl=SimpleNamespace(base_vlm=str(path))))


@pytest.mark.parametrize(
    "model_type,family", [("qwen3_vl", "QWen3"), ("qwen2_5_vl", "QWen2_5"), ("florence2", "Florence2")]
)
def test_local_snapshot_uses_saved_model_type(factory, tmp_path, model_type, family):
    snapshot = tmp_path / "qwen"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type": model_type}))
    cfg = config(snapshot)
    assert factory(cfg) == (family, cfg)


@pytest.mark.parametrize(
    "model_id,family",
    [
        ("Qwen/Qwen3-VL-2B-Instruct", "QWen3"),
        ("Qwen/Qwen2.5-VL-3B-Instruct", "QWen2_5"),
        ("microsoft/Florence-2-large", "Florence2"),
    ],
)
def test_remote_model_ids_keep_name_dispatch(factory, model_id, family):
    cfg = config(model_id)
    assert factory(cfg) == (family, cfg)


def test_unsupported_local_type_is_not_guessed_from_directory_name(factory, tmp_path):
    snapshot = tmp_path / "Qwen3-VL"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type": "unsupported"}))
    with pytest.raises(NotImplementedError, match="unsupported"):
        factory(config(snapshot))
