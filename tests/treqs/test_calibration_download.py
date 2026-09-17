"""The selected 24 task trees must not trigger a scan of the whole HF repo."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import HfHubHTTPError
from requests import Response

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "calibration_prepare", SOURCE / ".treqs/scripts/prepare_robocasa_calibration.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.mark.parametrize("rate_limited", [False, True])
def test_only_selected_subtrees_are_enumerated_and_downloaded(tmp_path, monkeypatch, rate_limited):
    import huggingface_hub._snapshot_download as snapshot_module

    selected = ["task-one", "task-two"]
    queries, downloads = [], []
    sleeps, attempts = [], []

    class FakeApi:
        def __init__(self, *args, **kwargs):
            pass

        def repo_info(self, *args, **kwargs):
            # The real repository's metadata is truncated: the SDK falls back
            # to enumerating its complete tree before applying allow_patterns.
            return SimpleNamespace(sha="a" * 40, siblings=None)

        def list_repo_tree(self, *, path_in_repo=None, **kwargs):
            assert path_in_repo in selected, "whole repository enumeration before filtering"
            assert kwargs["revision"] == "a" * 40
            assert kwargs["repo_type"] == "dataset"
            queries.append(path_in_repo)
            return [RepoFile(path=path_in_repo + "/meta/info.json", size=2, oid="b" * 40)]

    def download(*, filename, local_dir, **kwargs):
        assert kwargs["revision"] == "a" * 40
        assert kwargs["repo_type"] == "dataset"
        attempts.append(filename)
        if rate_limited and len(attempts) == 1:
            response = Response()
            response.status_code = 429
            response.headers["RateLimit"] = '"resolvers";r=0;t=2'
            raise HfHubHTTPError("Too Many Requests", response=response)
        downloads.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(snapshot_module, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(prepare.time, "sleep", sleeps.append)
    config = {"datasetRepository": "owner/dataset", "datasetRevision": "a" * 40, "datasetFolders": selected}
    result = prepare.download_dataset(config, tmp_path / "dataset", "test-token")
    assert result == tmp_path / "dataset"
    assert queries == selected
    assert sorted(downloads) == [folder + "/meta/info.json" for folder in selected]
    assert sleeps == ([3] if rate_limited else [])


@pytest.mark.parametrize("status,expected_attempts", [(403, 1), (429, 7)])
def test_download_errors_fail_without_unbounded_retries(monkeypatch, status, expected_attempts):
    attempts, sleeps = [], []
    response = Response()
    response.status_code = status
    response.headers["Retry-After"] = "5"
    error = HfHubHTTPError("download rejected", response=response)

    def rejected():
        attempts.append(1)
        raise error

    monkeypatch.setattr(prepare.time, "sleep", sleeps.append)
    with pytest.raises(HfHubHTTPError) as caught:
        prepare.retry_hub_rate_limit(rejected)
    assert caught.value is error
    assert len(attempts) == expected_attempts
    assert sleeps == [6] * (expected_attempts - 1)
