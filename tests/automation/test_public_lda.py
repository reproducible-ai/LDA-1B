"""Public LDA host checks, separate from worker/GPU recipe tests."""
import hashlib
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PINNED = os.environ.get("PUBLIC_CANARY_SUPERVISOR_SCRIPTS")
if not PINNED:
    pytest.skip("Set PUBLIC_CANARY_SUPERVISOR_SCRIPTS to the pinned supervisor source", allow_module_level=True)
sys.path.insert(0, str(ROOT / ".treqs/scripts"))
sys.path.append(PINNED)
import lda_public_verification as verify
import public_lda_canary as adapter
import public_canary_supervisor as runtime


def test_publication_uses_http_transport_without_changing_input_downloads():
    workflow = yaml.safe_load((ROOT / ".treqs/workflows/robocasa-demo-canary.yaml").read_text())
    tasks = workflow
    assert "env HF_HUB_DISABLE_XET=1 roar put " in tasks["publish"]["command"]
    assert "HF_HUB_DISABLE_XET" not in tasks["fetch"]["command"]


def test_pinned_hub_negotiates_only_http_when_xet_is_disabled(tmp_path):
    python = os.environ.get("PUBLIC_CANARY_UPLOAD_PYTHON")
    if not python:
        pytest.skip("Set PUBLIC_CANARY_UPLOAD_PYTHON to the pinned HF 0.36 environment")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    script = '''
import sys
import huggingface_hub as hub
import huggingface_hub._commit_api as api
assert hub.__version__ == "0.36.0"
assert not api.is_xet_available()
observed=[]
def batch(**kwargs):
    observed.append(kwargs["transfers"])
    assert kwargs["transfers"] == ["basic", "multipart"]
    return [], [], "basic"
api.post_lfs_batch_info=batch
api._upload_files(additions=[hub.CommitOperationAdd(path_in_repo="checkpoint.pt", path_or_fileobj=sys.argv[1])],
                  repo_type="model", repo_id="owner/model", headers={})
assert observed
'''
    result = subprocess.run([python, "-c", script, str(checkpoint)],
        env={**os.environ, "HF_HUB_DISABLE_XET": "1"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_retry_only_accepts_the_pinned_partial_publication():
    files = {"README.md", f"{verify.PREFIX}/result.json", f"{verify.PREFIX}/artifact-manifest.json"}
    plan = {"priorPublicAttempt": {"repositoryRevision": "a" * 40}}
    verify.check_publication_inventory(plan, "a" * 40, files)
    with pytest.raises(ValueError, match="changed"):
        verify.check_publication_inventory(plan, "b" * 40, files)
    with pytest.raises(ValueError, match="checkpoint artifacts"):
        verify.check_publication_inventory(plan, "a" * 40, files | {f"{verify.PREFIX}/{verify.CHECKPOINT}"})
    with pytest.raises(ValueError, match="checkpoint artifacts"):
        verify.check_publication_inventory({}, "a" * 40, files)


def test_retry_budget_reserves_idle_shutdown_and_confirmation():
    plan = {"shutdownReserveUsd": 1.2, "conservativeHourlyUsd": 3.5, "stopAtUsd": 1.5,
            "budgetUsd": 2.7, "maxJobSeconds": 1500}
    adapter.validate_retry_budget(plan, 15)
    with pytest.raises(RuntimeError, match="reserve"):
        adapter.validate_retry_budget({**plan, "shutdownReserveUsd": 0.5}, 15)
    with pytest.raises(RuntimeError, match="reserve"):
        adapter.validate_retry_budget({**plan, "maxJobSeconds": 1800}, 15)


def test_huggingface_custom_license_name_is_a_valid_metadata_slug():
    text = (ROOT / ".treqs/assets/robocasa-demo-canary-model-card.md").read_text()
    metadata = yaml.safe_load(text.split("---", 2)[1])
    assert re.fullmatch(r"[a-z0-9.-]+", metadata["license_name"])


def test_stream_verifies_all_bytes_with_bounded_reads():
    payload = b"checkpoint" * 1000000

    class Bounded(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 8 * 1024 * 1024
            return super().read(size)

    progress = []
    assert verify.stream_digest(Bounded(payload), len(payload), progress.append) == hashlib.sha256(payload).hexdigest()
    assert progress[-1] == len(payload)


@pytest.mark.parametrize("declared", [2, 4])
def test_stream_rejects_truncation_and_oversize(declared):
    with pytest.raises(ValueError):
        verify.stream_digest(io.BytesIO(b"123"), declared)


def topology():
    trained = {"artifactHash": "weight-hash", "path": verify.TRAINED_PATH, "artifact": {"size": "11"}}
    released = {**trained, "path": f"{verify.PREFIX}/{verify.CHECKPOINT}"}
    return {"total": 3, "jobs": [
        {"jobUid": "train", "command": "python .treqs/scripts/run_robocasa_canary.py", "inputs": [], "outputs": [trained]},
        {"jobUid": "package", "command": "roar run -n package python package.py", "inputs": [trained], "outputs": [released]},
        {"jobUid": "put", "command": "roar put release hf://owner/model --public", "inputs": [released], "outputs": []}]}


JOB = {"lineagePublishedSessionHash": "a" * 64, "lineagePublicationCounts": {"jobs": 3, "artifacts": 1, "links": 4}}


def test_lineage_handles_byte_identical_packaging_copy_and_requires_actual_training():
    graph = topology()
    assert verify.verify_lineage(JOB, graph, 11)["trainingToPutArtifactHashes"] == ["weight-hash"]
    graph["jobs"][0]["command"] = "python unrelated.py"
    with pytest.raises(ValueError, match="actual training"):
        verify.verify_lineage(JOB, graph, 11)


@pytest.mark.parametrize("corrupt", [False, True])
def test_public_readback_binds_streamed_checkpoint_result_inventory_and_source(tmp_path, monkeypatch, corrupt):
    remote, root = tmp_path / "remote", tmp_path / "run"
    remote.mkdir(); root.mkdir()
    payload = b"checkpoint!"
    digest = hashlib.sha256(payload).hexdigest()
    evaluation = {"status": "passed", "inputs_verified": True, "optimizer_steps_completed": 1, "global_step": 1,
                  "base_checkpoint": {"sha256": "b" * 64},
                  "checkpoint": {"sha256": digest, "size": len(payload), "changed_tensor": "action_model.weight", "tensor_count": 1441}}
    result = {"optimizerSteps": 1, "loadVerified": True, "artifactSha256": digest,
              "artifactSizeBytes": len(payload), "evaluation": evaluation}
    files = {verify.CHECKPOINT: payload,
             "loader/evaluation.json": json.dumps(evaluation).encode(),
             "loader/publication.json": json.dumps({"source_commit": "c" * 40, "repository": "owner/model", "evaluation": evaluation}).encode()}
    for name in ("README.md", "LICENSE", "NOTICE", "CC-BY-NC-4.0.md", "APACHE-2.0.txt", "DINOv3-LICENSE.md",
                 "input-manifest.json", "config.yaml", "dataset_statistics.json"):
        files[f"loader/{name}"] = name.encode()
    entries = [{"path": name, "sha256": hashlib.sha256(data).hexdigest(), "sizeBytes": len(data)} for name, data in sorted(files.items())]
    for name, data in files.items():
        path = remote / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    manifest = {"schema": "reproai.artifact-manifest/v1", "format": "pytorch-state-dict", "loadVerified": True, "files": entries}
    (remote / "artifact-manifest.json").write_text(json.dumps(manifest))
    (remote / "result.json").write_text(json.dumps(result))
    content = f"E2E_ARTIFACT={json.dumps({'files': entries})}\nE2E_RESULT={json.dumps(result)}\n"
    (root / "workload.jsonl").write_text(json.dumps({"result": {"chunks": [{"content": content}]}}) + "\n")

    class AnonymousApi:
        def __init__(self, *, token):
            assert token is False

        def model_info(self, repo, **kwargs):
            assert kwargs["token"] is False
            return SimpleNamespace(private=False, sha="d" * 40,
                siblings=[SimpleNamespace(rfilename=f"{verify.PREFIX}/{p.relative_to(remote)}") for p in remote.rglob("*") if p.is_file()])

    def download(repo, revision, name, destination):
        assert revision == "d" * 40
        assert not name.endswith(".pt"), "Large checkpoint must not be retained on disk"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / name.removeprefix(verify.PREFIX + "/"), destination)
        return destination

    monkeypatch.setattr(verify, "HfApi", AnonymousApi)
    monkeypatch.setattr(verify, "download_public", download)
    monkeypatch.setattr(verify, "urlopen", lambda *a, **kw: io.BytesIO(b"corruption!" if corrupt else payload))
    monkeypatch.setattr(verify, "public_json", lambda url: topology() if "/jobs?" in url else {})
    if corrupt:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            verify.verify_release({"hfRepository": "owner/model", "sourceCommit": "c" * 40}, root, JOB)
    else:
        receipt = verify.verify_release({"hfRepository": "owner/model", "sourceCommit": "c" * 40}, root, JOB)
        assert receipt["verified"] and len(receipt["files"]) == len(entries)
        assert not list((root / "readback").rglob("*.pt"))


def test_lda_adapter_uses_real_cli_with_public_mode_and_correct_workflow(tmp_path, monkeypatch):
    executable = tmp_path / "treqs"
    executable.write_text("#!/usr/bin/env python3\nimport json,sys,os\n"
        "assert 'TREQS_API_TOKEN' not in os.environ\n"
        "a=sys.argv[1:]\nassert a[:3] == ['--json','tr','create']\n"
        "assert a[a.index('--workflow-path')+1] == '.treqs/workflows/robocasa-demo-canary.yaml'\n"
        "assert a[a.index('--lineage-mode')+1] == 'public'\n"
        "assert a[a.index('--source-commit')+1] == 'c'*40\n"
        "print(json.dumps({'id':'request'}))\n")
    executable.chmod(0o755)
    monkeypatch.setenv("TREQS_API_TOKEN", "stale-test-token")
    plan = {"treqs": str(executable), "controlWorkspace": str(tmp_path), "sourceCommit": "c" * 40,
            "title": "test", "sourceBranch": "tb/test", "targetId": "target"}
    assert adapter.build_actions(runtime, plan, tmp_path).create() == {"id": "request"}


@pytest.mark.parametrize("retry", [False, True])
def test_notes_rendering_is_idempotent_and_keeps_historical_evidence(tmp_path, retry):
    source, output = tmp_path / "template", tmp_path / "notes"
    source.mkdir()
    (source / "README.md").write_text("# LDA-1B\n\nPrivate capture history.\n")
    (source / "row.json").write_text(json.dumps({"verified": False, "rebuild": {"costUsd": 1.24},
        "dataWarnings": ["Artifacts remain private"], "artifacts": [{"sha256": "old"}]}))
    (source / "costs.md").write_text("# Private campaign costs\n\n$6.24\n")
    plan = {"notesTemplate": str(source), "notesTemplateSha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
            "sourceCommit": "c" * 40, "supervisorCommit": "d" * 40, "runId": "test", "runDate": "2026-09-15"}
    if retry:
        plan["priorPublicAttempt"] = {"jobId": "failed-job", "costUsd": 2.0}
    state = {"verification": {"hfUrl": "https://hf.test/pinned", "glaasUrl": "https://glaas.test/dag", "lineage": {"dagHash": "a" * 64},
                             "result": {"artifactSizeBytes": 11, "artifactSha256": "new"}},
             "jobId": "job", "requestId": "request", "observedCostUsd": 1.5, "allocationActive": False}
    verify.render_notes(plan, state, output)
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    verify.render_notes(plan, state, output)
    assert before == {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert "Private capture history" in (output / "README.md").read_text()
    assert json.loads((output / "row.json").read_text())["verified"] is False
    row = json.loads((output / "row.json").read_text())
    assert row["artifacts"][0]["sha256"] == "new"
    assert len(row["runs"]) == (2 if retry else 1)
    assert "Artifacts remain private" not in row["dataWarnings"]
    assert "$6.24" in (output / "costs.md").read_text()
    if retry:
        assert "$3.50" in (output / "PUBLIC-RELEASE.md").read_text()
        assert row["runs"][0]["outcome"] == "failed"


def test_notes_schema_accepts_historical_inventory_and_rejects_bad_digest():
    row = {"name": "LDA-1B", "slug": "lda-1b", "org": "reproducible-ai", "type": "robotics",
           "date": "2026-09-15", "status": "progress", "summary": "One-step canary", "trainingMode": "partial",
           "verified": False, "rebuild": {"time": "1000s", "hw": "GPU", "costUsd": 1.5},
           "runs": [{"label": "Public canary", "attemptId": "public-test", "outcome": "ok", "costUsd": 1.5}],
           "truncationNote": "One of 300000 optimizer steps",
           "truncation": {"ran": 1, "full": 300000, "unit": "optimizer steps", "note": "Bounded canary"},
           "truncated": True, "artifacts": [{"path": verify.CHECKPOINT, "bytes": 11, "sha256": "a" * 64, "note": "Public"}]}
    verify.validate_notes_row(row)
    row["artifacts"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="inventory"):
        verify.validate_notes_row(row)
