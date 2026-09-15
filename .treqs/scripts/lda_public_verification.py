"""Public LDA read-back, with bounded memory and no retained checkpoint copy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.request import urlopen

from huggingface_hub import HfApi, get_token, hf_hub_download, hf_hub_url
from huggingface_hub.errors import RepositoryNotFoundError

from public_canary_supervisor import atomic_json
from public_canary_verify import require, public_json, log_receipts, download_public
from reproai_harness.artifact_verifier import _parse_manifest, _sha256
from reproai_harness.lineage import normalize_lineage_topology
from reproai_harness.notes_draft import _ROW_VALIDATOR

PREFIX = "artifacts/lda-robocasa-canary/release/checkpoints"
CHECKPOINT = "LDA-robocasa-treqs-canary.pt"
TRAINED_PATH = "artifacts/lda-robocasa-canary/training/robocasa-demo-canary/checkpoints/steps_1_pytorch_model.pt"
MODEL_DIRECTORY = "028-robotics-lda-1b"


def prepare_repository(plan, root):
    from check_hf_access import check_file, check_public_writable_repo, has_org_write_access, request_json
    from lda_canary_contract import BASE_MODEL_ID, BASE_MODEL_REVISION, QWEN_MODEL_ID, QWEN_MODEL_REVISION, DINO_MODEL_ID, DINO_MODEL_REVISION, DINO_LICENSE_NAME
    token = get_token()
    require(bool(token), "Saved Hugging Face token is required")
    for repo, revision, name in ((BASE_MODEL_ID, BASE_MODEL_REVISION, "config.yaml"),
                                  (QWEN_MODEL_ID, QWEN_MODEL_REVISION, "config.json"),
                                  (DINO_MODEL_ID, DINO_MODEL_REVISION, DINO_LICENSE_NAME)):
        check_file(repo, revision, name, token)
    identity = request_json("https://huggingface.co/api/whoami-v2", token)
    require(has_org_write_access(identity, "reproducible-ai"), "HF organization write access is required")
    api, repo = HfApi(token=token), plan["hfRepository"]
    try:
        info = api.model_info(repo)
    except RepositoryNotFoundError as exc:
        if exc.response.status_code != 404:
            raise
        api.create_repo(repo, private=False, exist_ok=True)
        info = api.model_info(repo)
    require(info.private is False, "Refusing to change an existing private repository")
    files = {f.rfilename for f in info.siblings}
    require(files <= {".gitattributes", "README.md"}, "Publication repository already contains artifacts")
    marker = f"<!-- public-canary:{plan['runId']} -->"
    if "README.md" in files:
        path = hf_hub_download(repo, "README.md", revision=info.sha, token=token, local_dir=str(root / "preflight"))
        require(marker in Path(path).read_text(), "Repository belongs to another publication")
    else:
        card = ("---\nlicense: other\nlicense_name: Component-specific licensing\nlicense_link: "
                f"https://github.com/reproducible-ai/LDA-1B/blob/{plan['sourceCommit']}/.treqs/README.md#scope\n---\n\n"
                "# LDA-1B RoboCasa public canary\n\nA fresh one-step public canary is being prepared. "
                "No checkpoint has passed public read-back yet.\n\nNon-commercial research only; "
                "CC BY-NC 4.0, Apache 2.0 and DINOv3 component terms apply. Built with DINOv3.\n\n" + marker + "\n")
        api.upload_file(repo_id=repo, path_in_repo="README.md", path_or_fileobj=card.encode(),
                        commit_message="docs: prepare authorized public LDA canary")
    check_public_writable_repo(repo, token)
    require(HfApi(token=False).model_info(repo, token=False).private is False, "Repository is not anonymously visible")
    # Validate the frozen notes template before any paid launch.
    for name, digest in plan["notesTemplateSha256"].items():
        require(_sha256(Path(plan["notesTemplate"]) / name) == digest, "Frozen notes template changed")
    atomic_json(root / "preflight.json", {"repository": repo, "public": True, "pinnedInputAccess": True,
                                          "writeAccess": True, "notesTemplateVerified": True})


def stream_digest(stream, expected_size, progress=lambda size: None):
    digest, size = hashlib.sha256(), 0
    while chunk := stream.read(8 * 1024 * 1024):
        size += len(chunk)
        require(size <= expected_size, "Public checkpoint is larger than the manifest")
        digest.update(chunk)
        if size // (64 * 1024 * 1024) != (size - len(chunk)) // (64 * 1024 * 1024):
            progress(size)
    require(size == expected_size, "Public checkpoint was truncated")
    progress(size)
    return digest.hexdigest()


def verify_lineage(job, graph, size):
    normalized = normalize_lineage_topology(dag_hash=job["lineagePublishedSessionHash"], payload=graph,
        publication_counts=job["lineagePublicationCounts"], artifact_path=f"{PREFIX}/{CHECKPOINT}", artifact_size_bytes=size)
    require(normalized["verified"], "Public checkpoint is not connected to PUT")
    # Packaging copies identical bytes to a new path. Match the actual training
    # producer and the publication consumer by artifact identity, not role labels.
    trained = {r["artifactHash"] for j in graph["jobs"] if "run_robocasa_canary.py" in j["command"]
               for r in j["outputs"] if (r.get("path") or "").endswith(TRAINED_PATH)}
    published = {r["artifactHash"] for j in graph["jobs"] if "roar put " in j["command"]
                 for r in j["inputs"] if (r.get("path") or "").endswith(f"{PREFIX}/{CHECKPOINT}")}
    require(bool(trained & published), "PUT is not connected to the actual training output")
    return {"dagHash": job["lineagePublishedSessionHash"], "counts": job["lineagePublicationCounts"],
            "trainingToPutArtifactHashes": sorted(trained & published)}


def verify_release(plan, root, job):
    repo = plan["hfRepository"]
    info = HfApi(token=False).model_info(repo, token=False, files_metadata=True)
    require(info.private is False and re.fullmatch(r"[0-9a-f]{40}", info.sha), "HF is not public and immutable")
    pin = root / "artifact-revision.json"
    if not pin.exists():
        atomic_json(pin, {"revision": info.sha})
    revision = json.loads(pin.read_text())["revision"]
    info = HfApi(token=False).model_info(repo, revision=revision, token=False, files_metadata=True)
    directory = root / "readback" / revision

    def get(name):
        return download_public(repo, revision, f"{PREFIX}/{name}", directory / name)

    manifest, entries = _parse_manifest(get("artifact-manifest.json"))
    require(manifest.get("loadVerified") is True and manifest.get("format") == "pytorch-state-dict", "Wrong checkpoint contract")
    required = {CHECKPOINT} | {f"loader/{name}" for name in (
        "README.md", "LICENSE", "NOTICE", "CC-BY-NC-4.0.md", "APACHE-2.0.txt", "DINOv3-LICENSE.md",
        "evaluation.json", "input-manifest.json", "publication.json", "config.yaml", "dataset_statistics.json")}
    require(required <= {entry["path"] for entry in entries}, "Manifest omits required loader resources or notices")
    expected = {f"{PREFIX}/{e['path']}" for e in entries} | {f"{PREFIX}/artifact-manifest.json", f"{PREFIX}/result.json"}
    actual = {f.rfilename for f in info.siblings if f.rfilename.startswith(PREFIX + "/")}
    require(actual == expected, "Remote inventory differs from manifest")
    receipts = log_receipts(root)
    require(receipts["ARTIFACT"]["files"] == entries, "Manifest differs from captured training receipt")
    result = json.loads(get("result.json").read_text())
    require(result == receipts["RESULT"], "Remote result differs from captured training result")
    evaluation = result["evaluation"]
    checkpoint = next(e for e in entries if e["path"] == CHECKPOINT)
    require(result.get("optimizerSteps") == 1 and result.get("loadVerified") is True
            and evaluation.get("status") == "passed" and evaluation.get("inputs_verified") is True
            and evaluation.get("optimizer_steps_completed") == 1 and evaluation.get("global_step") == 1,
            "One-step training contract failed")
    require(checkpoint["sha256"] == result["artifactSha256"] == evaluation["checkpoint"]["sha256"]
            and checkpoint["sizeBytes"] == result["artifactSizeBytes"] == evaluation["checkpoint"]["size"]
            and evaluation["base_checkpoint"]["sha256"] != checkpoint["sha256"]
            and evaluation["checkpoint"]["changed_tensor"].startswith("action_model.")
            and evaluation["checkpoint"]["tensor_count"] == 1441, "Checkpoint evaluation binding failed")
    dag = job["lineagePublishedSessionHash"]
    require(bool(re.fullmatch(r"[0-9a-f]{64}", dag)), "Invalid canonical DAG hash")
    public_json(f"https://api.glaas.ai/api/v1/public/sessions/{dag}")
    graph = public_json(f"https://api.glaas.ai/api/v1/public/sessions/{dag}/jobs?includeArtifacts=true")
    proof = verify_lineage(job, graph, checkpoint["sizeBytes"])
    atomic_json(root / "public-lineage.json", graph)
    checked = []
    for entry in sorted(entries, key=lambda e: e["path"] == CHECKPOINT):
        name = entry["path"]
        if name == CHECKPOINT:
            url = hf_hub_url(repo, f"{PREFIX}/{name}", revision=revision)
            with urlopen(url, timeout=60) as response:
                digest = stream_digest(response, entry["sizeBytes"], lambda size: atomic_json(
                    root / "download-progress.json", {"path": name, "bytes": size, "totalBytes": entry["sizeBytes"]}))
        else:
            path = get(name)
            require(path.stat().st_size == entry["sizeBytes"], f"Size mismatch: {name}")
            digest = _sha256(path)
        require(digest == entry["sha256"], f"SHA-256 mismatch: {name}")
        checked.append(entry)
        atomic_json(root / "readback-progress.json", {"revision": revision, "files": checked})
    publication = json.loads((directory / "loader/publication.json").read_text())
    require(publication["source_commit"] == plan["sourceCommit"] and publication["repository"] == repo
            and publication["evaluation"] == evaluation, "Publication identity differs from the run")
    require(json.loads((directory / "loader/evaluation.json").read_text()) == evaluation, "Evaluation sidecar differs")
    return {"schema": "reproai.public-canary/v1", "verified": True, "anonymousReadback": True,
            "repository": repo, "revision": revision, "path": PREFIX, "sourceCommit": plan["sourceCommit"],
            "result": result, "files": checked, "lineage": proof,
            "checkpointVerification": "Every byte anonymously streamed and SHA-256 verified; worker load/finite-tensor evaluation bound to the same digest.",
            "hfUrl": f"https://huggingface.co/{repo}/tree/{revision}/{PREFIX}", "glaasUrl": f"https://glaas.ai/dag/{dag}"}


def render_notes(plan, state, directory):
    receipt = state["verification"]
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in plan["notesTemplateSha256"].items():
        source = Path(plan["notesTemplate"]) / name
        require(_sha256(source) == digest, "Frozen notes template changed")
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    record = {**receipt, "runId": plan["runId"], "jobId": state["jobId"], "requestId": state["requestId"],
              "costUsd": state["observedCostUsd"], "computeStopped": not state["allocationActive"],
              "supervisorCommit": plan["supervisorCommit"]}
    atomic_json(directory / "evidence/public-release.json", record)
    (directory / "PUBLIC-RELEASE.md").write_text(
        "# Public LDA-1B RoboCasa canary\n\nA fresh one-step public run used the same pinned recipe, "
        "four demo episodes, batch four and frozen Qwen/DINO components as the successful private capture.\n\n"
        f"- [Immutable Hugging Face checkpoint]({receipt['hfUrl']})\n"
        f"- [Public GLaaS lineage]({receipt['glaasUrl']})\n"
        f"- [Executed source and automation](https://github.com/reproducible-ai/LDA-1B/tree/{plan['sourceCommit']}/.treqs)\n"
        f"- TReqs job: `{state['jobId']}`\n"
        f"- Full allocation cost: **${state['observedCostUsd']:.2f}**; compute stopped.\n\n"
        "The worker strict-loaded the checkpoint, checked all 1,441 tensors for finite floating values, "
        "and proved an action tensor changed after one optimizer update. The host independently streamed "
        "every published byte anonymously, verified the manifest sizes and SHA-256 digests, matched the "
        "logged result, and confirmed the actual training output feeds PUT in the public graph. "
        "The host does not claim a second full model load or an independent lineage-auditor verdict.\n\n"
        "The supervisor published this report automatically. The prior private capture and its audit remain "
        "historical evidence. This is a non-commercial demo training-path canary; cold replay, policy quality "
        "and semantic equivalence to RoboCasa remain untested. Roar's untracked-directory warning remains.\n\n"
        "CC BY-NC 4.0, Apache 2.0 and DINOv3 component terms are included with the checkpoint. Built with DINOv3. "
        "See [public verification evidence](evidence/public-release.json).\n")
    readme = directory / "README.md"
    heading, rest = readme.read_text().split("\n", 1)
    readme.write_text(heading + "\n\n## Latest public run\n\n"
                     "See the [fresh public canary](PUBLIC-RELEASE.md) for public artifacts, lineage and verification. "
                     "The original private capture is documented below.\n" + rest)
    row_path = directory / "row.json"
    row = json.loads(row_path.read_text())
    row.update(artifactUrl=receipt["hfUrl"], dagHash=receipt["lineage"]["dagHash"], dagUrl=receipt["glaasUrl"],
               date=plan["runDate"],
               statusSub="Public one-step capture passed",
               summary="A fresh public one-step LDA RoboCasa demo canary completed training, checkpoint publication, anonymous byte verification and full training-to-PUT lineage checks. The prior private capture remains documented. Model quality and cold reproduction are untested.",
               blocker="No cold rebuild or held-out policy evaluation; demo/RoboCasa semantic equivalence is untested.")
    row["rebuild"]["costUsd"] = state["observedCostUsd"]
    if state.get("job", {}).get("completedAt") and state["job"].get("startedAt"):
        from public_canary_supervisor import timestamp
        seconds = timestamp(state["job"]["completedAt"]) - timestamp(state["job"]["startedAt"])
        row["rebuild"]["time"] = f"{seconds:.3f}s"
    row["dataWarnings"] = [
        "Public links and rebuild fields describe the fresh public canary; the original private capture remains historical evidence.",
        "The host verified every published byte and training-to-PUT lineage; checkpoint load and finite-tensor checks were performed by the worker and bound to the same digest.",
        "No independent auditor verdict, cold replay, held-out policy evaluation or full-run cost estimate is claimed for the public run.",
        "Roar reported untracked artifact directories. The demo adapter does not establish RoboCasa semantic equivalence.",
    ]
    if receipt.get("result"):
        result = receipt["result"]
        row["artifacts"] = [{"path": CHECKPOINT, "bytes": result["artifactSizeBytes"],
                             "sha256": result["artifactSha256"],
                             "note": "Public one-step checkpoint; every byte independently streamed and SHA-256 checked. Worker load and tensor checks are bound to this digest."}]
        row.setdefault("runs", []).append({
            "label": "Public one-step canary", "attemptId": plan["runId"], "purpose": "public training capture",
            "params": "one optimizer step; 1 GPU; batch 4", "dataset": "Four source-pinned sim_pick_place demo episodes",
            "costUsd": state["observedCostUsd"], "costSource": "Finalized TReqs on-demand instance receipt",
            "schedulerStatus": "COMPLETED", "outcome": "ok", "note": f"Job {state['jobId']}; public byte and lineage verification passed; compute stopped."})
    costs = directory / "costs.md"
    if costs.exists():
        costs.write_text(costs.read_text() + "\n## Fresh public canary\n\n"
                         f"The new public run cost **${state['observedCostUsd']:.2f}**, including allocation shutdown. "
                         "The earlier ledger above describes the private campaign. `row.json.rebuild` now describes "
                         "this public run. See [PUBLIC-RELEASE.md](PUBLIC-RELEASE.md) for its evidence.\n")
    atomic_json(row_path, row)


def validate_notes_row(row):
    # The historical notes row also includes this optional artifact inventory,
    # which is not part of the harness's more restrictive generated-row schema.
    artifacts = row.get("artifacts", [])
    require(isinstance(artifacts, list), "Artifact inventory must be a list")
    for artifact in artifacts:
        require(set(artifact) == {"path", "bytes", "sha256", "note"}
                and artifact["path"] == CHECKPOINT
                and isinstance(artifact["bytes"], int) and artifact["bytes"] > 0
                and bool(re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]))
                and isinstance(artifact["note"], str), "Invalid notes checkpoint inventory")
    _ROW_VALIDATOR.validate({key: value for key, value in row.items() if key != "artifacts"})


def publish_notes(plan, root, state, actions):
    receipt = state["verification"]
    require(receipt["verified"] and not state["allocationActive"], "Public verification and shutdown are required")
    checkout = Path(plan["notesWorkspace"])
    branch = plan["notesBranch"]
    actions.command(["git", "fetch", "origin"], cwd=checkout)
    upstream = actions.command(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"], cwd=checkout).strip()
    require(upstream == f"origin/{branch}", "Notes branch tracking changed")
    actions.command(["git", "merge", "--ff-only", upstream], cwd=checkout)
    render_notes(plan, state, checkout / MODEL_DIRECTORY)
    validate_notes_row(json.loads((checkout / MODEL_DIRECTORY / "row.json").read_text()))
    actions.command(["git", "add", MODEL_DIRECTORY], cwd=checkout)
    if actions.command(["git", "diff", "--cached", "--name-only"], cwd=checkout).strip():
        actions.command(["git", "commit", "-m", "docs(lda): record verified public canary and lineage"], cwd=checkout)
    actions.command(["git", "push", "origin", f"HEAD:refs/heads/{branch}"], cwd=checkout, timeout=120)
    commit = actions.command(["git", "rev-parse", "HEAD"], cwd=checkout).strip()
    actions.env["GH_TOKEN"] = subprocess.run(["gh", "auth", "token", "--hostname", "github.com", "--user", "TrevorBasinger"],
        capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    prs = json.loads(actions.command(["gh", "pr", "list", "--repo", "reproducible-ai/notes", "--head", branch,
                                      "--state", "all", "--json", "number,state,url,headRefOid"]))
    require(len(prs) <= 1, "Ambiguous notes PR identity")
    if not prs:
        body = root / "notes-pr-body.md"
        body.write_text("Records a fresh public one-step LDA-1B RoboCasa canary, preserving the prior private "
                        "capture as historical evidence.\n\nThe supervisor verified anonymous checkpoint downloads, "
                        "full training-to-PUT lineage, all seven tasks, final allocation cost, and compute shutdown. "
                        "The report retains the model-quality, demo-adapter, licensing and cold-replay limitations.\n\n"
                        f"- [Public artifact]({receipt['hfUrl']})\n- [Public lineage]({receipt['glaasUrl']})\n"
                        f"- Job: `{state['jobId']}`; compute cost: ${state['observedCostUsd']:.2f}.\n\n"
                        "Generated and pushed automatically by the pinned public-canary supervisor.\n")
        actions.command(["gh", "pr", "create", "--repo", "reproducible-ai/notes", "--base", "main", "--head", branch,
                         "--draft", "--title", "docs(lda): record public one-step RoboCasa canary", "--body-file", str(body)], timeout=120)
        raise RuntimeError("Waiting for newly created notes PR to be readable")
    pr = prs[0]
    require(pr["state"] == "OPEN" and pr["headRefOid"] == commit, "Notes PR has not reflected the published commit")
    card = (root / "readback" / receipt["revision"] / "loader/README.md").read_text()
    card = card.replace("license_link: LICENSE", f"license_link: https://huggingface.co/{plan['hfRepository']}/blob/{receipt['revision']}/{PREFIX}/loader/LICENSE")
    card += (f"\n## Public verification\n\n- [Immutable checkpoint]({receipt['hfUrl']})\n"
             f"- [Public training lineage]({receipt['glaasUrl']})\n- [Run notes]({pr['url']})\n\n"
             "The host anonymously streamed every checkpoint byte and verified the published inventory and SHA-256 hashes. "
             "The worker's load, finite-tensor and changed-parameter checks are bound to the same checkpoint digest. "
             "Cold replay and held-out policy quality have not been tested. Roar reported untracked artifact directories.\n"
             f"<!-- public-canary:{plan['runId']} -->\n")
    published = HfApi(token=get_token()).upload_file(repo_id=plan["hfRepository"], path_in_repo="README.md", path_or_fileobj=card.encode(),
                                        commit_message="docs: link verified public LDA checkpoint and lineage")
    public_card = download_public(plan["hfRepository"], published.oid, "README.md", root / "public-model-card.md")
    require(public_card.read_text() == card, "Public root model card differs from the verified publication")
    return {"notesPr": pr["url"], "notesCommit": commit, "hfUrl": receipt["hfUrl"], "glaasUrl": receipt["glaasUrl"]}
