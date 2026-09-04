"""Fail fast on Hugging Face read and namespace permissions."""
from __future__ import annotations

import base64
import json
import os
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from lda_canary_contract import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    PUBLICATION_REPO_ID,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
)


def request_json(url: str, token: str) -> dict:
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def check_file(repo_id: str, revision: str, filename: str, token: str) -> None:
    repo_path = quote(repo_id, safe="/")
    rev = quote(revision, safe="")
    file_path = quote(filename, safe="/")
    request = Request(
        f"https://huggingface.co/{repo_path}/resolve/{rev}/{file_path}",
        headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-0"},
    )
    with urlopen(request, timeout=60) as response:
        response.read(1)


def has_org_write_access(identity: dict, org_name: str) -> bool:
    for org in identity.get("orgs", []):
        if org.get("name") != org_name:
            continue
        role = org.get("roleInOrg") or org.get("role")
        return role in {"admin", "write"}
    return False


def check_private_writable_repo(repo_id: str, token: str) -> None:
    repo_path = quote(repo_id, safe="/")
    try:
        repo_info = request_json(f"https://huggingface.co/api/models/{repo_path}", token)
    except HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                f"Hugging Face publication repo {repo_id} does not exist; "
                "roar put only writes to pre-existing repositories"
            ) from exc
        raise
    if repo_info.get("private") is not True:
        raise RuntimeError(f"Hugging Face publication repo {repo_id} must be private")

    sample = b"permission-check"
    payload = json.dumps(
        {
            "files": [
                {
                    "path": ".hf-write-permission-check",
                    "sample": base64.b64encode(sample).decode("ascii"),
                    "size": len(sample),
                }
            ]
        }
    ).encode()
    request = Request(
        f"https://huggingface.co/api/models/{repo_path}/preupload/main",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(
            f"HF_TOKEN lacks write access to publication repo {repo_id} ({exc.code})"
        ) from exc
    files = result.get("files", [])
    if len(files) != 1 or files[0].get("uploadMode") not in {"regular", "lfs"}:
        raise RuntimeError("Hugging Face pre-upload check returned an invalid response")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    checks = (
        (BASE_MODEL_ID, BASE_MODEL_REVISION, "config.yaml"),
        (QWEN_MODEL_ID, QWEN_MODEL_REVISION, "config.json"),
    )
    for repo_id, revision, filename in checks:
        try:
            check_file(repo_id, revision, filename, token)
        except HTTPError as exc:
            raise RuntimeError(
                f"HF_TOKEN cannot read {repo_id}@{revision}:{filename} ({exc.code})"
            ) from exc

    identity = request_json("https://huggingface.co/api/whoami-v2", token)
    namespace = PUBLICATION_REPO_ID.split("/", 1)[0]
    if not has_org_write_access(identity, namespace):
        raise RuntimeError(f"HF_TOKEN lacks write access to the {namespace} organization")
    check_private_writable_repo(PUBLICATION_REPO_ID, token)

    print(
        "Hugging Face preflight passed for the pinned LDA and Qwen inputs "
        f"and private writable repo {PUBLICATION_REPO_ID}"
    )


if __name__ == "__main__":
    main()
