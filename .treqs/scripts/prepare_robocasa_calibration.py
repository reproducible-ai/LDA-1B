"""Download the pinned full 24-task RoboCasa subset and model inputs."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from scripts.calibration_robocasa import encoded, load_inputs, sha256_file

ROOT = Path("artifacts/robocasa-calibration")
INPUTS = ROOT / "inputs"


def retry_hub_rate_limit(operation):
    """Respect Hub reset headers, with finite retries inside the workflow cap."""
    from huggingface_hub.utils import HfHubHTTPError

    for attempt in range(7):
        try:
            return operation()
        except HfHubHTTPError as exc:
            response = exc.response
            cause = exc.__cause__
            # Hub 0.36 wraps a failed metadata HEAD (including HTTP 429) in
            # LocalEntryNotFoundError, preserving the HTTP error as its cause.
            if response is None and isinstance(cause, HfHubHTTPError):
                response = cause.response
            if response is None or response.status_code != 429 or attempt == 6:
                raise
            retry_after = response.headers.get("Retry-After", "")
            reset = re.search(r'(?:^|;)\s*t=(\d+)', response.headers.get("RateLimit", ""))
            delay = int(retry_after) + 1 if retry_after.isdigit() else int(reset[1]) + 1 if reset else 310
            if delay > 600:
                raise
            print(f"Hub rate limit: waiting {delay}s before retry {attempt + 1}/6", flush=True)
            time.sleep(delay)


def download_dataset(spec, destination, token):
    """List pinned task subtrees directly, avoiding a scan of unrelated data."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    destination = Path(destination)
    api = HfApi(token=token)
    common = {
        "repo_id": spec["datasetRepository"], "revision": spec["datasetRevision"],
        "repo_type": "dataset",
    }

    def download(filename):
        return retry_hub_rate_limit(
            lambda: hf_hub_download(**common, filename=filename, local_dir=destination, token=token)
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        for index, folder in enumerate(spec["datasetFolders"], 1):
            print(f"Listing pinned dataset task {index}/{len(spec['datasetFolders'])}: {folder}", flush=True)
            files = retry_hub_rate_limit(lambda: [
                entry.path for entry in api.list_repo_tree(**common, path_in_repo=folder, recursive=True)
                if isinstance(entry, RepoFile)
            ])
            if not files or any(not filename.startswith(folder + "/") for filename in files):
                raise ValueError("missing or unexpected dataset task files: " + folder)
            for _ in pool.map(download, files):
                pass
            print(f"Downloaded task {index}/{len(spec['datasetFolders'])}: {len(files)} files", flush=True)
    return destination


def storage_inventory(block_root=Path("/sys/block")):
    volumes = []
    for device in sorted(block_root.glob("nvme*n1")):
        model = (device / "device/model").read_text().strip()
        if model != "Amazon Elastic Block Store":
            continue
        size = int((device / "size").read_text()) * 512
        if size <= 0:
            raise RuntimeError("invalid attached EBS capacity")
        volumes.append({"device": device.name, "model": model, "sizeBytes": size})
    if not volumes or sum(item["sizeBytes"] for item in volumes) > 1024**4:
        raise RuntimeError("allocation exceeds the reviewed 1 TiB EBS allowance or lacks evidence")
    return volumes


def validate_dataset(root, folders, expected_episodes):
    import pyarrow.parquet as pq

    from lda.dataloader.gr00t_lerobot.schema import DatasetStatisticalValues

    for folder in folders:
        task = root / folder
        info = json.loads((task / "meta/info.json").read_bytes())
        episodes = sorted(task.glob("data/chunk-*/episode_*.parquet"))
        if len(episodes) != expected_episodes or info["total_episodes"] != expected_episodes:
            raise ValueError("incomplete RoboCasa task dataset: " + folder)
        # Prevent the loader from silently recomputing/mutating input statistics.
        stats = json.loads((task / "meta/stats.json").read_bytes())
        for entry in stats.values():
            DatasetStatisticalValues.model_validate(entry)
        modality = json.loads((task / "meta/modality.json").read_bytes())
        if not {"state", "action", "video", "annotation"} <= set(modality):
            raise ValueError("dataset modality metadata is incomplete")
        for key in ("left_arm", "right_arm", "left_hand", "right_hand", "waist"):
            if key not in modality["state"] or key not in modality["action"]:
                raise ValueError("dataset is not the reviewed GR1 embodiment")
        # Decode a real record before starting a distributed training process.
        table = pq.read_table(episodes[0])
        if table.num_rows <= 16:
            raise ValueError("dataset episode is too short for the future observation")
        if len(list(task.glob("videos/chunk-*/*/episode_*.mp4"))) != expected_episodes:
            raise ValueError("incomplete RoboCasa task videos")


def main():
    import torch
    from huggingface_hub import snapshot_download

    started = time.time()
    plan, _, _ = load_inputs(".treqs/calibration/plan.json", ".treqs/calibration/resolved-config.json")
    spec = json.loads(Path(".treqs/calibration/input-spec.json").read_bytes())
    expected = plan["recipe"]["hardware"]
    if torch.cuda.device_count() != expected["gpuCount"] or any(
        torch.cuda.get_device_name(i) != expected["accelerator"]
        or torch.cuda.get_device_properties(i).total_memory < 39 * 1024**3
        for i in range(torch.cuda.device_count())
    ):
        raise RuntimeError("calibration requires the reviewed eight A100 40 GB GPUs")
    if any(path.is_file() and path.name != ".gitkeep" for path in INPUTS.rglob("*")):
        raise RuntimeError("refusing previously prepared calibration inputs")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    volumes = storage_inventory()
    INPUTS.mkdir(parents=True, exist_ok=True)

    def download(repo, revision, destination, **kwargs):
        return Path(
            retry_hub_rate_limit(lambda: snapshot_download(
                repo, revision=revision, token=token, local_dir=destination, max_workers=16, **kwargs
            ))
        )

    base = download(
        spec["BASE_MODEL_ID"],
        spec["BASE_MODEL_REVISION"],
        INPUTS / "base",
        allow_patterns=["README.md", "config.yaml", "dataset_statistics.json", spec["BASE_CHECKPOINT_NAME"]],
    )
    if sha256_file(base / "config.yaml") != spec["BASE_CONFIG_SHA256"]:
        raise ValueError("base configuration pin changed")
    qwen = download(spec["QWEN_MODEL_ID"], spec["QWEN_MODEL_REVISION"], INPUTS / "qwen")
    dino = download(
        spec["DINO_MODEL_ID"],
        spec["DINO_MODEL_REVISION"],
        INPUTS / "pretrained/dinov3-vits16-pretrain-lvd1689m",
        allow_patterns=[spec["DINO_LICENSE_NAME"]],
    )
    if sha256_file(dino / spec["DINO_LICENSE_NAME"]) != spec["DINO_LICENSE_SHA256"]:
        raise ValueError("DINO license pin changed")
    (dino / "config.json").write_bytes(encoded(spec["DINO_CONFIG"]))
    (dino / "preprocessor_config.json").write_bytes(encoded(spec["DINO_PREPROCESSOR_CONFIG"]))
    dataset = download_dataset(spec, INPUTS / "dataset", token)
    validate_dataset(dataset, spec["datasetFolders"], spec["episodesPerFolder"])
    files = []
    for label, directory in [("base", base), ("backbone", qwen), ("vision", dino), ("dataset", dataset)]:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                files.append(
                    {"input": label, "path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_file(path)}
                )
    manifest = {
        "schema": "lda.calibration-inputs/v1",
        "startedUnixSeconds": started,
        "finishedUnixSeconds": time.time(),
        "files": files,
        "episodes": len(spec["datasetFolders"]) * spec["episodesPerFolder"],
        "planSha256": sha256_file(".treqs/calibration/plan.json"),
        "hardware": expected,
        "attachedEbsVolumes": volumes,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "dinoWeightsSource": "embedded in strict-loaded released LDA checkpoint",
    }
    (ROOT / "input-manifest.json").write_bytes(encoded(manifest))
    print(f"Prepared {len(files)} pinned files and {manifest['episodes']} episodes", flush=True)


if __name__ == "__main__":
    main()
