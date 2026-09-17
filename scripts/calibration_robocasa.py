"""Ordinary, isolated LDA RoboCasa calibration points using a committed resolved recipe.

This program knows nothing about orchestration or lineage services. The caller
supplies immutable input files, and collects the raw timing and checkpoint outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recipe_digest(recipe):
    return hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_inputs(plan_path, config_path):
    plan = json.loads(Path(plan_path).read_bytes())
    data = Path(config_path).read_bytes()
    config = json.loads(data)
    if hashlib.sha256(data).hexdigest() != plan["recipe"]["configSha256"]:
        raise ValueError("resolved configuration differs from the pinned plan")
    recipe, protocol = plan["recipe"], plan["protocol"]
    training, runtime = config["trainer"], config["calibration_runtime"]
    if (
        training["max_train_steps"] != recipe["fullSteps"]
        or config["datasets"]["vla_data"]["per_device_batch_size"] != recipe["perDeviceBatchSize"]
        or training["gradient_accumulation_steps"] != recipe["gradientAccumulationSteps"]
        or recipe["gradientAccumulationSteps"] != 1
        or runtime["world_size"] != recipe["hardware"]["gpuCount"]
        or config["seed"] != protocol["seed"]
        or training["is_resume"]
        or config.get("calibration")
        or not training["strict_pretrained_checkpoint"]
        or config["framework"]["action_model"]["vision_encoder_load_weights"]
        or training["lr_scheduler_type"] != recipe["scheduler"]["name"]
        or training["num_warmup_steps"] != recipe["scheduler"]["warmupSteps"]
        or training["save_interval"] != recipe["checkpoint"]["everySteps"]
        or training["eval_interval"] != recipe["evaluation"]["everySteps"]
        or recipe["precision"] != "bf16"
        or training["freeze_modules"] != "action_model.vision_encoder"
        or set(recipe["trainableModules"]) != {"qwen_vl_interface", "action_model except vision_encoder"}
    ):
        raise ValueError("runtime configuration disagrees with the calibration recipe")
    for filename, key in (
        ("input-spec.json", "inputSpecSha256"),
        ("deepspeed.json", "deepspeedSha256"),
        ("accelerate.yaml", "accelerateSha256"),
    ):
        if sha256_file(Path(config_path).parent / filename) != runtime[key]:
            raise ValueError("pinned input or launcher configuration changed")
    inputs = json.loads((Path(config_path).parent / "input-spec.json").read_bytes())
    if (
        recipe["baseModel"]["revision"] != inputs["BASE_MODEL_REVISION"]
        or recipe["baseModel"]["repository"] != "https://huggingface.co/" + inputs["BASE_MODEL_ID"]
        or recipe["dataset"]["revision"] != inputs["datasetRevision"]
        or recipe["dataset"]["repository"] != "https://huggingface.co/datasets/" + inputs["datasetRepository"]
    ):
        raise ValueError("pinned model or dataset disagrees with the recipe")
    ds = json.loads((Path(config_path).parent / "deepspeed.json").read_bytes())
    if (
        ds["train_micro_batch_size_per_gpu"] != recipe["perDeviceBatchSize"]
        or ds["train_batch_size"] != recipe["perDeviceBatchSize"] * runtime["world_size"]
        or ds["gradient_accumulation_steps"] != 1
        or not ds["bf16"]["enabled"]
    ):
        raise ValueError("DeepSpeed batch or precision differs from the recipe")
    last = max(point["steps"] for point in protocol["points"])
    if last >= min(training["save_interval"], training["eval_interval"]):
        raise ValueError("points may not cross periodic checkpoint or evaluation boundaries")
    return plan, config, data


def run_child(command, *, output, timeout, env=None):
    """Stop the entire process group on timeout; retain output and propagate failure."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output.open("x") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            text=True,
            errors="replace",
            bufsize=1,
        )

        def copy_output():
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)

        pump = threading.Thread(target=copy_output, daemon=True)
        pump.start()
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()
            raise
        finally:
            pump.join(timeout=5)
            if pump.is_alive():
                # A descendant holding the pipe must not outlive a point.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                pump.join(timeout=5)
                if pump.is_alive():
                    raise RuntimeError("child log pipe did not close")
            process.stdout.close()
    if code:
        raise subprocess.CalledProcessError(code, command)
    return started, time.monotonic()


def run_point(args):
    import torch

    plan, config, _ = load_inputs(args.plan, args.config)
    point = next(point for point in plan["protocol"]["points"] if point["id"] == args.point)
    hardware = plan["recipe"]["hardware"]
    if torch.cuda.device_count() != hardware["gpuCount"] or any(
        torch.cuda.get_device_name(i) != hardware["accelerator"] for i in range(torch.cuda.device_count())
    ):
        raise RuntimeError("runtime accelerators differ from the pinned plan")
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(args.config.parent / "accelerate.yaml"),
        "lda/training/train_LDA.py",
        "--config_yaml",
        str(args.config),
        f"--run_root_dir={args.output}",
        f"--run_id={point['id']}",
        f"--calibration.stop_steps={point['steps']}",
        f"--calibration.world_size={hardware['gpuCount']}",
        f"--calibration.exclude_first_updates={config['calibration_runtime']['exclude_first_updates']}",
        f"--calibration.timing_path={args.output / (point['id'] + '.json')}",
        "--calibration.measure_evaluation=" + str(point["id"] == plan["protocol"]["points"][0]["id"]).lower(),
    ]
    # Keep the process-group identity so parent timeouts kill all eight ranks.
    os.execv(sys.executable, command)


def record_event(journal, event):
    """Persist diagnostics locally and in captured stdout before proceeding."""
    line = json.dumps(event, allow_nan=False)
    journal.write(line + "\n")
    journal.flush()
    os.fsync(journal.fileno())
    print("CALIBRATION_EVENT=" + line, flush=True)


def run_points(args):
    plan, _, config_bytes = load_inputs(args.plan, args.config)
    if any(p.is_file() and p.name != ".gitkeep" for p in args.output.rglob("*")):
        raise ValueError("refusing to reuse a calibration output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "resolved-config.json").write_bytes(config_bytes)
    journal_path = args.output / "processes.jsonl"
    with journal_path.open("x") as journal:
        for point in plan["protocol"]["points"]:
            # Make the declared cache policy repeatable: warm source files, but give
            # each fresh training process its own initially empty compiler caches.
            inputs = json.loads(Path("artifacts/robocasa-calibration/input-manifest.json").read_bytes())
            for entry in inputs["files"]:
                if sha256_file(entry["path"]) != entry["sha256"]:
                    raise ValueError("calibration input bytes changed between independent points")
            command = [
                sys.executable,
                "-m",
                "scripts.calibration_robocasa",
                "point",
                "--plan",
                str(args.plan),
                "--config",
                str(args.config),
                "--output",
                str(args.output),
                "--point",
                point["id"],
            ]
            attempt = {
                "pointId": point["id"],
                "requestedSteps": point["steps"],
                "startedUnixSeconds": time.time(),
                "command": command,
            }
            record_event(journal, {"event": "point-start", **attempt})
            try:
                started, exited = run_child(
                    command,
                    output=args.output / f"{point['id']}.log",
                    timeout=plan["protocol"]["maxPointSeconds"],
                    env=dict(
                        os.environ,
                        PYTHONUNBUFFERED="1",
                        WANDB_MODE="disabled",
                        HF_HUB_OFFLINE="1",
                        TRANSFORMERS_OFFLINE="1",
                        TRITON_CACHE_DIR=str((args.output / f"{point['id']}-triton").resolve()),
                        CUDA_CACHE_PATH=str((args.output / f"{point['id']}-cuda").resolve()),
                    ),
                )
                result = json.loads((args.output / f"{point['id']}.json").read_bytes())
                if result["completedSteps"] != point["steps"] or result["fullSteps"] != plan["recipe"]["fullSteps"]:
                    raise ValueError("child result disagrees with requested point or full schedule")
                finished = result["trainingFinishedMonotonicSeconds"]
                if not started < finished < exited:
                    raise ValueError("child monotonic training interval is outside its process")
                record_event(
                    journal,
                    {
                        "event": "point-complete",
                        **attempt,
                        "startedMonotonicSeconds": started,
                        "processExitedMonotonicSeconds": exited,
                        "finishedUnixSeconds": time.time(),
                        "timing": result,
                    },
                )
            except BaseException as exc:
                record_event(
                    journal,
                    {
                        "event": "point-failed",
                        **attempt,
                        "finishedUnixSeconds": time.time(),
                        "failureType": type(exc).__name__,
                    },
                )
                raise
            if point["id"] != plan["protocol"]["finalPointId"]:
                import shutil

                for directory in ("checkpoints", "final_model"):
                    shutil.rmtree(args.output / point["id"] / directory)
            print(
                f"Completed independent calibration point {point['id']}: {point['steps']} updates",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "point"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point")
    args = parser.parse_args()
    if args.action == "point":
        if not args.point:
            parser.error("point requires --point")
        run_point(args)
    else:
        # Workflow timeout must unwind run_child and stop its GPU process group.
        def interrupted(signum, frame):
            raise InterruptedError(f"calibration interrupted by signal {signum}")

        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            run_points(args)
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
