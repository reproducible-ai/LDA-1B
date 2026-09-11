"""Launch the bounded single-GPU LDA fine-tuning canary."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import torch

from lda_canary_contract import (
    BASE_CHECKPOINT,
    BASE_SNAPSHOT,
    PRETRAINED_ROOT,
    QWEN_SNAPSHOT,
    ROOT,
    RUN_ID,
    RUN_ROOT,
)


def validate_runtime() -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one CUDA GPU; found {torch.cuda.device_count()}")
    if torch.version.cuda != "12.8" or str(torch.__version__).split("+")[0] != "2.9.0":
        raise RuntimeError("Expected PyTorch 2.9.0 with CUDA 12.8")
    for index in range(1):
        if "RTX PRO 6000" not in torch.cuda.get_device_properties(index).name or torch.cuda.get_device_capability(index) != (12, 0):
            raise RuntimeError("Expected RTX PRO 6000 Blackwell")
        memory_gib = torch.cuda.get_device_properties(index).total_memory / (1024**3)
        if memory_gib < 90:
            raise RuntimeError(f"GPU {index} has only {memory_gib:.1f} GiB; at least 90 GiB is required")


def main() -> None:
    validate_runtime()
    required = (
        BASE_CHECKPOINT,
        BASE_SNAPSHOT / "config.yaml",
        QWEN_SNAPSHOT / "config.json",
        PRETRAINED_ROOT / "dinov3-vits16-pretrain-lvd1689m" / "config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing prepared inputs: {missing}")

    command = [
        "accelerate",
        "launch",
        "--config_file",
        ".treqs/assets/accelerate-zero2-cpu.yaml",
        "--num_processes", "1",
        "lda/training/train_LDA.py",
        "--config_yaml", str(BASE_SNAPSHOT / "config.yaml"),
        "--framework.qwenvl.base_vlm", str(QWEN_SNAPSHOT),
        "--framework.qwenvl.attn_implementation", "sdpa",
        "--framework.action_model.vision_encoder_path", str(PRETRAINED_ROOT),
        "--framework.action_model.vision_encoder_load_weights", "false",
        "--framework.action_model.only_policy", "true",
        "--framework.action_model.policy_and_video_gen", "false",
        "--framework.action_model.only_wo_video_gen", "false",
        "--datasets.vla_data.data_root_dir", "playground/demo_data",
        "--datasets.vla_data.data_mix", "demo_data",
        "--datasets.vla_data.demo_canary_adapter", "true",
        "--datasets.vla_data.per_device_batch_size", "4",
        "--datasets.vla_data.training_tasks", '["policy"]',
        "--datasets.vla_data.training_task_weights", "[1.0]",
        "--trainer.freeze_modules", "action_model.vision_encoder,qwen_vl_interface",
        "--trainer.max_train_steps", "1",
        "--trainer.gradient_accumulation_steps", "1",
        "--trainer.save_interval", "1",
        "--trainer.eval_interval", "1000",
        "--trainer.logging_frequency", "1",
        "--trainer.repeated_diffusion_steps", "1",
        "--trainer.num_warmup_steps", "0",
        "--trainer.learning_rate.base", "1e-6",
        "--trainer.learning_rate.action_model", "1e-6",
        "--trainer.pretrained_checkpoint", str(BASE_CHECKPOINT),
        "--trainer.strict_pretrained_checkpoint", "true",
        "--run_root_dir", str(RUN_ROOT),
        "--run_id", RUN_ID,
        "--wandb_project", "lda-robocasa-canary",
        "--wandb_entity", "disabled",
        "--is_debug", "false",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": "/tmp/lda-robocasa-hf",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
            "PYTHONUNBUFFERED": "1",
        }
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
