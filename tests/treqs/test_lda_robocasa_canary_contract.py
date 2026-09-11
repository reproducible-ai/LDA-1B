from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TREQS = ROOT / ".treqs"
SCRIPTS = TREQS / "scripts"
ASSETS = TREQS / "assets"
WORKFLOW_PATH = TREQS / "workflows" / "robocasa-demo-canary.yaml"
CONTRACT_PATH = SCRIPTS / "lda_canary_contract.py"


def load_contract():
    spec = importlib.util.spec_from_file_location("lda_canary_contract", CONTRACT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def test_inputs_and_private_publication_are_immutable():
    contract = load_contract()
    assert contract.SOURCE_BASE_COMMIT == "06e6a274a9086cc26635a9fe663866335eb30fc5"
    assert contract.BASE_MODEL_ID == "Wayer2/LDA-robocasa"
    assert contract.BASE_MODEL_REVISION == "811d14d8c22d3e98021c035948118143f53dd312"
    assert contract.QWEN_MODEL_ID == "Qwen/Qwen3-VL-4B-Instruct"
    assert contract.QWEN_MODEL_REVISION == "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    assert contract.DINO_MODEL_ID == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert contract.DINO_MODEL_REVISION == "114c1379950215c8b35dfcd4e90a5c251dde0d32"
    assert contract.DATASET_PATH == ROOT / "playground" / "demo_data" / "sim_pick_place"
    assert contract.EXPECTED_EPISODES == 4
    assert contract.PUBLICATION_REPO_ID in load_workflow()["publish"]["command"]
    assert contract.PUBLICATION_VERSION == "robocasa-demo-canary-0.0.1"


def test_build_backend_is_pinned():
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert 'requires = ["setuptools==80.9.0"]' in pyproject
    assert 'build-backend = "setuptools.build_meta"' in pyproject


def test_workflow_is_one_clean_lineage_dag():
    workflow = load_workflow()
    readme = (TREQS / "README.md").read_text()
    stages = ["fetch", "train", "evaluate", "package", "label", "publish"]
    assert workflow["secrets"] == ["HF_TOKEN"]
    assert workflow["setup"]["trace"] == "off"
    setup = workflow["setup"]["command"]
    assert setup.index("check_hf_access.py") < setup.index("pip install")
    assert "--with 'huggingface-hub==0.36.0'" in setup
    assert "--with huggingface-hub " not in setup
    assert any(line.strip().endswith(".venv/bin/python -m pytest -q tests/treqs") for line in setup.splitlines())
    for stage in stages:
        assert workflow[stage]["trace"] == ("run" if stage == "train" else "off")
    for stage in ("fetch", "evaluate", "package"):
        assert f"roar run -n {stage}" in workflow[stage]["command"]
    assert "supervisor-managed training trace" in readme
    assert "roar" not in workflow["train"]["command"]
    assert "Every workload stage is an explicit named `roar run`" not in readme
    assert "roar label set" in workflow["label"]["command"]
    assert workflow["publish"]["glaas_creds"] is True
    publish = workflow["publish"]["command"]
    assert "roar put" in publish
    import shlex
    upload = [line for line in publish.splitlines() if "roar put" in line]
    assert len(upload) == 1
    args = shlex.split(upload[0])
    assert args[-1].startswith("hf://")
    assert args[-1].endswith("/artifacts/lda-robocasa-canary/release/checkpoints")
    assert args.count("-m") == 1 and args[args.index("-m") + 1].strip()
    assert args[:2] == ["roar", "put"]
    assert args[2] == "artifacts/lda-robocasa-canary/release/checkpoints"
    assert args[3:-1] == ["--private", "--yes", "--no-tag", "-m", "private reproducibility canary"]
    assert "--anonymous" not in args
    assert "--private --yes --no-tag" in publish
    assert "--public" not in publish
    assert "hf upload" not in publish
    assert "huggingface-cli upload" not in publish


def test_workflow_hard_bounds_external_operations():
    workflow = load_workflow()
    hard_timeout = "timeout --signal=TERM --kill-after=30"
    commands = "\n".join(
        stage["command"] for stage in workflow.values() if isinstance(stage, dict) and "command" in stage
    )

    for line in commands.splitlines():
        if "timeout " in line:
            assert hard_timeout in line

    setup = workflow["setup"]["command"]
    assert f"{hard_timeout} 300 uv tool install" in setup
    assert f"{hard_timeout} 60 roar tracer use preload" in setup
    assert f"{hard_timeout} 60 roar tracer" in setup
    assert f"{hard_timeout} 60 roar init --no-gitignore" in setup
    assert "roar init || true" not in setup
    assert 'test "$(timeout' not in setup
    assert f"GPU_COUNT=\"$({hard_timeout} 30 nvidia-smi --list-gpus | wc -l | tr -d ' ')\"" in setup
    assert 'test "${GPU_COUNT}" = "1"' in setup
    assert f'ROAR_VERSION="$({hard_timeout} 60 env PATH=/usr/local/bin:/usr/bin:/bin roar --version)"' in setup
    assert 'test "${ROAR_VERSION}" = "roar, version 0.4.5"' in setup

    stage_timeouts = {
        "fetch": 5400,
        "train": 3600,
        "evaluate": 1800,
        "package": 600,
    }
    for stage, seconds in stage_timeouts.items():
        command = workflow[stage]["command"]
        prefix = "env" if stage == "train" else f"roar run -n {stage} -- env"
        assert f"{hard_timeout} {seconds} {prefix}" in command
        assert "roar run" not in command.split("timeout", 1)[0]

    assert f"{hard_timeout} 180 roar label set" in workflow["label"]["command"]
    publish_lines = workflow["publish"]["command"].splitlines()
    assert any(f"{hard_timeout} 180 roar status --untracked-dirs" in line for line in publish_lines)
    assert sum(line.startswith("roar put artifacts/lda-robocasa-canary/release/checkpoints ") for line in publish_lines) == 1


def test_workflow_stage_commands_are_valid_bash():
    workflow = load_workflow()

    for stage, config in workflow.items():
        if not isinstance(config, dict) or "command" not in config:
            continue
        result = subprocess.run(
            ["bash", "-n"],
            input=config["command"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{stage}: {result.stderr}"


def test_runtime_state_and_outputs_do_not_dirty_source_tree():
    for path in (".roar/probe", "artifacts/lda-robocasa-canary/probe"):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", "--", path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, f"workflow-generated path is not ignored: {path}"


def test_training_is_bounded_to_one_step_on_one_gpu():
    workflow = load_workflow()
    train = workflow["train"]["command"]
    assert "run_robocasa_canary.py" in train
    runner = (SCRIPTS / "run_robocasa_canary.py").read_text()
    assert '"--num_processes", "1"' in runner
    assert '"--trainer.max_train_steps", "1"' in runner
    assert '"--trainer.save_interval", "1"' in runner
    assert '"--datasets.vla_data.data_mix", "demo_data"' in runner
    assert '"--datasets.vla_data.per_device_batch_size", "4"' in runner
    assert '"--datasets.vla_data.training_tasks", \'["policy"]\'' in runner
    assert '"--datasets.vla_data.training_task_weights", "[1.0]"' in runner
    assert '"--framework.action_model.only_policy", "true"' in runner
    assert '"--framework.action_model.policy_and_video_gen", "false"' in runner
    assert '"--framework.action_model.only_wo_video_gen", "false"' in runner
    dataloader = (ROOT / "lda" / "dataloader" / "__init__.py").read_text()
    assert 'training_tasks = cfg.datasets.vla_data.get("training_tasks", TRAINING_TASKS)' in dataloader
    assert "tasks=training_tasks" in dataloader
    assert "Unsupported training tasks" in dataloader
    assert "must match training_tasks" in dataloader
    assert '"WANDB_MODE": "disabled"' in runner
    assert "action_model.vision_encoder,qwen_vl_interface" in runner


def test_preparation_hashes_the_repo_demo_and_pins_hub_downloads():
    prepare = (SCRIPTS / "prepare_robocasa_canary.py").read_text()
    assert "revision=BASE_MODEL_REVISION" in prepare
    assert "revision=QWEN_MODEL_REVISION" in prepare
    assert "DINO_MODEL_REVISION" in prepare
    assert "EXPECTED_EPISODES" in prepare
    assert "sha256_file" in prepare
    assert "input-manifest.json" in prepare


def test_component_licenses_are_pinned_packaged_and_required_for_publication():
    contract = load_contract()
    prepare = (SCRIPTS / "prepare_robocasa_canary.py").read_text()
    package = (SCRIPTS / "package_robocasa_canary.py").read_text()
    workflow = load_workflow()
    card = (ASSETS / "robocasa-demo-canary-model-card.md").read_text()
    readme = (TREQS / "README.md").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert contract.DINO_LICENSE_NAME == "LICENSE.md"
    assert contract.DINO_LICENSE_SHA256 == ("25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e")
    assert contract.SOURCE_LICENSE_SHA256 == ("6f79f90412086c4bef35afa6257d5c85c5e16fb43e71be068deec028869feaf6")
    assert contract.APACHE_LICENSE_SHA256 == ("cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30")
    assert hashlib.sha256((ASSETS / "APACHE-2.0.txt").read_bytes()).hexdigest() == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )
    assert "allow_patterns=[DINO_LICENSE_NAME]" in prepare
    assert "DINO_LICENSE_SHA256" in prepare
    assert 'RELEASE_ROOT / "DINOv3-LICENSE.md"' in package
    assert 'RELEASE_ROOT / "APACHE-2.0.txt"' in package
    assert 'RELEASE_ROOT / "CC-BY-NC-4.0.md"' in package
    assert "SOURCE_LICENSE_SHA256" in package
    assert "APACHE_LICENSE_SHA256" in package
    assert '"licenses": [' in package

    publish = workflow["publish"]["command"]
    for filename in (
        "LICENSE",
        "CC-BY-NC-4.0.md",
        "APACHE-2.0.txt",
        "DINOv3-LICENSE.md",
        "NOTICE",
    ):
        assert filename in publish
    label = workflow["label"]["command"]
    assert "license.id=LicenseRef-LDA-Composite" in label
    assert "CC BY-NC 4.0; Apache 2.0; DINOv3 License" in label

    assert "license: other" in card
    assert "license_name: Component-specific licensing" in card
    assert "Built with DINOv3" in card
    assert "DINOv3-LICENSE.md" in card
    assert "APACHE-2.0.txt" in card
    assert "component-specific" in readme.lower()
    assert 'license = {file = "LICENSE"}' in pyproject
    assert "License :: OSI Approved :: MIT License" not in pyproject


def test_evaluation_proves_a_real_updated_checkpoint():
    verify = (SCRIPTS / "verify_robocasa_canary.py").read_text()
    trainer = (ROOT / "lda" / "training" / "train_LDA.py").read_text()
    assert "load_tensor_state_dict(path)" in verify
    assert "torch.load" not in verify
    assert "summary.jsonl" in verify
    assert "base_sha256" in verify
    assert "checkpoint_sha256" in verify
    assert "Fine-tuned checkpoint is byte-identical" in verify
    assert "global_step" in verify
    assert "verify_manifest_files" in verify
    assert 'manifest["files"]' in verify
    assert '"optimizer_steps_completed": self.completed_steps' in trainer
    assert '"trainable_parameters": trainable_parameters' in trainer
    assert 'summaries[-1].get("optimizer_steps_completed")' in verify
    assert 'summaries[-1].get("trainable_parameters")' in verify
    assert "changed_tensor not in trainable_parameters" in verify
    assert "torch.isfinite(after).all()" in verify
    assert "Non-finite floating tensor in trained checkpoint" in verify


def test_package_is_loader_compatible_and_documents_scope():
    package = (SCRIPTS / "package_robocasa_canary.py").read_text()
    assert "checkpoints" in package
    assert "config.yaml" in package
    assert "dataset_statistics.json" in package
    assert "publication.json" in package
    assert "vision_encoder_load_weights" in package
    assert "shutil.copytree" in package
    assert "DINO_SNAPSHOT" in package
    assert 'config["framework"]["qwenvl"]["revision"] = QWEN_MODEL_REVISION' in package
    assert 'config["framework"]["action_model"]["only_policy"] = True' in package
    assert 'config["framework"]["action_model"]["policy_and_video_gen"] = False' in package
    assert 'config["framework"]["action_model"]["only_wo_video_gen"] = False' in package
    assert 'config["datasets"]["vla_data"]["training_task_weights"] = [1.0]' in package
    assert "Packaged checkpoint hash differs from evaluation" in package
    card = (ASSETS / "robocasa-demo-canary-model-card.md").read_text()
    assert "license: other" in card
    assert "base_model: Wayer2/LDA-robocasa" in card
    assert "base_model_relation: finetune" in card
    assert "one optimizer step" in card.lower()
    assert "not a quality" in card.lower()
    assert "download qwen normally" not in card.lower()
    readme = " ".join((TREQS / "README.md").read_text().lower().split())
    assert "closed as not planned" in readme
    assert "does not authorize compute or publication" in readme


def test_hf_preflight_requires_existing_private_writable_repo():
    preflight = (SCRIPTS / "check_hf_access.py").read_text()
    assert "api/models/{repo_path}" in preflight
    assert "preupload/main" in preflight
    assert 'repo_info.get("private")' in preflight
    assert "must be private" in preflight
    assert "PUBLICATION_REPO_ID" in preflight


def test_dino_architecture_can_be_built_without_gated_weight_download():
    framework = (ROOT / "lda" / "model" / "framework" / "QwenMMDiT.py").read_text()
    assert "action_model.MMDiT_ActionHeader import" in framework
    action_head = (ROOT / "lda" / "model" / "modules" / "action_model" / "MMDiT_ActionHeader.py").read_text()
    runner = (SCRIPTS / "run_robocasa_canary.py").read_text()
    trainer = (ROOT / "lda" / "training" / "train_LDA.py").read_text()
    tools = (ROOT / "lda" / "training" / "trainer_utils" / "trainer_tools.py").read_text()
    prepare = (SCRIPTS / "prepare_robocasa_canary.py").read_text()
    preflight = (SCRIPTS / "check_hf_access.py").read_text()

    assert "vision_encoder_load_weights" in action_head
    assert "DINOv3ViTConfig.from_pretrained" in action_head
    assert '"--framework.action_model.vision_encoder_load_weights", "false"' in runner
    assert '"--trainer.strict_pretrained_checkpoint", "true"' in runner
    assert "strict_pretrained_checkpoint" in trainer
    assert "strict=strict" in tools
    assert "DINO_CONFIG" in prepare
    assert "DINO_PREPROCESSOR_CONFIG" in prepare
    assert "allow_patterns=[DINO_LICENSE_NAME]" in prepare
    assert "DINO_LICENSE_NAME" in preflight
    assert "DINO_MODEL_ID" in preflight
    assert 'allow_patterns=["*.safetensors", "*.bin", "*.pt"]' not in prepare


def test_qwen_wrapper_honors_configured_attention_backend():
    source = (ROOT / "lda" / "model" / "modules" / "vlm" / "QWen3.py").read_text()
    assert 'qwenvl_config.get("attn_implementation", "flash_attention_2")' in source
    assert "attn_implementation=attn_implementation" in source
    assert 'qwenvl_config.get("revision")' in source
    assert source.count("revision=revision") == 2


def test_deepspeed_config_offloads_optimizer_and_gathers_checkpoint():
    config = yaml.safe_load((ASSETS / "accelerate-zero2-cpu.yaml").read_text())
    assert config["distributed_type"] == "DEEPSPEED"
    assert config["num_processes"] == 1
    ds_path = ROOT / config["deepspeed_config"]["deepspeed_config_file"]
    ds = yaml.safe_load(ds_path.read_text())
    assert ds["zero_optimization"]["stage"] == 2
    assert ds["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert ds["train_micro_batch_size_per_gpu"] == "auto"


def test_accelerate_accepts_file_owned_bf16_configuration(monkeypatch):
    from accelerate import DeepSpeedPlugin

    config = yaml.safe_load((ASSETS / "accelerate-zero2-cpu.yaml").read_text())
    ds_config = config["deepspeed_config"]
    # Reproduce the launcher's config-field environment without launching workers.
    monkeypatch.setenv("ACCELERATE_CONFIG_DS_FIELDS", ",".join(config.keys() | ds_config.keys()))
    monkeypatch.setenv("ACCELERATE_DEEPSPEED_CONFIG_FILE", str(ROOT / ds_config["deepspeed_config_file"]))
    monkeypatch.setenv("ACCELERATE_DEEPSPEED_ZERO3_INIT", "false")
    plugin = DeepSpeedPlugin()
    assert plugin.deepspeed_config["bf16"]["enabled"] is True
    assert plugin.deepspeed_config["fp16"]["enabled"] is False
    assert plugin.deepspeed_config["gradient_accumulation_steps"] == 1
    # The former duplicate must reproduce the observed initialization failure.
    monkeypatch.setenv("ACCELERATE_CONFIG_DS_FIELDS", "mixed_precision")
    import pytest
    with pytest.raises(ValueError, match="mixed_precision"):
        DeepSpeedPlugin()


def test_dependency_free_candidate_contracts():
    import sys
    subprocess.run([sys.executable, str(SCRIPTS / "check_local_candidate.py")],
                   cwd=ROOT, check=True)
