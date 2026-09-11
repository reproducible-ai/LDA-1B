# LDA RoboCasa private platform canary

## Local validation and current handoff

Run the full recipe tests without installing the Linux GPU requirements on macOS:

```bash
uv run --offline --no-project --with pytest==8.4.2 --with pyyaml==6.0.3 --with torch python -m pytest -q tests/treqs
```

For a dependency-free workspace check, run
`python3 .treqs/scripts/check_local_candidate.py`. This checks stage shell syntax,
the complete checkpoint-directory private upload, supervisor repository replacement, and actual receipt
generation using synthetic files, including markers, metric, and package hashes.
The package emits E2E_ARTIFACT and E2E_RESULT and preserves loader metadata and
component notices as physical files under checkpoints/loader, inventoried in the artifact manifest. These local checks
do not establish GPU execution or constitute the independent artifact audit.
See `iteration-1-notes.md` for checks run and current limitations.

This private platform canary fine-tunes the pinned `Wayer2/LDA-robocasa` checkpoint for exactly
one optimizer step on LDA's four-episode committed demo dataset. The purpose is
to verify TReqs orchestration, ROAR capture, GLaaS lineage, checkpoint
verification, and private Hugging Face publication. It is not a campaign
certification or model-quality claim. Campaign issue #5 was closed as not
planned because its non-commercial license fails the campaign gate. This branch
preserves private platform-canary code; it does not authorize compute or
publication.

## Immutable inputs

- Fork base commit: `06e6a274a9086cc26635a9fe663866335eb30fc5`
- LDA checkpoint: `Wayer2/LDA-robocasa@811d14d8c22d3e98021c035948118143f53dd312`
- Qwen VLM: `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17`
- DINO encoder architecture: `facebook/dinov3-vits16-pretrain-lvd1689m@114c1379950215c8b35dfcd4e90a5c251dde0d32`; its weights are supplied by the strict-loaded LDA checkpoint
- Dataset: `playground/demo_data/sim_pick_place` (four episodes, committed in source)
- Compute target: one 96 GB RTX PRO 6000 Blackwell GPU
- ROAR: `roar-cli==0.4.5`, `huggingface-hub==0.36.0`, preload tracer
- Build backend: `setuptools==80.9.0`
- Required target secret: `HF_TOKEN`

The Hugging Face token needs read access to the pinned public LDA and Qwen
inputs, the exact DINOv3 license at the pinned revision, and write access to the
`reproducible-ai` organization. DINOv3 is built from the vendored architecture
plus a deterministic four-register-token config; strict loading proves every
DINO tensor comes from the starting LDA checkpoint.

## Lineage DAG

The workflow uses a supervisor-managed training trace (`trace: run`) with a plain
training command. Fetch, evaluate, and package use explicit named `roar run` operations
with TReqs tracing off. Label and publish then use
ROAR's metadata and storage commands directly against those captured artifacts:

1. `fetch` downloads exact LDA/Qwen revisions and the exact pinned DINOv3
   license, generates the pinned DINO architecture config, and hashes all model
   and demo files;
2. `train` performs one policy-task optimizer step on one GPU (per-device batch four, accumulation one) with Qwen and
   DINO frozen;
3. `evaluate` verifies step 1, matching state-dict keys, distinct checkpoint
   hashes, and at least one changed `action_model.*` tensor;
4. `package` creates LDA's required config/statistics/checkpoints layout and
   includes licenses, model cards, manifests, and evaluation evidence;
5. `label` attaches model, version, license, description, and documentation
   metadata to the released checkpoint;
6. `publish` receives GLaaS credentials and publishes privately through `roar put`
   as the complete checkpoints directory, containing checkpoint, receipts, and loader resources. The supervisor replaces the placeholder
   repository in the workflow; preflight derives its repository from that destination.
   `checkpoints/result.json` records `optimizerSteps`, and
   `checkpoints/artifact-manifest.json` hashes all other published files using checkpoint-directory-relative paths, including
   loader metadata and component notices. Independent audit remains the supervisor's task.

The supervisor must enforce the $15 total budget before scheduling the single Blackwell GPU;
per-command timeouts alone do not establish a dollar cap.
The publication command uses one literal checkpoint-directory source and starts directly
with `roar put` so the supervisor can bind it. The supervisor must also bound the
publication stage duration; this command has no shell timeout wrapper.

The final DAG should be inspectable with:

```bash
roar reproduce --lineage --run
```

## Scope

This is a training-path canary, not an accuracy, convergence, or deployment
claim. Release licensing is component-specific: the source and demo dataset are
identified as CC BY-NC 4.0, Wayer2 and Qwen model metadata identify Apache 2.0,
and embedded DINOv3 materials remain under the DINOv3 License. The package
hash-verifies and includes all three notices before creating release output; it
does not present CC BY-NC 4.0 as superseding the other component terms. This
branch still does not authorize compute or publication.

The publication includes `checkpoints/loader/` with all notices, input hashes,
configuration, statistics, and upstream metadata as ordinary files. Every published
file except the two receipts has a SHA-256 and `sizeBytes` inventory entry.
To restore LDA's layout, copy `checkpoints/loader/` contents into a new release
root and place the published state dictionary under that root's `checkpoints/`.
Retain the original private package for independent verification. Paths in the
manifest are relative to the published checkpoint directory, without traversal.

Runtime pins are PyTorch 2.9.0+cu128, torchvision 0.24.0+cu128, and torchcodec
0.8.1. The requirements install uses uv's `--index-strategy unsafe-best-match`
to consider both PyPI and the PyTorch CUDA index for exact pinned versions.
The default first-index strategy stopped at the CUDA index's incomplete certifi
versions in the supplied failed-run log. See `remediation-1-notes.md` for the fix
and local validation limits. Runtime pins remain unchanged, including torchcodec
0.8.1. PyTorch resolves its compatible CUDA dependencies; the obsolete CUDA 12.4
pins have been removed. BF16 and DeepSpeed ZeRO-2 remain. The current
SIGKILL remediation disables CPU optimizer offload (device `none`) to reduce
host memory demand during optimizer preparation. The legacy `zero2-cpu`
filenames are retained. Host OOM is a hypothesis, not a verified cause; GPU
memory fit and an optimizer update still require the supervisor-run canary.
This hardware adaptation does not establish hardware equivalence or full reproduction.

### Explicit batch configuration for the Blackwell canary

The one-GPU canary uses batch four and one accumulation step. LDA's custom batch sampler leaves `DataLoader.batch_size` unset, so DeepSpeed must explicitly set both `train_micro_batch_size_per_gpu` and `train_batch_size` to 4. Leaving either contract to automatic inference caused campaign-queue #34 to stop after strict pretrained-checkpoint loading and before the first optimizer update. A regression test exercises a real PyTorch custom batch sampler and verifies the configured effective batch. This continuation preserves the input pins, frozen modules, BF16, and one-step schedule.
