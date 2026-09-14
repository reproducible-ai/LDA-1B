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
See `issue-37-iteration-1-notes.md` for current checks and handoff limitations.

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
- ROAR: `treqs/roar@61e5e98ca823a25c19522870cb81d3ce730c391d` (main on
  2026-09-14, package version 0.4.7), `huggingface-hub==0.36.0`, preload tracer
- Build backend: `setuptools==80.9.0`
- Required target secret: `HF_TOKEN`

The Hugging Face token needs read access to the pinned public LDA and Qwen
inputs, the exact DINOv3 license at the pinned revision, and write access to the
`reproducible-ai` organization. DINOv3 is built from the vendored architecture
plus a deterministic four-register-token config; strict loading proves every
DINO tensor comes from the starting LDA checkpoint.

### Roar source installation on existing compute targets

Setup runs `scripts/install_roar_source.sh` before model dependency downloads.
It fetches the exact main commit above and uses Roar's upstream development
installer to build the Python extension and native tracers in a separate Python
3.11 environment. A plain Git pip install does not build those tracer binaries.
The model environment remains separate. `/usr/local/bin/roar` and `roar-worker`
point at the verified install so later agent task shells select the same build.

The installer checks the actual imported source, clean tracked checkout, package
and Hugging Face versions, native binary hashes, and preload tracer preflight.
It records `ROAR_SOURCE_BUILD=<json>` in setup logs and retains `build.json`
under the user-base `share/reproai/roar/<commit>` directory. Matching successful
builds are reused; altered source or cached binaries fail setup. Cold builds are
bounded to 1,200 seconds. This adds native compilation time to a new instance.

This is a workflow setup override on the existing Blackwell targets, not a
change to their AMI or create-time TReqs bootstrap settings. Target administration
is unavailable to the current organization member. No target recreation or API
deployment is required. Updating the pin requires a reviewed source change;
the installer never follows a moving `main` during a run. After a PyPI release
includes this commit, the source build can be replaced with that exact wheel
version. Existing issue/attempt commits remain immutable and require a new
candidate pinned to this updated recipe before it takes effect.

For an infrastructure-only Linux container check with `uv` available, run:

```bash
bash tests/treqs/verify_roar_source_install.sh /tmp/roar-install-check
```

The directory must be new. The check builds the source, reuses the validated
cache, runs an ordinary Python file-copy command under Roar and checks its
recorded input/output edges, then verifies rejection of modified source and
active native binaries. It needs no GPU, model downloads, credentials or
publication. An optional second argument reuses an existing installer cache
while retaining a fresh directory for validation results.

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

For issue 37, the supervisor must enforce a $5 attempt cap including setup,
retries, and shutdown before scheduling the single Blackwell GPU. Prior finalized
cost is $3.71, leaving $11.29 of the original $15 approval; this attempt may use
at most $5 of that remainder.
per-command timeouts alone do not establish a dollar cap.
The publication command uses one literal checkpoint-directory source and starts directly
with `roar put` so the supervisor can bind it. The supervisor must also bound the
publication stage duration; this command has no shell timeout wrapper.

Require a fresh full published lineage containing train and evaluate, independent
checkpoint readback, and an independent auditor PASS. The task packet reports
that issue 36 completed one optimizer update but failed lineage verification;
its cause is not established by the local evidence. Retain that checkpoint and
all prior evidence. A supervising-agent intervention after launch disqualifies
an unattended-success claim. Notes stay local, without notes-repository commits,
pushes, or pull requests.

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
filenames are retained. The EC2 kernel log confirmed a host OOM in issue #35;
the GPU optimizer then initialized successfully. A completed optimizer update
still requires the supervisor-run canary.
This hardware adaptation does not establish hardware equivalence or full reproduction.

### Explicit batch configuration for the Blackwell canary

The one-GPU canary uses batch four and one accumulation step. LDA's custom batch sampler leaves `DataLoader.batch_size` unset, so DeepSpeed must explicitly set both `train_micro_batch_size_per_gpu` and `train_batch_size` to 4. Leaving either contract to automatic inference caused campaign-queue #34 to stop after strict pretrained-checkpoint loading and before the first optimizer update. A regression test exercises a real PyTorch custom batch sampler and verifies the configured effective batch. This continuation preserves the input pins, frozen modules, BF16, and one-step schedule.

### Pinned checkpoint and demo compatibility

`assets/robocasa-pinned-config.yaml` is the exact config from
`Wayer2/LDA-robocasa@811d14d8c22d3e98021c035948118143f53dd312`; input preparation
verifies its SHA-256. It defines state width 58, action width 138, 32 embodiment
slots, and two observation frames. The explicit demo adapter preserves all
12 source state values, zero-pads the remaining 46, retains loader-padded
actions and their mask, and maps the demo's `franka_robotiq` metadata to existing
Franka slot 4. It samples video/state history at [-5, 0] and future video at 16
through the existing dataset loader. Checkpoint shapes and strict loading are
unchanged. This establishes a training compatibility canary only; it does not
establish matching embodiment semantics, policy quality, or full reproduction.
