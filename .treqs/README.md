# LDA RoboCasa public platform canary

## Local validation and current handoff

Run the full recipe tests without installing the Linux GPU requirements on macOS:

```bash
uv run --offline --no-project --with pytest==8.4.2 --with pyyaml==6.0.3 --with torch python -m pytest -q tests/treqs
```

For a dependency-free workspace check, run
`python3 .treqs/scripts/check_local_candidate.py`. This checks stage shell syntax,
the complete checkpoint-directory public upload, supervisor repository replacement, and actual receipt
generation using synthetic files, including markers, metric, and package hashes.
The package emits E2E_ARTIFACT and E2E_RESULT and preserves loader metadata and
component notices as physical files under checkpoints/loader, inventoried in the artifact manifest. These local checks
do not establish GPU execution or constitute the independent artifact audit.
Evaluation checkpoint paths are relative to the source repository. This preserves
exact agreement between the published result and the logged receipt when TReqs
redacts machine-local workspace prefixes.
See `iteration-1-notes.md` for checks run and current limitations.

This public platform canary fine-tunes the pinned `Wayer2/LDA-robocasa` checkpoint for exactly
one optimizer step on LDA's four-episode committed demo dataset. The purpose is
to verify TReqs orchestration, ROAR capture, GLaaS lineage, checkpoint
verification, and public Hugging Face publication. It is not a campaign
certification or model-quality claim. Campaign issue #5 was closed as not
planned because its non-commercial license fails the campaign gate. This branch
preserves the bounded recipe; the external launch plan records the separate public-run authorization.

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
6. `publish` receives GLaaS credentials and publishes publicly through `roar put`
   as the complete checkpoints directory, containing checkpoint, receipts, and loader resources. The destination is `reproducible-ai/lda-1b-robocasa`; preflight verifies that it is public and writable.
   `checkpoints/result.json` records `optimizerSteps`, and
   `checkpoints/artifact-manifest.json` hashes all other published files using checkpoint-directory-relative paths, including
   loader metadata and component notices. Independent audit remains the supervisor's task.

The public run has a $5 compute cap, an early stop at a conservative $3.50,
and a 45-minute queued-job deadline. Allocation cost includes startup and idle
shutdown. The target has automatic idle shutdown within 15 minutes. This is a
supervised stop policy, not a provider-enforced dollar limit; host/API availability
is required. A failed paid run does not launch a replacement.

The source, dependencies, demo adapter, optimizer schedule and pinned Roar source
remain those of the successful private issue #38 candidate. Earlier private
artifacts and audit results are retained as historical evidence. This public
canary does not claim a new independent-auditor verdict or cold certification.

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
Retain the original package evidence for independent verification. Paths in the
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

## Fully automated public run

`.treqs/scripts/public_lda_canary.py` adapts the tested public supervisor from
`reproducible-ai/Isaac-GR00T@c084d6eab090a2b5822db474d790a00dfe7fae5d`.
It verifies that source revision before importing it. The plan also pins this
LDA source, harness verification helpers, target/AMI, public HF repository, and
an isolated bound TReqs control checkout. It records the public-run authorization
and a frozen template of the prior local LDA notes.

Run the frozen host environment under launchd and `caffeinate -i`:

```bash
/path/to/host/python .treqs/scripts/public_lda_canary.py --plan /absolute/run/plan.json
```

The supervisor creates one request with `--lineage-mode public`, queues once,
monitors full allocation cost and shutdown, verifies all seven tasks, streams
all published files anonymously at an immutable HF revision, and matches their
sizes and SHA-256 hashes to the manifest and logged result. It separately checks
that the actual training output feeds PUT in the public GLaaS graph, including
the byte-identical copy made by packaging. It then generates and pushes the
notes record, creates a draft PR, and updates the HF model card with the links.

The 14.4 GB checkpoint is streamed without retaining another local copy. The
worker performs the full state-dict load, finite-floating-tensor checks and
changed-parameter test; the host verifies that the public bytes match that
checkpoint's digest. The host does not claim a second full model load.

`state.json`, `events.jsonl`, `download-progress.json`, `verification.json` and
publication receipts live beside the immutable plan. `complete` requires final
compute settlement, public verification, and a pushed/read-back notes PR.
Restarting resumes the same request/job; lost create/queue replies are reconciled
without another paid launch. Post-run publication errors retry without GPU work.
Do not edit the plan or remove state to retry training. Operational TReqs commands
belong in the separate control checkout.

Host automation tests live in `tests/automation`; they are separate from the
worker's `tests/treqs` recipe checks. Set `PUBLIC_CANARY_SUPERVISOR_SCRIPTS` to the
pinned supervisor's `.treqs/scripts` directory when running them.
