# LDA RoboCasa private platform canary

## Local validation and current handoff

Run the full recipe tests without installing the Linux GPU requirements on macOS:

```bash
uv run --offline --no-project --with pytest==8.4.2 --with pyyaml==6.0.3 --with torch python -m pytest -q tests/treqs
```

This cached environment passed all 30 tests on September 11. These tests do not
establish GPU execution or compliance with the current harness evidence contract.
The retained issue-32 candidate is a development checkpoint: its upload directory
and manifest/result fields still need to match the supervisor's candidateContract
packet. In particular, emit E2E_ARTIFACT and E2E_RESULT and preserve the complete
loader metadata and component notices inside the directory the supervisor verifies.
Add tests against those exact receipt fields before requesting compute.

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
- Compute target: `d557ec14-5941-4e42-9848-daff50e1ad9d` (4x NVIDIA L40S)
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
2. `train` performs one policy-task optimizer step on four GPUs with Qwen and
   DINO frozen;
3. `evaluate` verifies step 1, matching state-dict keys, distinct checkpoint
   hashes, and at least one changed `action_model.*` tensor;
4. `package` creates LDA's required config/statistics/checkpoints layout and
   includes licenses, model cards, manifests, and evaluation evidence;
5. `label` attaches model, version, license, description, and documentation
   metadata to the released checkpoint;
6. `publish` receives GLaaS credentials and publishes privately through `roar put`
   as one complete release directory upload. The supervisor replaces the placeholder
   repository in the workflow; preflight derives its repository from that destination.
   `checkpoints/result.json` records `optimizerSteps`, and
   `checkpoints/artifact-manifest.json` hashes all other release files, including
   loader metadata and component notices. Independent audit remains the supervisor's task.

The supervisor must enforce the $15 total budget before scheduling four L40S GPUs;
per-command timeouts alone do not establish a dollar cap.
The publication command uses a literal release-directory source and starts directly
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
