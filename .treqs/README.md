# LDA RoboCasa private platform canary

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
inputs and write access to the `reproducible-ai` organization. DINOv3 is built
from the vendored architecture plus a deterministic four-register-token config;
strict loading proves every DINO tensor comes from the starting LDA checkpoint.

## Lineage DAG

The four traced workload stages are explicit named `roar run` operations, while
TReqs tracing is off to avoid nested tracer graphs. Label and publish then use
ROAR's metadata and storage commands directly against those captured artifacts:

1. `fetch` downloads exact LDA/Qwen revisions, generates the pinned DINO
   architecture config, and hashes all model and demo files;
2. `train` performs one policy-task optimizer step on four GPUs with Qwen and
   DINO frozen;
3. `evaluate` verifies step 1, matching state-dict keys, distinct checkpoint
   hashes, and at least one changed `action_model.*` tensor;
4. `package` creates LDA's required config/statistics/checkpoints layout and
   includes licenses, model cards, manifests, and evaluation evidence;
5. `label` attaches model, version, license, description, and documentation
   metadata to the released checkpoint;
6. `publish` receives GLaaS credentials and publishes privately through `roar put`
   to `hf://reproducible-ai/harness-test-lda-robocasa-issue-5/robocasa-demo-canary-0.0.1`.

The final DAG should be inspectable with:

```bash
roar reproduce --lineage --run
```

## Scope

This is a training-path canary, not an accuracy, convergence, or deployment
claim. The source repository states CC BY-NC 4.0 for the work and dataset, so the
published canary uses the same non-commercial license even though some upstream
Hub metadata is less restrictive.
