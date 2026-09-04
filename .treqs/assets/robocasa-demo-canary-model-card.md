---
license: cc-by-nc-4.0
base_model: Wayer2/LDA-robocasa
base_model_relation: finetune
tags:
- robotics
- vision-language-action
- robocasa
- reproducible-ai
- treqs
- glaas-lineage
---

# LDA RoboCasa demo reproducibility canary

This is a one optimizer step fine-tuning canary for `Wayer2/LDA-robocasa`.
It demonstrates that pinned LDA source, model dependencies, and the four-episode
demo dataset can run through TReqs, produce a changed checkpoint, record a GLaaS
lineage DAG, and publish the verified release with `roar put`.

It is **not a quality, accuracy, convergence, or deployment-safety claim**.

## Reproducible inputs

- Source: `reproducible-ai/LDA-1B@{{SOURCE_COMMIT}}`
- Starting checkpoint: `Wayer2/LDA-robocasa@{{BASE_MODEL_REVISION}}`
- VLM: `Qwen/Qwen3-VL-4B-Instruct@{{QWEN_MODEL_REVISION}}`
- Vision encoder architecture: `facebook/dinov3-vits16-pretrain-lvd1689m@{{DINO_MODEL_REVISION}}`; weights come from the strict-loaded starting checkpoint
- Dataset: four committed `playground/demo_data/sim_pick_place` episodes
- Training: one optimizer step, global batch size 4, Qwen and DINO frozen
- Release path: `{{PUBLICATION_VERSION}}`

`evaluation.json` records both checkpoint SHA-256 digests, tensor metadata, and
the name of an `action_model.*` tensor that changed. `input-manifest.json`
records pinned revisions and hashes for the downloaded inputs and demo files.

## Loading layout

The release preserves LDA's required layout:

```text
config.yaml
dataset_statistics.json
checkpoints/LDA-robocasa-treqs-canary.pt
```

LDA resolves the Qwen dependency at the exact revision recorded above and in
`config.yaml`; loading must retain that revision rather than using the model's
mutable default branch. The DINO architecture config is bundled under
`pretrained/dinov3-vits16-pretrain-lvd1689m`; no separate DINO weight download is
needed because strict loading obtains every DINO tensor from this checkpoint.

## License

The LDA source repository states that the work and dataset are licensed under
CC BY-NC 4.0; this release therefore uses that stricter license. Upstream model
cards are included under `upstream/` for attribution. DINO provenance is recorded
in the manifest and publication metadata.
