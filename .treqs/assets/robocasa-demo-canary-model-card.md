---
license: other
license_name: Component-specific licensing
license_link: LICENSE
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
- Training: one optimizer step, global batch size 4, BF16, one 96 GB RTX PRO 6000 Blackwell GPU, Qwen and DINO frozen
- Release path: `{{PUBLICATION_VERSION}}`

`evaluation.json` records both checkpoint SHA-256 digests, tensor metadata, and
the name of an `action_model.*` tensor that changed. `input-manifest.json`
records pinned revisions and hashes for the downloaded inputs and demo files.

## Loading layout

The private package includes loader resources under `checkpoints/loader/`.
Copy those resources to a new release root and place the published state dictionary
under its `checkpoints/` directory to restore LDA's required layout:

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

Licensing is component-specific; no single license supersedes every upstream
term. The LDA source and bundled demo dataset are identified as CC BY-NC 4.0 in
`CC-BY-NC-4.0.md`. The Wayer2 checkpoint and Qwen model metadata identify Apache
2.0; its terms are included as `APACHE-2.0.txt`. Embedded DINOv3 materials remain
subject to the DINOv3 License included as `DINOv3-LICENSE.md`.

Use and redistribution must comply with all applicable component terms. Upstream
model cards are included under `upstream/` for provenance and attribution.

**Built with DINOv3.**

The private demo training adapter preserves the 12 source state values and
right-pads them with 46 zeros for the checkpoint's 58-wide state encoder. It
routes demo NEW_EMBODIMENT ID 32 to existing Franka slot 4 (the committed data
identifies `franka_robotiq`). Actions retain the loader's 29-wide padding and
mask. This shape-only canary adaptation does not establish RoboCasa feature
semantics or policy quality and does not resize checkpoint tensors. It is
recorded as `datasets.vla_data.demo_canary_adapter: true` in the packaged config.
