# LDA-1B RoboCasa calibration

## Restricted operator checks

The harness operator has no network access. Run `bash .treqs/scripts/check_calibration_local.sh` from the frozen candidate checkout. It runs these commands using only the Python standard library:

```bash
python3 -S -c 'from scripts.calibration_robocasa import load_inputs; load_inputs(".treqs/calibration/plan.json", ".treqs/calibration/resolved-config.json"); print("PASS: actual runtime recipe and pinned input/launcher checks")'
python3 -S -m unittest tests.treqs.test_calibration_process
```

Retain check output only under the ignored `artifacts/operator-checks/` directory (including any temporary files). Do not write untracked diagnostics beside reviewed source/configuration.

Also inspect the prepared workflow and compare the plan SHA with the task packet. These are local contract/process checks, not the complete CPU suite or proof of GPU fit. After they pass, return the frozen candidate with a `treqs-run` request for the supervisor. The operator must not fetch packages or replace these commands with an unavailable system pytest.

## Supervisor and CI host checks

Before authorizing a source pin, the supervising maintainer runs `bash .treqs/scripts/check_calibration_cpu.sh`. This uses a pinned, cache-managed CPU environment and runs the complete `tests/treqs` suite. It requires package access on first use and belongs outside the restricted operator. The complete suite passed for the reviewed implementation. Worker setup independently runs all calibration pytest modules in the GPU environment before input preparation or training. The supervisor still validates the actual private workflow binding and the clean pinned source before allocating compute.

This optional workflow measures three **independent 100/200/400-update points**.
The ordinary trainer retains the **300,000-update cosine scheduler and 15,000 warmup updates**.
Every process loads the same pinned released RoboCasa checkpoint with a fresh optimizer and RNG.
The points do not resume one another.

## Projected recipe

- All 24 upstream GR1 RoboCasa task folders, 1,000 episodes each, pinned by HF revision.
- Eight A100 40 GB GPUs on one p4d.24xlarge; BF16, ZeRO-2, batch 10 per GPU, accumulation 1 (global batch 80).
- Qwen and the action model train; DINO stays frozen. Four training objectives and diffusion repetition four.
- SDPA attention is explicit in this scenario. It differs from the upstream launcher's default Flash Attention.
- Checkpoint every 10,000 updates: full model weights, matching the ordinary writer. Optimizer state is not saved.
- Evaluation every 1,000 updates: ordinary prediction on a training batch with 20 DDIM steps. This is not a held-out quality evaluation.

The starting weights are the released **post-trained** RoboCasa model. The estimate concerns continued fine-tuning with this recipe, not an original pretraining reproduction. The full schedule is 750 times the largest measured point; the estimate must retain that extrapolation limitation. Runtime and slope gates can withhold an estimate.

## Measurements and lifecycle

`resolved-config.json` contains the complete pinned runtime settings, plus hashes of the input specification and distributed launcher configurations. The driver rejects changes before importing the GPU trainer. Prepared input files are hashed and pre-read before every point, and each point uses empty compiler caches. All ranks synchronize at measured update boundaries; startup belongs to each independent point, while checkpoint/evaluation work is measured outside its training interval.

A normal checkpoint is measured after every point. Evaluation is measured after the first point only. The final point still performs the upstream additional final-model save; that one-time work belongs to finalization. Earlier model copies are removed after their timing receipts are durable; their raw timing and process logs remain. Only the final checkpoint is staged. The verifier loads every tensor, checks shapes/finiteness and optimizer-step metadata, verifies frozen DINO weights, and requires both Qwen and action-model changes.

Worker payloads leave cold setup, allocation shutdown and ancillary costs unknown. The harness derives a separate, audit-bound projection only after a matched single-job allocation terminates. It preserves the original worker files. The forecast uses an explicit allocation-hour allowance (all eight GPUs together), measured checkpoint/evaluation time, and a transfer allowance. Actual charges remain a separate ledger. Unproven inputs produce a null headline rather than a guessed cost.

The workload does not import Roar or orchestration APIs. The workflow provides tracing and private staging. Child stdout/stderr is retained locally and forwarded into workload logs; workflow path additions preserve injected Python bootstrap paths. The normal harness handles queue claims, budget monitoring, final reconciliation, independent audit and notes draft publication.

## Operating scope

Use only with a frozen issue, reviewed plan SHA, pinned model/harness commits and a matching target/policy. This campaign has a separate **$100 all-in LDA cap**, including setup, retries and shutdown; reserve margin for ancillary costs and stop compute before the cap. Do not launch retries against a still-active attempt or silently reduce batch size/objectives to make a point fit. Keep existing public model and lineage links in the notes baseline.

Local CPU tests validate timing, actual optimizer-loop early stopping, two-rank synchronization, process cleanup and real checkpoint readback. They do not prove eight-GPU memory fit or throughput; those require the bounded live workflow.
