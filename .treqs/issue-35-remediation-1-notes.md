# Issue 35 — SIGKILL remediation, local iteration 1

## Evidence and findings

- `issue35-sigkill-observed`: supplied externalObservationsPath document,
  job 9e20d416-65db-4ce4-bbcb-6cba05a3e561, reports successful setup/fetch,
  strict checkpoint load, BF16, micro/global batches four, then SIGKILL (-9)
  after parameter statistics and before any completed optimizer update.
  No kernel OOM or memory measurements were supplied. Cause remains null.
- `issue35-host-pressure-hypothesis`: host memory pressure during DeepSpeed
  preparation with CPU optimizer offload is plausible, not established.
  Source places accelerator preparation immediately after parameter statistics.
- `issue35-gpu-optimizer-decision`: disable optimizer offload (device none,
  remove pinned host memory) to reduce host demand. Keep ZeRO-2 and all
  validated runtime repairs. GPU memory fit remains unverified. Add before/after
  preparation log markers. Legacy config filenames remain for compatibility.
- `issue35-remediation-checks`: dependency-free local checks passed, including
  synthetic receipt equality, hashes, complete inventory, invalid receipts,
  shell syntax, supervisor destination replacement, mocked launch and runtime
  rejections. Training entrypoint AST parsing and git diff --check passed.
  Python lacks pytest; full tests remain in remote setup before input fetch.
  These checks are not an independent artifact audit or evidence of training.
- `issue35-budget-after-failure`: supplied finalized cost is $0.62 for this
  continuation, leaving $12.92 of its $13.54 cap. Including issue 34's $1.46,
  known combined spend is $2.08 of $15. Supervisor must enforce remaining cap.

## Parameter changes against retained/published recipe

This remediation changes only optimizer placement: CPU offload with pin_memory
true -> device none, pin_memory omitted. BF16, PyTorch 2.9.0/cu128, one 96 GB
Blackwell, accumulation one, batch four, strict loading, pinned inputs and
Qwen/DINO freezing remain unchanged.

Retained partial deviations documented in issue-35-iteration-1-notes.md:
steps 100000 -> 1; batch/device 16 -> 4; warmup 5000 -> 0; base/action LR
1e-5/1e-4 -> 1e-6/1e-6; diffusion repetitions 4 -> 1; save/eval/log intervals
5000/100/10 -> 1/1000/1; four committed demo episodes, policy only;
Flash Attention 2 -> SDPA. No full-run restoration or quality claim.

## Commands actually run

- pwd; rg --files and rg -n on local instructions, configs, scripts and tests;
  cat/sed on the same files; git status --short; ls -d .venv.
- python3 -m pytest --version: failed, No module named pytest.
- apply_patch: optimizer placement, config comment, README, preparation markers,
  updated checks and these notes.
- python3 .treqs/scripts/check_local_candidate.py >
  .treqs/issue-35-remediation-1-check-results.txt: passed.
- git diff --check: passed; cat of check results: confirmed six PASS lines.
- python3 stdin script using ast.parse on lda/training/train_LDA.py: passed.
- git diff --stat: inspected change inventory.

Evidence paths: .treqs/issue-35-remediation-1-check-results.txt,
.treqs/assets/deepspeed-zero2-cpu.json, lda/training/train_LDA.py,
tests/treqs/test_lda_robocasa_canary_contract.py,
.treqs/scripts/check_local_candidate.py, .treqs/workflows/robocasa-demo-canary.yaml.
External evidence above is a summary of the supplied packet, not a fresh query.
No APIs, credentials, compute, publication, or independent audit were used.
