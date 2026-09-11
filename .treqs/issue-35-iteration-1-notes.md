# Issue 35 continuation — iteration 1

Local preparation only; no credentials, services, training, publication, or
independent artifact audit used. No optimizer update or quality claim is made.

## Findings

- `issue35-publication-destination`: source inspection found a prior attempt's
  staging destination. Replaced it with harness-test-pending as specified in the
  current contract. Directory upload retains notices and loader metadata. Tests
  accept supervisor repository replacement. Verified cause: stale workflow literal.
- `issue35-runtime-preserved`: preserved torch 2.9.0/cu128, one 96 GB Blackwell,
  file-owned BF16 and explicit DeepSpeed micro/global batch sizes four. Added
  dependency-free assertions for both batch sizes and accumulation consistency.
  Existing pytest coverage exercises the custom sampler's absent batch_size.
- `issue35-local-checks`: dependency-free checks passed (see adjacent results).
  System Python lacks pytest; full pytest remains required in workflow setup.
  Synthetic receipt and mocked launch checks are not training or artifact audit.
- `issue35-runtime-pending`: memory fit and a real finite optimizer update remain
  unverified; cause null. The packet reports no prior completed optimizer update.
- `issue35-budget`: current packet supersedes prior budget notes: $1.46 spent,
  continuation NTE $13.54, total cap $15. Supervisor must enforce the dollar cap,
  bind the private destination and arrange independent package verification.
  Stage timeouts alone do not enforce the dollar cap.

## Parameters

No numerical training settings changed this iteration. Retained deviations from
the committed recipe: steps 100000 -> 1; batch/device 16 -> 4; warmup 5000 -> 0;
base/action LR 1e-5/1e-4 -> 1e-6/1e-6; diffusion repetitions 4 -> 1;
save/eval/log intervals 5000/100/10 -> 1/1000/1; freeze Qwen/DINO; four committed
demo episodes, policy only; Flash Attention 2 -> SDPA. Accumulation one, BF16,
ZeRO-2 CPU optimizer offload and strict checkpoint loading are retained.
Runtime starts from the pinned checkpoint config. No full-run restoration.

## Commands actually run

- Workspace inspection: pwd, rg --files (including hidden .treqs), ls -a,
  git status --short, cat on workflow, scripts, assets, tests and prior notes.
- apply_patch: workflow placeholder, local batch assertions, and these notes.
- python3 .treqs/scripts/check_local_candidate.py > .treqs/issue-35-local-check-results.txt
  — passed; cat of that results file confirmed all checks.
- python3 -m pytest --version — failed: No module named pytest.
- git diff --check — passed.
- git diff --stat and git status --short — final change inventory.

Evidence: .treqs/issue-35-local-check-results.txt,
.treqs/workflows/robocasa-demo-canary.yaml,
.treqs/scripts/check_local_candidate.py, .treqs/assets/deepspeed-zero2-cpu.json,
.treqs/assets/accelerate-zero2-cpu.yaml, requirements.txt,
.treqs/scripts/run_robocasa_canary.py, tests/treqs/test_lda_robocasa_canary_contract.py.
