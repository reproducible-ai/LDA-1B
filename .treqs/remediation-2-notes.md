# Remediation 2: Accelerate configuration ownership

- `accelerate-precision-conflict` (verified cause, P1): the supplied
  externalObservationsPath executionLogs.tail for job
  9f8ea77a-413c-49d3-9733-2de2f07437fe shows setup and fetch succeeded, then
  DeepSpeedPlugin initialization rejected ignored Accelerate config fields.
  The workspace launcher supplied mixed_precision: bf16 alongside
  deepspeed_config_file. The JSON already enables BF16 and disables FP16.
  Removed the duplicate launcher field; no numerical training setting changed.
- `accelerate-regression-coverage` (decision): dependency-free checks reject all
  conflicting file-owned fields. Added a real DeepSpeedPlugin initialization
  test, including the former duplicate as a negative case, to the existing
  remote setup test suite. It does not launch workers or train a model.
- `remediation-2-local-checks` (observation): local dependency-free checks passed,
  covering shell syntax, private directory upload and destination replacement,
  synthetic receipt equality/inventory hashing, rejection gates, mocked runtime,
  batch four, frozen encoders, strict load and DeepSpeed ownership. Python syntax
  checks include the new test. pytest is unavailable locally; the plugin test
  and full pytest suite have not been executed here. These checks are not an
  independent artifact audit.
- `remediation-2-runtime-pending` (hypothesis, cause null): memory fit and a real
  optimizer update remain unverified. The supplied run never reached training.
  No checkpoint, quality, lineage or immutable readback claim is made.

Parameter change: inherited launcher mixed_precision=bf16 -> absent, with
DeepSpeed bf16.enabled=true retained. No full-run restoration. Published recipe
deviations remain in iteration-1-notes.md: 100000 -> 1 step, batch/device 16 -> 4,
warmup 5000 -> 0, base/action LR 1e-5/1e-4 -> 1e-6/1e-6, diffusion repeats 4 -> 1,
save 5000 -> 1, eval 100 -> 1000, logging 10 -> 1, Qwen/DINO frozen, four demos
policy only, SDPA. Retained the authorized one-GPU Blackwell CUDA 12.8 stack.

Commands actually run: pwd; rg --files and rg -n workspace searches; head;
git status --short; cat and sed reads of assets, scripts, tests and prior notes;
python3 -m pytest --version (failed: no pytest);
python3 .treqs/scripts/check_local_candidate.py > .treqs/remediation-2-check-results.txt
(passed); cat .treqs/remediation-2-check-results.txt; git diff --check (passed);
git diff --stat. Changes made with apply_patch. No commit created.

Evidence: .treqs/assets/accelerate-zero2-cpu.yaml,
.treqs/assets/deepspeed-zero2-cpu.json,
tests/treqs/test_lda_robocasa_canary_contract.py,
.treqs/remediation-2-check-results.txt and supplied externalObservationsPath.
Supplied cumulative cost is $1.01 of $15; supervisor must enforce remaining
campaign budget and arrange independent verification. No external calls,
credentials, compute, publication or audit were performed by this operator.
