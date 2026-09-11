# Issue 36 candidate, iteration 1

Local preparation only. No compute, credentials, external APIs, publication,
artifact audit or model-quality claim. No optimizer update is evidenced locally.

## Findings

- `i36-publication-destination` (verified issue): workflow named a prior staging
  repository. Replaced it with harness-test-pending for supervisor substitution.
  Preserved one literal private directory upload, destination last, including all
  loader resources and notices. Cause: inherited workflow literal; resolved.
- `i36-adapter-checks` (decision): dependency-free checker now executes the real
  adapter with distinct synthetic histories, verifying all 12 values survive,
  46 zeros are appended, actions and masks retain identity, slot 32 becomes 4,
  incompatible samples fail, and history/future indices remain [-5,0]/16.
  The pinned config hash and dimensions are checked. These are local regression
  checks, not proof that actual video decoding or GPU execution succeeds.
- `i36-local-dependencies` (verified limitation): direct adapter test cannot
  import yaml; full pytest cannot import pytest. No dependencies installed.
  Dependency-free checks pass; workflow setup retains full pytest before fetch.
- `i36-remote-update` (observation): no completed update is available in the
  supplied packet. Runtime memory fit and successful optimizer execution remain
  unverified; cause null. Supervisor must enforce continuation NTE $12.47,
  with $2.53 prior cost and $15 aggregate cap, and arrange independent audit.

Evidence: .treqs/issue-36-local-check-results.txt,
.treqs/workflows/robocasa-demo-canary.yaml,
.treqs/scripts/check_local_candidate.py, lda/utils/demo_canary_inputs.py,
.treqs/assets/robocasa-pinned-config.yaml, tests/treqs/test_demo_canary_inputs.py.

## Parameters

No training parameters changed this iteration. Relative to the hash-bound pinned
RoboCasa config, retained overrides are: steps 300000 -> 1; device batch 10 -> 4;
warmup 15000 -> 0; base/action LR 4e-5/1e-4 -> 1e-6/1e-6; repeated diffusion
steps 4 -> 1; save interval 5000 -> 1; logging 100 -> 1; DINO frozen -> Qwen and
DINO frozen; four tasks -> policy only; RoboCasa data -> four committed demo
episodes. Eval interval 1000 and accumulation 1 remain unchanged. SDPA replaces
the Qwen wrapper's default Flash Attention 2. Preserve torch 2.9.0/cu128, one
96 GB Blackwell, file-owned BF16, DeepSpeed micro/global batch 4 and GPU optimizer.
The demo adapter maps source state 12 -> 58 (right zero padding), existing demo
slot 32 -> Franka slot 4, video/state [0] -> [-5,0], future [5] -> [16]. Action
width 138 and checkpoint architecture remain unchanged; strict loading remains
enabled. Shape adaptation does not establish embodiment semantic compatibility.
None of these overrides restores the full published run.

## Commands actually run

Inspection: pwd; rg --files (including hidden .treqs); ls -a; git status --short;
cat of workflow, scripts, tests, pinned config and inherited notes; rg of adapter
and mask wiring. All succeeded; one large inspection output was truncated.
Edits used apply_patch.

- python3 .treqs/scripts/check_local_candidate.py: passed before and after edits.
  Final output saved in .treqs/issue-36-local-check-results.txt.
- python3 tests/treqs/test_demo_canary_inputs.py: failed, missing yaml.
- python3 -m pytest -q tests/treqs: failed, missing pytest.
- git diff --check: passed after code/workflow edits.

No commit created. Candidate is ready for the supervisor's bounded run; package
verification and independent audit remain external obligations.
