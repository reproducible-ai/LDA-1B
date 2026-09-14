# Issue 37 — local candidate iteration 1

Attempt: c7097037-aa89-4628-8838-f5b08b92998c. Source supplied by task packet:
e5ecbac8b63882777f3f321a5316b04507e585e5. No compute, credentials, service
APIs, publication, or independent audit were used in this iteration.

## Findings

- `i37-publication-destination` (verified): workflow retained a prior staging
  repository. Replaced it with the contract's harness-test-pending placeholder,
  retaining the complete directory source, destination prefix, and private flags.
  Repository replacement is exercised by the dependency-free check.
- `i37-budget-handoff` (verified): README referred only to the original $15 cap.
  Updated to $5 for this attempt including setup, retries and shutdown, with
  $3.71 prior finalized cost and $11.29 remaining as supplied by the task packet.
  Dollar enforcement belongs to the supervisor; shell timeouts are insufficient.
- `i37-prior-lineage` (reported, unresolved cause): task packet says issue 36
  trained successfully but failed lineage verification. No failure trace was
  provided locally to establish a cause. Preserve the reviewed Roar source pin
  and tracing structure; require fresh full published train/evaluate lineage,
  independent checkpoint readback and auditor PASS. Prior evidence and checkpoint
  must remain retained. No claim that this local change fixes the prior failure.
- `i37-local-validation` (verified): dependency-free checks pass after changes.
  Added exact reviewed Roar revision, installer time bound and named lineage stage
  checks. Checks cover shell/Python syntax, synthetic receipt generation, complete
  inventory hashes, repository replacement and mocked runtime/launch validation.
  These are candidate tests, not a model run or artifact audit.
- `i37-pytest-unavailable` (verified): `python3 -m pytest -q tests/treqs` failed
  before collection because this interpreter has no pytest. No workspace .venv
  exists. Full pytest remains required by workflow setup before fetch/train.

## Parameters

No training or input parameter changes against the task's published partial
recipe. Preserve one optimizer step, batch four, accumulation one, BF16 owned by
the DeepSpeed file, GPU optimizer, one 96 GB Blackwell, PyTorch 2.9.0/cu128,
frozen Qwen/DINO, strict checkpoint loading and all input revisions. Preserve
state 12→58 padding, action width 138 and mask, Franka slot 4, history [-5,0],
future video 16, and obs_horizon 2. This is a shape adaptation, not evidence of
matching embodiment semantics or model quality. Full upstream schedule remains
out of scope. No parameters were changed to restore a full run.

## Commands and evidence

- Local discovery/read commands: `pwd`, `rg --files`, `ls -a`, `git status --short`,
  and `cat`/`rg -n` on workflow, scripts, tests and existing notes. No AGENTS.md
  was found; initial tracked workspace was clean.
- `python3 .treqs/scripts/check_local_candidate.py > .treqs/issue-37-local-check-results.txt 2>&1`
  — passed before and after edits; final output retained at that path.
- `python3 -m pytest -q tests/treqs > .treqs/issue-37-pytest-results.txt 2>&1`
  — failed before collection; interpreter dependency diagnostic retained.
- `ls .venv/bin/python` — absent; no dependency environment was installed.

Supervisor handoff: bind a fresh private destination, enforce the $5 all-in cap,
run the complete workflow unattended, retain all prior artifacts, independently
read back and audit the published package. Intervention after launch prevents an
unattended-success claim. No remote artifact or auditor PASS is claimed locally.
