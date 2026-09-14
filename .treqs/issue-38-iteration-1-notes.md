# Issue 38, iteration 1

Source HEAD: `3308fc909418fdf7645b0ad097c654a7d9375d63`.
Local candidate preparation only; no compute, credentials, API calls, publication,
or independent artifact audit performed. No new checkpoint or quality claim.

## Findings

- `i38-relative-receipts` (observation): The source recipe already constructs both
  evaluation checkpoint paths relative to ROOT. The supplied task packet reports
  issue 37's two absolute paths were changed by log redaction; that remote cause
  was not independently investigated here. Added dependency-free checks of the
  actual evaluation path expressions and full synthetic result equality after
  workspace-prefix redaction, including nested evaluation and repeated writes.
  Existing inventory/hash and invalid-evaluation rejection checks remain intact.
  Evidence: `.treqs/scripts/check_local_candidate.py`,
  `.treqs/scripts/verify_robocasa_canary.py`, `.treqs/issue-38-local-check-results.txt`.
- `i38-publication-placeholder` (decision): The workflow retained a previous
  staging repository. Restored `reproducible-ai/harness-test-pending` for supervisor
  replacement. Complete checkpoint-directory upload, notices, single literal
  private upload command, and dynamic repository derivation remain unchanged.
  Evidence: `.treqs/workflows/robocasa-demo-canary.yaml` and local check results.
- `i38-budget-handoff` (decision): README now states this attempt's $5 cap including
  setup, retries, and shutdown, $5 prior finalized cost, and $10 remaining original
  approval. Dollar enforcement and publication duration remain supervisor duties;
  command timeouts alone do not guarantee the cap. Fresh lineage, readback, exact
  sidecar equality, independent auditor PASS, retained prior evidence, and no
  post-launch intervention for an unattended-success claim are required.
  Evidence: `.treqs/README.md` and supplied task packet.
- `i38-local-validation` (observation): Eight dependency-free check groups pass.
  Full pytest could not start: `/usr/local/bin/python3: No module named pytest`.
  Verified cause is missing pytest in the selected local interpreter. No dependency
  installation attempted. Full `tests/treqs` remains required in workflow setup,
  which installs pinned pytest. Native Roar build and GPU execution are untested
  locally; runtime checks use mocks, receipts use synthetic files. This is local
  contract validation, not an independent audit or optimizer-update evidence.

## Commands actually run

- Workspace discovery: `pwd`, `ls -a`, `rg --files` (including hidden `.treqs`
  paths and AGENTS.md search), `git status --short`, `git rev-parse HEAD`.
  No workspace AGENTS.md found. Read workflow, candidate checker, contract,
  packaging/evaluation scripts, runtime tests, installer/check script, and README
  using `cat`; read installer pin/build markers with `rg -n`.
- `python3 .treqs/scripts/check_local_candidate.py`: passed before changes.
- `python3 -m pytest -q tests/treqs`: failed before collection, missing pytest.
- `python3 .treqs/scripts/check_local_candidate.py > .treqs/issue-38-local-check-results.txt 2>&1`:
  passed after changes; eight PASS groups retained in the log.
- `git diff --check`: passed. `git diff --stat` and `git status --short` inspected.

## Parameters

No training parameter changes against published recipe 3308fc9. Preserved one
optimizer step, batch four, BF16 owned by the DeepSpeed file, GPU optimizer,
one 96 GB Blackwell, PyTorch 2.9.0/cu128, frozen Qwen/DINO, strict loading,
checkpoint-bound adapter, component revisions/licenses, and Roar source revision
61e5e98ca823a25c19522870cb81d3ce730c391d with 1200-second installation bound.
The only destination change restores the task packet's supervisor placeholder.
This remains a partial private non-commercial training-path canary.

Candidate is ready for a supervisor-managed bounded run; independent readback
and auditor PASS remain external requirements. Notes remain local and uncommitted.
