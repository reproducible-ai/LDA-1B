# Iteration 1 — local candidate handoff

No external service, credential, training workload, publication, or independent
artifact audit was used. No remote artifact or model quality is claimed.

## Findings

- `publication-placeholder`: The workflow contained a previous staging repository.
  Replaced it with the packet's harness-test-pending destination. Preserved the
  complete release-directory source, private flags, and one literal upload command.
  Cause verified by the workflow text; resolved locally.
- `receipt-markers`: The package wrote result and manifest files but did not emit
  E2E_ARTIFACT/E2E_RESULT. Added markers and extracted the real receipt writer for
  dependency-free fixture testing. It rejects optimizer counts other than one and
  excludes an existing manifest from its own records on repeat writes.
  Cause verified by source; resolved locally.
- `local-test-dependencies`: System Python lacks pytest. Offline resolution with
  a workspace-only UV cache cannot find pytest. The full pytest suite was not run.
  Dependency-free checks passed; workflow setup retains the full suite before fetch.
  This is a validation limitation, not evidence of a GPU runtime failure.
- `supervisor-gates`: The supervisor must enforce the $15 total budget, bind the
  private destination, run the candidate, and independently verify the package.
  Per-command timeouts are not a dollar cap. GPU compatibility and the optimizer
  update remain unverified locally; their causes are unresolved (null).

## Parameter scope

No training parameters changed this iteration. Existing canary overrides relative
to the committed `lda/config/training/LDA_robocasa.yaml` are: 100000 to 1 training
steps; per-device batch 16 to 1; warmup 5000 to 0; base/action learning rates
1e-5/1e-4 to 1e-6/1e-6; trainer diffusion repetitions 4 to 1; save interval
5000 to 1; evaluation interval 100 to 1000; logging interval 10 to 1; no frozen
modules to frozen Qwen and DINO; four task weights to policy only. SDPA replaces
Flash Attention 2. Four GPUs, BF16 and CPU optimizer offload implement the task's
bounded schedule. These are a training-path canary, not restoration of the full
published run. Runtime starts from the pinned checkpoint's config; uninspected
remote configuration values are not claimed to match the committed recipe.

## Commands and results

Local inspection used pwd, rg --files (including hidden .treqs files), ls -a,
git status --short, and cat on the workflow, README, local checker, contract,
packager, verifier, runner, runtime/contract tests, and committed training config.
`command -v python3`, `command -v uv`, and the local rg search for summary,
completed_steps, optimizer.step and bf16 succeeded. No network commands ran.

- `python3 -m pytest -q tests/treqs`: failed, No module named pytest.
- `UV_CACHE_DIR=.local-check-cache uv run --offline --no-project --with pytest==8.4.2 --with pyyaml==6.0.3 --with torch python -m pytest -q tests/treqs`:
  failed offline dependency resolution; pytest not cached.
- `python3 .treqs/scripts/check_local_candidate.py`: passed HF_TOKEN declaration,
  seven stage shell syntax checks, private upload contract, Python syntax,
  repository replacement, receipt markers, metric, complete package hashes,
  repeat writes, and invalid step rejection. All runtime files were synthetic.
- `git diff --check`: passed.
- `git status --short`: confirmed candidate edits only; no commit created.

Evidence: `.treqs/workflows/robocasa-demo-canary.yaml`,
`.treqs/scripts/check_local_candidate.py`, `.treqs/scripts/package_robocasa_canary.py`,
`tests/treqs/test_canary_runtime.py`. These are candidate checks, not self-audit.

## Contract-failure correction in the current iteration

- `publication-exact-files` supersedes `publication-placeholder`: supplied candidate
  validation rejected the directory upload because exactly three literal artifact
  files are required. Source inspection confirms both local checkers incorrectly
  required that directory. Workflow now names only the checkpoint, manifest, and
  result, with the harness-test-pending destination last. Both checkers require
  the exact sources and flags while accepting supervisor repository replacement.
- `supporting-file-preservation`: replacing the directory source alone would omit
  notices and loader resources. The manifest now embeds every supporting file's
  exact bytes as content_hex, with existing relative paths, sizes and hashes.
  Checkpoint and result remain external files; the checkpoint state dict is
  unchanged by packaging. README documents reconstruction and path validation.
  Synthetic fixture checks verify embedded bytes round-trip and that only the
  checkpoint and result omit embedded content. This is local candidate testing,
  not an independent artifact audit or proof of training.
- `current-validation-limit`: python3 -m pytest -q tests/treqs still fails with
  No module named pytest. No dependency installation or external calls attempted.
  python3 .treqs/scripts/check_local_candidate.py passes shell syntax, exact
  publication contract, replacement destination, Python parsing, receipt markers,
  metric, hashes, embedded bytes, repeat writes and invalid step rejection.
  git diff --check passes. Full pytest remains in remote setup before fetch.

Commands actually run in this correction: pwd; rg --files searches for canary,
workflow, tests and AGENTS.md; git status --short; git diff --stat; cat and sed
reads of workflow, notes, README, contract, packager and tests; rg publication
references; ls -a; workspace-only Python edits; python3
.treqs/scripts/check_local_candidate.py; python3 -m pytest -q tests/treqs;
git diff --check. AGENTS.md search returned no matches. No training parameters
changed; the canary overrides documented above remain in place. The supervisor
must enforce the $15 budget and arrange independent verification after execution.
