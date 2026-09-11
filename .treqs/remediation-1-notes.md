# Remediation 1: dependency index selection

- `uv-first-index-certifi` (verified cause): supplied externalObservationsPath
  executionLogs.tail reports setup failed because certifi==2025.11.12 was absent
  on the first matching index, https://download.pytorch.org/whl/cu128. The local
  requirements file enables that extra index; setup omitted index strategy.
  This does not establish whether all pins are available on PyPI.
- `uv-cross-index-fix` (decision): set --index-strategy unsafe-best-match only
  on the requirements installation. This considers PyPI and the PyTorch index
  together as suggested by the supplied uv error. All exact dependency pins,
  including certifi and the Blackwell stack, remain unchanged. The local checker
  guards this invocation and retains the certifi pin and CUDA index checks.
- `remediation-local-checks` (observation): dependency-free checks pass, including
  shell parsing, publication destination replacement, structured receipts,
  complete inventory hashes, rejection gates, and mocked single-GPU runtime.
  These are candidate tests with synthetic inputs, not an artifact audit.
- `remediation-runtime-pending` (hypothesis, cause null): complete Linux dependency
  resolution, GPU memory fit, and the real optimizer update remain unverified.
  System Python lacks pytest; the workspace-only offline uv cache lacks pytest.
  Full tests remain required by remote setup before fetch. No network, API,
  credentials, compute, publication, or independent audit was performed locally.

Parameter change this remediation: uv first-index default -> unsafe-best-match
for requirements installation. No training or software version pins changed.
The inherited canary remains one step, one 96 GB Blackwell GPU, effective batch
four, BF16, frozen Qwen/DINO, policy-only four-demo training. Published recipe
deviations remain documented in iteration-1-notes.md's final capacity-fallback
section. This change does not restore the full run or imply model quality.

Commands actually run: pwd; rg --files searches (including hidden AGENTS.md,
none found); head; git status --short; cat on requirements, workflow, pyproject,
README, tests, checker and inherited notes; ls -a .treqs; command -v uv.
python3 -m pytest --version failed (pytest unavailable).
python3 .treqs/scripts/check_local_candidate.py >
.treqs/remediation-1-check-results.txt passed.
UV_CACHE_DIR=.local-check-cache uv run --offline --no-project --with
pytest==8.4.2 --with pyyaml==6.0.3 --with torch python -m pytest -q tests/treqs
failed (offline cache miss). git diff --check passed.
Edits used apply_patch. No commit created.

Evidence: .treqs/remediation-1-check-results.txt,
.treqs/workflows/robocasa-demo-canary.yaml, requirements.txt,
.treqs/scripts/check_local_candidate.py; supplied externalObservationsPath
executionLogs.tail. Prior supplied cost is $0.56; supervisor must enforce the
$15 campaign total and arrange independent verification after the bounded run.
