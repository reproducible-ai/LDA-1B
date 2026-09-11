# Issue 35, supplied remediation 2, local iteration 1

- i35-r2-state-width (verified issue): supplied job 5020d01f failed before any
  completed update: state width 12 versus encoder width 58. Local mixture loader
  concatenates state without padding. Added opt-in shape-only forward adapter:
  retain 12 values, right-pad 46 zeros, leave action masks/weights unchanged.
- i35-r2-index (hypothesis, cause null): CUDA gather assertions may result from
  demo NEW_EMBODIMENT ID 32 exceeding the checkpoint category table. Actual table
  size is not available locally. Demo metadata says franka_robotiq; deliberately
  route this canary to upstream Franka slot 4, requiring at least five slots.
  This is an explicit compatibility choice, not a claim of semantic equivalence.
- i35-r2-cleanup (verified issue): finally deleted output_dict before assignment
  on forward failure, masking the original error. Initialize it to None; regression
  verifies the original failure survives.
- i35-r2-ready (decision): local workflow checks pass; request supervisor run
  and independent verification. No compute, credentials, external API calls,
  publication, artifact audit, optimizer completion or model quality claim.

Evidence: lda/dataloader/gr00t_lerobot/datasets.py,
lda/dataloader/gr00t_lerobot/embodiment_tags.py,
playground/demo_data/sim_pick_place/meta/info.json,
lda/utils/demo_canary_inputs.py, tests/treqs/test_demo_canary_inputs.py,
.treqs/issue-35-remediation-2-check-results.txt. Supplied external observations
are the source of the remote traceback; not newly retrieved.

Parameters: demo adapter absent -> true; state 12 -> 58 via right-zero-padding;
demo category 32 -> existing Franka 4. No architecture or state-dict key changes.
Retained one step, batch four, BF16 file ownership, PyTorch 2.9.0/cu128,
DeepSpeed micro/global four, frozen Qwen/DINO, pinned inputs, private directory
upload. Inherited recipe deviations remain documented in iteration-1-notes.md.
None of these changes restores full training. Supplied continuation spend
$1.07 leaves $12.47 of the $13.54 continuation cap; supervisor enforces budget.

Commands: pwd; rg/rg --files; head; sed; cat; ls; command -v; git status --short;
python3 import probes (torch/numpy missing); python import probe (torch missing);
Python workspace edit scripts; python3 tests/treqs/test_demo_canary_inputs.py
(3 passed); python3 .treqs/scripts/check_local_candidate.py (passed, including
receipt, inventory, replacement destination, runtime and adapter wiring checks);
git diff --check (passed). Full tests/treqs requires unavailable dependencies;
remote setup retains that suite. No packages installed, no commit created.
