# Wine model bundle verification, 2026-11-08 Malaysia time

Local Python3.12.14 / PyTorch2.14.1+cpu. No GPU, no paid compute.

- Final complete source pytest: **2003 passed**, 388.83 seconds. Includes 20 primary
  new tests and 23 independently authored review tests; do not count them twice.
- Standalone suites: 59 passed /4.91s; 45 passed /5.26s; unittest 34 /1.037s.
  Total **2141 unique tests**.
- Ruff check/format with the existing historical frozen-file exclusion passed;
  compileall and pip check passed.
- First source pass: 2 failed, 1978 passed, 396.37s. Two old assertions expected
  24 CLI entries, corrected to the exact 25-entry interface; full final rerun passed.
- Initial independent algebra oracle used matmul plus bias rather than fused addmm,
  observed 1.78e-15 arithmetic difference. Corrected oracle to matching fused formula,
  without relaxing tolerance. Reviewer then tested schema, train-only statistics,
  heldout-feature poisoning, RNG preservation, exact previous endpoint state,
  fresh-process loading, invalid values and actual SGD/CE calls. All 23 passed.
- Early standalone invocation used the wrong relative interpreter path, exited
  before tests. Corrected working path and reran all three suites above.
- Initial whole-tree lint included an archived historical pre-capture source file
  excluded by existing CI. Used the documented CI exclusion; did not rewrite history.
- Installed noneditable wheel passed all 25 CLI commands. A final numeric-rejection
  ordering change rejects enormous integers with ValueError before float coercion;
  source final suite includes this fix. Final wheel rebuilt and installation checked
  separately; final installed run/remote CI outcomes are recorded in daily notes.

The final original training report and standalone model/request/prediction JSON
are committed. 140 actual updates /2140 training visits; 107 public training-feature
rows produce identical logits before and after a fresh-process reload in this run.
No validation/test accuracy or performance benchmark is added. Prior full-suite
Wine tests still score their historical fixed test protocol, not used by this model
export experiment. SHA256 is integrity, not authenticity.

Publication and exact-commit CI are verified after this commit and recorded in the
linked learning repository daily report; no CI success was assumed before running.
