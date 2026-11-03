# Wine momentum local verification

Date: 2026-11-03 (Asia/Kuala_Lumpur). Baseline:
`1c4aaaf7eee34d32b7dab1f5f34bf3232efa8fb0`. All 347 baseline Git blobs were
verified before edits; no old result or data blob was modified.

## Scientific execution

The original `wine_momentum_cli` ran on Python 3.12.14 / PyTorch 2.14.1+cpu,
CPU float64, one thread. It produced six matched model/seed pairs, twelve arms,
84 paired batches / 168 optimizer updates, 2,568 optimizer sample visits, and
24 full-training evaluation forwards (2,568 additional evaluation visits).
There is no audit replay inside the experiment. Independent tests and the
regression suite execute extra runs and are not included in these costs.

All eight controls in all six pairs pass. Maximum actual SGD equation residual:
`6.245004513516506e-17` under predeclared `atol=rtol=1e-12`.
Frozen plan SHA-256:
`ce5abc56ef68aa4d6fd622d01d6c42f4c4e9bac1fd787f7d508a2e2fcbc8c421`.
Raw result SHA-256:
`582162de5a620811b02a34c0f83bb18ef53fa4c201897262fe2191cf46ca5c4b`.

Independent review checks real optimizer snapshots at `1e-14`, both arms against
the original uninstrumented trainer, heldout poisoning/forward guards, exact
budgets and caller RNG/thread preservation. Its seven tests are part of the
main suite, not extra tests to add to the aggregate count.

## Local test execution

| Check | Observed result |
|---|---|
| Final complete main `python -m pytest -q` | 1,773 passed in 340.68 s |
| First-step standalone | 59 passed in 5.17 s |
| Saved-forward standalone | 45 passed in 5.22 s |
| Delivery unittest | 34 passed in 1.016 s |
| Total source tests | **1,911 = 1,773 + 59 + 45 + 34** |

The main suite adds 43 tests: 16 core, 20 CLI and 7 independent review. Its final
execution follows the last test-only integration expectation fix. The focused
43-test pass is included, not counted again.

Ruff lint and final format check pass (171 files), compileall passes, source pip
check passes. The optional missing-NumPy Torch warning remains; these experiments
do not depend on NumPy. No tolerance, optimizer configuration, epoch budget or
historical evidence was relaxed to obtain success.

## Fresh wheel delivery

A newly built noneditable wheel was installed into a fresh virtual environment
using ten cached official runtime wheels and the repository hash locks. All
47 package members match source, wheel archive and installed bytes. Wheel SHA:
`b2070656bf293df0ab92687f42e20b4ba477a43c5338823d9d5ebeee80ea1d5d`.

`python -I scripts/check_installed.py` ran outside the source tree with PYTHONPATH
removed. **All nineteen actual installed CLIs pass**, including the appended
momentum CLI; prior eighteen command tuples remain unchanged. Its complete
momentum report equals the source JSON in this same local runtime. The structured
checker stdout and output file agree. Full checker report SHA-256:
`d56b1d63c5e833864e8f0b3bb91296231257b05f66cbe5697fecfc7fb21dabbd`.
See the [compact wheel summary](2026-11-03-momentum-installed-wheel-summary.json).

This is not full pytest inside the wheel environment and not a local Docker run.
The existing whole-delivery suite separately reruns the historical Wine benchmark
with heldout scoring; those outputs did not tune this new train-only comparison.
Remote CI and real Docker outcomes must be checked for the eventual published
commit rather than inferred from this local verification.

## Development failures and corrections

- Independent review found the first paired-distance formula reconstructed final
  parameters via `before + delta` and introduced subtraction-order artifacts.
  Production measurement was corrected to directly subtract actual post-update
  tensors. The initial development JSON remains outside public release evidence.
  Training trajectories, CE, cumulative path lengths and all scientific settings
  were unchanged; the final raw output was regenerated after this correction.
- Initial core tests used Python 3.12's compensated `sum` against production's
  sequential `+=`, producing last-bit discrepancies in path length/online CE.
  Tests now reconstruct the same sequential accumulation, without relaxed tolerance.
- An initial subprocess CLI test resolved the old editable checkout. Installing
  this source checkout fixed module resolution; subsequent focused and clean
  installed-wheel execution passed.
- One lint import-order issue was corrected before final aggregate execution.

The first focused run had seven failures (six accumulation-order assumptions,
one stale editable subprocess); the corrected focused run passed 43 tests in
5.36 seconds. These are development records, not additional aggregate passes.

The first full main suite completed with 1 failed / 1,772 passed in 369.27 s.
The sole failure was an older exact installed-command list expecting eighteen
entries. Its expectation was extended with the new nineteenth command, retaining
all earlier entries. The complete main suite was rerun after this test-only fix;
the final count above is not assembled from partial runs.
