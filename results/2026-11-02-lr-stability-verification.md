# Local learning-rate stability verification

Date: 2026-11-02, Asia/Kuala_Lumpur. Baseline engineering commit
f3b42b7fbd5ef10b88dc87865ac9e5e14756bbb4. All 313 baseline Git blob hashes matched
before implementation. Historical science, numerical thresholds, dependency locks
and timeout budgets remain unchanged.

## Frozen experiment and observed results

The plan docs/lr-stability-plan.json was written before the first numerical run.
Four rates (.1,.75,1,1.25), 12 updates each, deterministic five-row CPU float64 affine
regression and plain SGD. The loss has Hessian diag(1,2), making the exact-arithmetic
error factors 1-lr and 1-2lr. The fixed 1e-12 absolute/relative oracle gate is unchanged.

The actual source CLI produced [48-step raw output](2026-11-02-lr-stability.json),
matching stdout; all main checks passed. Final losses, in rate order:
0.16071347777446246, 1.4901168299275014e-08, 0.25, 4208.528049007069.
The critical case first decreases from 2.25 to .25, then keeps a bias two-cycle.
The unstable case initially reduces total loss, later grows. Finite-prefix
observations do not themselves prove asymptotic behavior; the latter statement
comes from the exact-arithmetic quadratic analysis only.

## Preserved stress negative and test correction

The initial core-focused run had 47 passes and one failure: an extra lr=1.5,
32-step sweep incorrectly required every accepted finite configuration to pass
its floating-point oracle. In this runtime, weight comparisons fail at steps
28,29,31,32 under the unchanged 1e-12 gate while bias magnitude reaches about 2e9.
[Full stress output](2026-11-02-lr-stability-stress.json) retains false
all_oracle_matches and all raw values. No production code or tolerance was changed.

Execution validity and scientific agreement are separate. The revised portable
stress test requires finite values and truthful comparison aggregation rather
than promising agreement for that extra configuration or requiring all runtimes
to reproduce this mismatch. The independent reviewer recomputes each comparison
from the measured trajectory and scalar closed form. The frozen four-case
acceptance is unchanged. Focused core rerun: 48 passed, 2.43 seconds.
The numerical cause was not isolated; cancellation is a hypothesis, not a finding.

Initial lint checks found overlong/new compact statements before formatting;
formatting fixed them before final aggregate. A baseline-check shell quoting
error was corrected before successfully verifying all 313 hashes. Neither changed
repository scientific content or results.

## Final complete source validation

- Main pytest: 1,583 passed, 1 warning, 404.70 seconds.
- First-step standalone pytest: 59 passed, 6.49 seconds.
- Saved-forward standalone pytest: 45 passed, 5.35 seconds.
- Delivery unittest: 34 passed, .953 seconds.
- Total: 1,721 tests; 85 new relative to 1,636 baseline.
- The 48 core, 11 CLI and 26 independent tests are included, not additional counts.
- Ruff lint/format, compileall and pip check passed. Source Python 3.12.14,
  torch 2.14.1+cpu. Existing missing-NumPy warning is retained; this path uses Torch.

The existing CI commands were run from the repository root and the fresh source
was editable-installed first, so subprocesses imported the correct new package.
The prior diagnostics learning-rate bound and all original 15 installed command
tuples are preserved; a 16th command was appended. Three existing wiring tests
were adjusted to the new count/positions without dropping prior coverage.

[Independent review](2026-11-02-lr-stability-independent-review.md) includes eight
rates and 96 closed-form step comparisons, gradient/update norms, zero learning
rate, invalid inputs and RNG preservation. These tests are a subset of the main
suite. No numerical failure in the broader stress record was hidden by software
acceptance.

## Installed runtime and publication

Clean offline non-editable wheel validation passed: all 16 real CLI commands ran
from an unrelated temporary directory with python -I; 41 package members matched
source and installed bytes, all 10 dependency wheel hashes matched the existing
locks, and pip check passed. Final lint/format covered 148 files including document
code blocks. No source/test changed after the aggregate above.

Installed-wheel verification is reported separately in
[wheel summary](2026-11-02-lr-installed-wheel-summary.json). Local Docker and a full
pytest rerun inside the wheel environment were not performed. Exact remote commit
CI and actual Docker execution are checked after publication, recorded in the
learning report rather than inferred from local results.

No universal optimizer guarantee, arbitrary-model convergence, GPU, stochastic
training, momentum stability, cross-runtime equality or performance claim is made.
