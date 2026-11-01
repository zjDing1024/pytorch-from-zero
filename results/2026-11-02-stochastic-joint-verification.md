# Local stochastic joint-training verification

Date: 2026-11-02 (Asia/Kuala_Lumpur). Engineering baseline:
07a1a9fbe2f851a127909a1c24a856ce23c74628. All 291 baseline remote blob hashes
were checked before copying the working tree. This is an assistant-executed artifact
record, not evidence of the learner's personal mastery.

## Final source tests

- Main suite: 1,421 passed, 1 warning, 340.70 seconds.
- Standalone first-step suite: 59 passed, 4.67 seconds.
- Standalone saved-forward suite: 45 passed, 5.05 seconds.
- Standalone delivery validator: 34 tests, OK, 1.006 seconds.
- Total: 1,559 tests, 156 new relative to the 1,403-test baseline. The 32 independent
  review tests are already in the main count, not an extra addition.
- Ruff lint, Ruff format (132 files including Markdown code blocks), compileall and pip check passed. The PyTorch
  warning concerns absent NumPy; these checks use Torch, not NumPy.
- Local Python 3.12.14, torch 2.14.1+cpu. CPU float64 only.

Commands were the existing source CI commands: main pytest, both standalone pytest
suites and unittest discovery for tools/wine-delivery. The final source installation
was updated to this working tree before testing, so CLI subprocesses imported the
new modules. No prior run's passing tests are counted as this run's execution.

## Actual fresh-process experiment

[Raw complete report](2026-11-02-stochastic-joint.json) was produced by
`python -m pytorch_lab.stochastic_joint_cli --output <absent-path>` and matched its
stdout. It used six worker processes with seeds 7, 42, 123 and three epochs:

- Nine normal resumes at completed microbatch cuts 1, 3 and 5: 108 suffix steps and
  117 safe-point comparisons, all exact.
- Fifteen fault resumes across after_fetch, after_forward, after_backward,
  after_update and before_commit: 177 suffix steps and 192 safe-point comparisons,
  all exact.
- Each path compares committed indices, complete step traces and canonical full
  state digests. The report retains 45 full baseline traces, including outputs,
  loss, dropped inputs and accumulated raw gradients, not just a success flag.
- The failed owner rejects continue, snapshot and save. Recovery reloads the whole
  old snapshot; caller default CPU RNG is preserved. Parameter updates are real.

These are controlled Python exceptions, not kill-process or power-loss tests. Cases,
steps, safe points and tests are different units and must not be summed together.

## Independent mathematical and failure checks

[Independent review](2026-11-02-stochastic-joint-independent-review.md) reports
18 configurations and 144 manually differentiated affine/SGD microsteps. The oracle
uses actual dropped inputs, with atol=rtol=1e-12; the RNG algorithm is not independently
implemented. This gate is separate from exact resume equality. Four operation-then-
raise tests perform the real forward, backward, optimizer or epoch-advance mutation
before throwing, so failure testing is not limited to a flag that raises too early.

The separate core math tests cover nine configurations and 135 steps; they are
part of the main suite too, not additional experiment or test counts.

## Failure history retained

1. Initial core-focused run: 104 passed and nine oracle-test failures because the
   test reconstructed output lists as default float32. The test was corrected to
   float64; no production change or tolerance relaxation. Full focused rerun: 113 passed.
2. Independent reviewer initially saw 31 passed and one malformed-test-construction
   failure: scalar NaN could not be rehashed by the existing JSON canonicalizer.
   Replaced that fixture with finite invalid p=1.1 to reach semantic validation;
   full independent rerun: 32 passed. Production and threshold unchanged.
3. First main aggregate: 1,419 passed, two failed, 341.59 seconds. Both were old
   installed-CLI count/list assertions expecting 13 entries after adding the 14th.
   Focused checking of the first fix then caught the remaining negative-position
   assumptions (one fail, one pass); that intermediate aggregate was stopped rather
   than presented as successful. Final assertions preserve the exact original 13
   command tuples and append the new 14th, using stable positions for old entries.
   Independent AST review confirmed no old command or assertion coverage was removed.
   The focused pair then passed, and the complete final aggregate above was rerun.

A later lint invocation from the parent directory failed to match the repository-relative
exclusion for the frozen historical source snapshot. Re-running the unchanged CI commands
from the repository root passed; the frozen artifact was not edited.

All historical scientific inputs, outputs, numerical gates and timeout budgets remain
unchanged. Only two existing wiring tests were adapted to the appended CLI.

## Installed wheel and publication boundary

The non-editable wheel was built and installed into a fresh virtual environment using
the existing hash-locked offline wheelhouse. All 14 real CLI executions passed from
an unrelated temporary directory under python -I. All 37 wheel package members matched
final source and installed bytes; all 10 runtime wheel hashes matched the locks.
Pip check passed. Actual CLI and package-byte verification
is recorded in [installed-wheel summary](2026-11-02-installed-wheel-summary.json).
It is a separate runtime path; full pytest is not rerun inside that wheel environment.
Local Docker was not run. Source CI and actual Docker results for the exact public
commit are recorded after publication in the learning report, never inferred from
this local record.

No GPU, AMP, DDP, workers/prefetch, concurrent RNG consumers, arbitrary modules,
cross-runtime checkpoint exchange, external-effects exactly-once, authenticated
checkpoint provenance or durable power-loss recovery claim is made.
