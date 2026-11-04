# Reconstructed gradient sampling: actual November4 validation

The November3 uncommitted artifacts were unavailable. All results below come from
this newly implemented and executed replacement. The prior 1952-test/1842-call
report is historical, not evidence of this implementation. The frozen replacement
uses32draws per size and1266gradient calls/18054sample visits, zero updates.

## Local source and distribution checks

- Complete main suite: **1812 passed, 366.70 seconds**.
- Standalone suites:59passed5.02s,45passed5.03s,34unittest0.923s.
- Total **1950 = 1812 +59+45+34**,39new tests (11core,20CLI,8independent).
- Ruff lint and final180-file format check pass (prewheel178; intermediate179; final observed180),
  compileall and pipcheck pass. Existing optional NumPy warning remains.
- Fresh offline noneditable wheel:20actual installed CLIs passed from unrelated
  temporary working directories using isolated Python;49package members match
  source/wheel/install bytes. Ten dependencies use existing official hash locks.
- Installed new gradient JSON exactly equals this runtime's source raw JSON.
- No local Docker execution and no full pytest inside the installed environment.
  Exact-commit remote CI is recorded separately after publication.

## Actual scientific evidence

Sixmodel/seed configurations, all five deterministic controls pass. Weighted
partition maxabs error≤2.220446049250313e-16. Naive equal-batch mean errorL2
0.07795060920469293–0.09396175119916425. Empirical/theoretical squared-error ratios
range0.762392084431612–1.2375402944172407; these are descriptive32draw estimates,
not pass thresholds or significance. Independent tests capture actual tensors,
check analytic linear gradients, exhaustive finitepopulation subsets, and poisoned
heldout inputs with forward guards. Existing gradients/RNG/threads are preserved.

Frozen plan SHA256 `3fd206d745f1c95e8d2f320707528b0724f1f7ea91eece81990448cf0667fa8c`.
Raw evidence SHA256 `c128f5510554f4f14282f34bc689b6f6338607a9c65add9646165ee6af588c53`.

## Development record and limits

Initial hand-written formatting triggered lint line-length errors; formatting and
splitting long literals resolved them before the aggregate run. Focused31tests
and independent8tests passed; these are included in the final main count, not
added again. No scientific tolerance, sampling budget or fixture was changed to
make results pass. All historical code/data/results outside the small integration
changes remain unchanged. The existing19CLI command-list assertions were updated
explicitly for20entries before the full run, without removing earlier commands.

Fixed CPU float64 initial parameters and small Wine data do not establish training
speed, convergence or generalization. The legacy delivery suite separately reruns
historical heldout scoring, which is not used to select this experiment's settings.
