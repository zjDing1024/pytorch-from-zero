# Local verification: development curves (2026-11-07 UTC+8)

CPU float64, Python3.12.14, PyTorch2.14.1+cpu. Existing CPU dependency environment was reused; editable binding was explicitly updated to the new checkout and isolated `python -I` package path checked before the main suite. No paid resource was used.

## Completed scoped checks

- New CLI suite: 20 passed (6.71s).
- Independent scientific review: 14 passed (4.46s initial,5.83s after dynamic accounting changes). These are included in the final main suite, not added again.
- Combined focused accumulation/new CLI/independent review: 74 passed (20.13s), including repaired command-list integration assertion.
- Existing standalone suites: 59 passed (8.22s), 45 passed (8.80s), 34 unittest passed (3.214s). These are outside the main suite.
- Ruff lint passed; initial 209-file format check passed (final checkout check: 210 files); compileall and pip check passed.
- First full run found two stale integration expectations for 23 installed commands. The checker now has 24. Both exact command-list/count tests were updated to retain all prior entries and include the new CLI. No scientific tolerance was relaxed. A diagnostic `-x` rerun reached the second stale assertion; neither incomplete/failed run counts as a passing full run.

## Installed package

A newly built noneditable wheel was installed in an existing separate CPU environment. 56 package Python/data files matched source bytes. From unrelated `/tmp`, all 24 installed CLI checks passed. The entire new experiment JSON matched the source final JSON on this host. [Machine-readable summary](2026-11-07-learning-curves-wheel-summary.json) retains the actual new experiment and every CLI outcome.

This is not a fresh locked-dependency installation, not a local Docker build, and not full pytest in the installed environment. Docker validation belongs to remote CI. The optional NumPy-unavailable PyTorch warning remained; NumPy was not installed to hide it.

## Scientific boundaries

The independent tests used actual SGD.step and evaluation spies: 1680 updates, 25680 training visits,528 evaluations and36048 evaluation visits. Poisoning test features did not enter development tensors; validation perturbations changed validation results without changing training parameter hashes or fitted preprocessing. An independent explicit LR loop matched the complete 20-epoch trajectory. Seed-grid omission, duplicate input, malformed curves and nonfinite values fail.

The final report was regenerated after converting work accounting from constants to trace-derived sums and adding the installed-checker success flag. The original first-run JSON remains separately retained; trajectories are unchanged. Rounding-safe mean computation (divide each finite term before summing) can change a final bit of an aggregate relative to the first run. Neither run changed the frozen candidate grid or selected endpoint.

## Final complete local suite

After repairing both stale integration expectations, the isolated final main suite passed: **1960 passed**, one optional NumPy warning,586.81s. With the separate59/45/34 suites, **2098 unique tests passed**. Scoped new tests are already included in1960 and are not counted twice. Final main test command: `python -I -m pytest -q`. The final run includes the14 independent scientific tests.

Remote publication and exact-commit CI are recorded in the learning repository's 2026-11-07 daily report after verification; no CI success is inferred from local tests.
