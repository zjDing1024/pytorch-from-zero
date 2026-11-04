# Fresh local verification: bounded batch update-loop measurement

Execution date 2026-11-04 UTC / report date 2026-11-05 Asia/Kuala_Lumpur.
Baseline commit `31950fcf0032f2cdaaacb2f0854039dc183568f4`.

## Actual tests

- Main project `python -m pytest -q`: **1887 passed, 419.42 seconds**.
- First-step standalone: 59 passed, 5.27 seconds.
- Saved-forward standalone: 45 passed, 5.56 seconds.
- Delivery standalone, from its own documented directory: 34 tests, 1.043 seconds, OK.
- Total **2025 = 1887 + 59 + 45 + 34**, 35 new tests (11 core,20 CLI,4 independent).
  Focused/independent reruns are included once, not added again.
- Ruff lint and final format **195 files** passed with the project's existing
  historical-source exclusion. compileall and pip check passed.
- Missing optional NumPy warning retained; no NumPy dependency is needed.

## Actual installed delivery

A fresh noneditable wheel was built with the existing official build tooling and
installed offline using 10 hash-locked runtime dependencies in a new venv.
53 package members were byte-compared across source, wheel and installed paths.
From an unrelated directory, `python -I scripts/check_installed.py` actually ran
all **22 CLI entries**, including every earlier entry, with all checks passing.
The new experiment's scientific projection (states, CE, budgets, phase/repeat)
equals local source output in this runtime. Timing fields differ naturally and
are never included in an equality assertion.

See [installed summary](2026-11-05-compute-installed-wheel-summary.json) and
[independent review](2026-11-05-batch-compute-independent-review.md).
No local Docker execution or full installed-environment pytest was performed.
Remote CI is a separate execution and must be verified against its exact commit.

## Scientific result and cost

The raw [JSON](2026-11-05-wine-batch-compute.json) retains all warmup and measured
observations. All 72 lean runs exactly match their respective instrumented
references in final parameter tensors, CE and sample order. Reference18 +
warmup18 + measured54 =90 trajectories:1080 updates,24390 training visits,
90 endpoint train forwards/9630 evaluation visits. Timing values are descriptive
within-process observations, not significance tests or portable speed claims.

Independent clock spies verified actual optimizer steps inside both clocks and
evaluation/hash outside; a poisoned-heldout forward guard independently counted
the complete budget and checked caller RNG/threads. The tests' synthetic clock
values are not performance data. Main and installed checks were concurrent on
this shared host; the published raw timing run preceded those aggregate checks.

## Development issues preserved

Initial focused tests:30 passed,1 failed because the copied integration test
expected the new command at index20 rather than21. Corrected only that expected
position and source artifact filename, then all31 focused tests passed.
Initial lint found long strings/import formatting; corrected without changing
scientific settings. Independent tests corrected their lint formatting before
passing. During remote restoration, shell interpolation damaged backticks in
some historical Markdown; blob comparisons caught it and safe literal writes
restored the original remote bytes before final tests/manifests. Historical data
and results remain unchanged. No numerical threshold, training budget, timeout
or timing acceptance gate was relaxed or introduced to obtain success.
