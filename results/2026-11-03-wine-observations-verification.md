# Wine training observations: local verification

Date: 2026-11-03, Asia/Kuala_Lumpur. Baseline engineering commit:
`e3be1f1680c6014ef3238ec66731de2e959ec1e6`. All 336 baseline blobs were checked
against the fixed remote tree before work. Taskbook and current roadmap were read.

## Final executed checks

| Check | Actual result |
|---|---|
| Main `python -m pytest -q` | 1,730 passed, 390.71 s |
| First-step standalone pytest | 59 passed, 6.07 s |
| Saved-forward standalone pytest | 45 passed, 5.98 s |
| Delivery unittest discovery | 34 passed, 1.127 s |
| Distinct suite test total | **1,868**, 62 more than baseline |
| New focused core tests | 35 passed, 3.30 s |
| New CLI tests | 20 passed, 3.94 s |
| Independent review subset | 7 passed, 2.61 s; already included in main |
| Existing installed-count tests | 7 passed, 117 deselected; not added to total |
| Ruff lint / final format | passed / **164 files** |
| compileall / source and wheel pip check | passed |
| Fresh non-editable offline wheel | **18 actual CLIs passed** |

Ruff excludes only the previously frozen saved-forward source snapshot, using
the existing repository command. The early format check saw 162 files before
the new protocol document was added, then 163 before this verification record;
164 is the final count. Initial static
checking overlapped the reviewer's in-progress test writing, then final lint and
format were rerun after all tests were frozen. A final core-test whitespace-only
format occurred during the aggregate run; scientific assertions were unchanged.
The final source/test hashes were frozen and checked again after completion.

Python 3.12.14, Torch 2.14.1+cpu, CPU float64. The existing optional-NumPy warning
remains: NumPy is absent, and this implementation uses native tensors. Pip also
reported an unwritable optional cache and disabled caching; installs completed.
No paid compute, local Docker, GPU, AMP or distributed execution was used.

## Scientific evidence

The frozen plan SHA-256 is
`a13ef89e592f988493cd8c8276170fa2de5af8e1e69d7e5848de26a334a025dc`.
All six architecture/seed configurations retain 14 updates / 214 training visits.
The observation report contains 84 batches and 252 parameter records. All enabled
versus disabled diagnostic audit traces agree exactly, and legacy uninstrumented
final parameters, histories and budgets agree exactly. Including both references,
execution contains 252 optimizer steps / 3,852 training sample visits.

Maximum actual-delta versus momentum reconstruction residual:
6.245004513516506e-17, under the unchanged atol=rtol=1e-12 check. Nine initial
bias records have an undefined ratio represented by null plus explicit status.
No thresholds assert that one layer/model is better or uniquely diagnose causes.

Source CLI stdout and the public raw JSON have identical bytes, SHA-256:
`2cf12f7a007c88a98db4bfa4f6be14fb14382f30607f4c9b5967d5095d5a7b1d`.
The [independent review](2026-11-03-wine-observations-independent-review.md) checks
SGD decay/momentum equations independently, measurements against real snapshots,
held-out feature/label contamination, training-only forward guards and actual
runtime work counters. Deliberate optimizer no-op and diagnostic mutation tests
demonstrate that measurement/reconstruction and noninterference failures surface.

## Installed delivery evidence

Built a project wheel locally, created a fresh virtual environment, installed
10 runtime wheels offline with the existing `--require-hashes` lock, and installed
the project non-editably with no dependencies fetched. All 45 package members
match final source, wheel archive and installed bytes. Wheel SHA-256:
`27bd9e77dd92a736ff4ca165169821c3c90d65d3f210ee8c22cc396ce1fda104`.

Executed `python -I scripts/check_installed.py` using the fresh interpreter from
an unrelated working directory. The first 17 command tuples are unchanged from
the baseline; Wine observations is appended as the eighteenth. All 18 exited zero
and passed their report checks. Raw installed output is retained privately outside
the repository; the [public compact summary](2026-11-03-wine-installed-wheel-summary.json)
contains hashes and scope without local machine paths.

These are actual installed CLI runs, not a full installed-environment pytest or
local Docker run. The new probe never scores validation/test data. The unchanged
delivery suite separately reruns the historical full Wine benchmark, including
held-out scoring; none of its output guided settings or changes in this probe.

## Development corrections and boundaries

Initial focused tests had two incorrect bitwise expectations from Python `sum`
versus the original sequential `+=` accumulation, and one mixed-float32/float64
fixture. Test-only correction of the dtype initially affected three nonfinite
fixtures as well; those were restored to consistent dtype. The final 35 core
tests passed. Scientific code, thresholds and historical results were not changed
to fit those failures. One initial CLI invocation resolved an older editable
checkout; setting the current source path and installing this checkout fixed
module resolution. Subsequent source and fresh-wheel CLI executions succeeded.
The independent review records its separate test-oracle corrections.

The two-epoch observation prefix is not the original 120-epoch benchmark or proof
of convergence, accuracy improvement, universal optimizer superiority, measurement
overhead, cross-platform bitwise behavior or personal learner mastery. Local work
does not itself prove a remote commit or CI result; publication is verified
separately by the publishing task.
