# Wine update-clock independent review

Date: 2026-11-06 Asia/Kuala_Lumpur (execution 2026-11-05 UTC).

No blocking findings within the declared scope. Core repository files were read only by the reviewer. This review does not establish full-suite, installed-wheel, remote CI or new-process recovery success.

## Executed checks

The reviewer separately ran inline protocol and negative-control scripts, then retained the full-protocol test in `tests/test_wine_update_schedule_independent.py`. The retained test plus the implementation's core and CLI tests passed: **39 passed, 1 optional NumPy warning, 17.86 seconds**. After formatting and adding strict zip cardinality checks, the retained test alone passed again: **1 passed, 1 warning, 3.58 seconds**. That rerun is not an additional unique test. Pytest 9.1.1 and Torch 2.14.1+cpu were used. Ruff passed using the repository's pyproject configuration.

- Poisoning every nontraining row's feature with NaN and label with -100 did not affect the experiment. A forward guard confirmed every scored/trained input was finite.
- Instrumented actual work: 588 optimizer updates, 12,486 training sample visits, 60 endpoint evaluations and 6,420 evaluation sample visits. All agree with reported counters.
- All twelve constant-rate runs exactly match the existing batch-budget implementation's parameter trajectory and endpoint train CE.
- A separate explicit SGD momentum/weight-decay recurrence, without invoking the scheduler or UpdateRun.step, exactly reproduces all 168 scheduled parameter and momentum hashes. It uses LR 0.05 for updates 1–7 and 0.025 for updates 8–14 and independently reconstructs batch grouping from permutations.
- Caller's CPU RNG state remains unchanged.
- Restoration after updates 6, 7 and 8 reproduces complete state at the cut and every remaining boundary. Only trusted internal snapshots restored into fresh objects in the same process and on the same input are covered.
- Additional inline faults: removing momentum after update 7 changes parameters on update 8. Resetting scheduler state after update 7 can leave immediate parameters unchanged while complete state differs. An extra scheduler step is caught by full-state comparison. Injecting a scheduler failure after optimizer mutation invalidates the owner; step, snapshot and report all refuse further use.

## Scope and remaining limits

The snapshots do not bind or validate arbitrary replacement datasets, hostile checkpoint contents, concurrency or arbitrary external mutation. Those are outside the declared trusted internal experiment contract. No file durability, cross-process, AMP, GPU or distributed claim is supported by this review. A step7 schedule creates a next LR of 0.0125 after the final update 14; it is not used by the bounded experiment. Equal update counts across batch16/32 do not mean equal sample exposure.

## Reviewed code identities

- wine_update_schedule.py SHA-256: 842994ee30f6a8bfae8a506df69d9e306c35e3f18d70de6bc702a1ebd812f484
- wine_update_schedule_cli.py SHA-256: 2239da1548b3af4a64217e65391c36fc1af6cf3f104933c5cd557a47bdbd8175
- retained independent test SHA-256: 72db9e3c81973f209682e4e43d72633c4531b0558397fe7bd03b25fc43e2425e

Run from the repository root with its source on PYTHONPATH:

`PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_wine_update_schedule_independent.py tests/test_wine_update_schedule.py tests/test_wine_update_schedule_cli.py`

The initial wheel-only Python lacked pytest; no test ran in that failed invocation. The reviewer used the existing development interpreter instead. Initial standalone test lint showed import/strict-zip formatting issues, corrected only in the reviewer-owned file and rerun; no numerical thresholds changed.
