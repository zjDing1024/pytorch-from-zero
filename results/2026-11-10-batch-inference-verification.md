# Batch inference verification, 2026-11-10 Malaysia time

Implemented on the actual current date, starting from remote project
aca507c1e6569e81acd0aa1700314590c34f48ab. No backdated November9 execution claim.
Python3.12.14 / PyTorch2.14.1+cpu / x86_64 Linux; CPU only, no paid resources.

## Verification scope

- Final complete source pytest: **2060 passed**,437.42s. With the three standalone
  suites below, **2198 unique tests** passed. This includes the corrected interface
  assertions and final29-test independent review file.

- Primary new tests: 28. Independently authored review tests: 29. Included in the
  final main count, not added again. A focused final rerun of the changed integration
  and inference suites passed 146 tests in 31.77s.
- Standalone suites: 59 passed in7.14s, 45 passed in5.70s, unittest34 in1.018s.
- Ruff check and format with the existing archived-file exclusion, compileall,
  git diff whitespace check and source/wheel pip check passed.
- Noneditable wheel built offline and installed into the existing isolated wheel
  environment; 61 package source/data files matched byte-for-byte. All26 installed
  CLI commands passed, including all68 new batch experiment checks. See
  [wheel summary](2026-11-10-installed-wheel-summary.json). This is not full pytest
  inside the installed environment or local Docker verification.
- Optional NumPy is absent and Torch emits its existing initialization warning.
  This increment does not use NumPy conversions; no missing dependency was hidden.

First whole-tree run: 2056passed /3failed /479.38s. Two older tests still expected
25 CLI entries rather than26; updated only those interface expectations. The third
failure was a review-test parameter ID containing a million-byte test payload;
pytest exported it in PYTEST_CURRENT_TEST, causing process launch failure. Explicit
short test IDs fixed this test harness issue. A final review test was then added,
so the final collected count differs by one. No implementation defect or tolerance
relaxation was required. Earlier targeted run used a nonexistent test selector and
ran zero tests; the corrected146-test run above completed successfully.

## Actual measurements

[Final raw JSON](2026-11-10-batch-inference-final.json) contains the complete samples,
component timings, environment, row outcomes and68/68 verification checks (64 numerical/class oracle checks plus4 mixed-error
contract checks).
One warmup and3 retained repetitions per batch size, one CPU thread, float64.
This final measurement ran before starting the final pytest suite, after the prior
suite and installed checks completed. No other work from this task ran concurrently;
ambient host load was not controlled or recorded.

- 1 rows: warm total median 0.400214 ms; forward median 0.128594 ms; fresh-process file-to-file median 1.409428 s. Input 290 bytes; last response 766 bytes.
- 16 rows: warm total median 0.607707 ms; forward median 0.139941 ms; fresh-process file-to-file median 1.399932 s. Input 1317 bytes; last response 4067 bytes.
- 107 rows: warm total median 0.875652 ms; forward median 0.144738 ms; fresh-process file-to-file median 1.307051 s. Input 7488 bytes; last response 24207 bytes.
- 1024 rows: warm total median 3.303665 ms; forward median 0.197117 ms; fresh-process file-to-file median 1.403125 s. Input 69788 bytes; last response 227945 bytes.

Do not compare these wall-clock figures as a universal speed ranking. Three samples,
fixed warm-then-fresh ordering, unrecorded CPU model/host load and OS page caching
limit inference. Fresh-process totals include Python and PyTorch startup, imports,
model loading, JSON I/O and exit; warm totals do not. Child file diagnostics end
before output encoding. Neither timing measures fsync durability or concurrency/SLA.
Fixture training (140updates /2140training visits) is excluded from these timings.
No new held-out score or monetary-cost extrapolation is made.

The preliminary source run and installed-wheel benchmark were executed alongside
other verification work; they are retained, not selected for a favorable ranking.
Their elapsed values are not treated as controlled performance comparisons.
The separately executed mixed-file example returns2 successes /2 errors with all
original indices preserved; its one-off timings are functional evidence only.

## Publication

Exact remote commit and CI are checked after publication and recorded in the
learning repository's November10 daily report. A passing local suite is not
reported as a remote CI/container result.
