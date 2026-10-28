# Shuffled index stream — executed verification

Date: 2026-10-29 (Asia/Kuala_Lumpur). These are assistant-run engineering checks,
not evidence of independent learner mastery or training recovery.

## Exact baseline and change boundary

- Engineering baseline: `570b47c78c6e7b82ef8b67ae38fc73d2f7f6b6b4`.
  All 262 baseline file bytes matched their remote Git blob hashes before editing.
- Roadmap baseline: `b29db426b1d1206b1816edb40790de8810ae60fa`, 27 verified files.
- Runtime: Python 3.12.14, PyTorch 2.14.1+cpu; pytest 9.1.1 and Ruff 0.16.10.
  Fresh virtual environments used the existing official-index CPU wheelhouse,
  revalidated by the unchanged `requirements/runtime-cpu.lock` hash requirements.
- Prior scientific datasets, fixtures, results and strict/8-ULP thresholds were not changed.
  No new model accuracy, throughput, or cross-platform numerical claim was made.

## Checks actually completed

| Check | Result |
|---|---|
| New focused `tests/test_shuffle_stream.py` | 96 passed, 7.59 s |
| Full main `python -m pytest -q` | 1,062 passed, 267.34 s |
| First-step standalone pytest suite | 59 passed, 5.34 s |
| Saved-forward standalone pytest suite | 45 passed, 6.34 s |
| Delivery standalone unittest suite | 34 passed, 1.246 s |
| Distinct total (focused tests not added again) | **1,200 passed**, including 96 new |
| Ruff lint / format | Passed / 109 files already formatted (including Markdown code blocks) |
| Source and fresh-wheel environment `pip check` | Passed |
| Source new shuffled-index CLI | Six fixtures, all 99 checkpoint suffixes and final states exact |
| Independent script | 144 configurations, 1,836 cutpoints, 3,672 future-epoch comparisons passed |
| Independent invalid-input checks | 51 malformed snapshots, 6 expected-config mismatches, 8 invalid files rejected |
| Fresh noneditable wheel, outside source tree, `python -I` | All **11** CLI entries passed, old ten in original order |

The checkpoint/case totals are experiment cases, not additional pytest tests.
The new source CLI uses one fresh producer and a distinct fresh consumer for all
99 saved checkpoints, not 99 independent consumer processes. The standalone oracle
adds a separate fresh-process partial-epoch check and two future epochs.

A benign warning reported that optional NumPy was absent; the locked dependency
closure does not include NumPy, and no test used NumPy conversion. Runtime installation
was offline/hash-locked; development tools used their existing pinned requirements.
This is not a full transitive development-dependency lock claim.

## What the result establishes

The uninterrupted stream and every resumed suffix match an expected sequence built
from independently seeded CPU `randperm` plus Python slicing. Current-epoch order,
tail policy, explicit next-epoch RNG continuation, stable exhaustion, snapshot isolation,
no-draw restore, failure atomicity, bounded parsing and no-clobber save are covered.
The index-only contract handles empty and drop-all epochs without implicit looping.

At the fixed ten-indices-issued point, an intentionally recomputed checksum with a
rewound valid cursor duplicates ten indices; advancing it skips five. Resetting same-seed
RNG preserves the current suffix but changes the next epoch. Wrong expected seed is
rejected; restarting with the wrong seed also diverges from the correct oracle.
These are semantic fault controls, not cryptographic signatures or training metrics.

The local new CLI report, independent report and
[compact installed-wheel summary](2026-10-29-installed-wheel-summary.json) contain
public-safe configuration/status/source hashes. Full raw installation and execution
logs are retained privately; their provenance hashes appear below, without machine paths.

## Limitations and CI scope

No training integration, downstream acknowledgment, worker/prefetch/distributed
recovery, GPU, cross-runtime exactness, arbitrary DataLoader compatibility, or actual
local Docker execution was tested. The stream checks a caller-supplied dataset identity,
not the actual data bytes; restore callers must supply independently known configuration.

The workflow preserves the 15-minute source and 20-minute container budgets, adds the
new source CLI and independent verifier, and lets the installed-wheel checker run all
11 entries in the actual container job. Local wheel success is **not** evidence that
Docker CI ran; exact-commit remote workflow status is recorded separately in GitHub
and the accompanying roadmap daily report. The container checker runs CLIs, not the
full pytest/unittest suites.

## Provenance hashes

- private main-suite log: `6b5cd31d9b5fdd7cc47aa760d6aa5fb38bc4bff5dce6c4556b9a82cfb0bbfdf4`
- private first-step log: `b6e9e1fe4b7083ca4d940f287b9bb5c3f3d90d5539fae483415238ea71f29e4a`
- private saved-forward log: `d85c65e9a3e952427899baf1a227b74a30a6a5dd90f4c56d2afb5e4c7509be17`
- private delivery log: `b6127f01b0b2025f78156fb13ed8c241dd1f2f8d58b9e02a9d4303145e5cecaf`
- private installed-checker full JSON: `58e1f986e039835659c95a944794d44456ff73b5ade3abd546478374175871d1`
- source new-CLI compact JSON: `df523950816be5dd4b1485b4e6e213d18603029ac6beed89ba6a16668ac86c21`
- independent verifier compact JSON: `e4c83c8db16ccbbb359599e74a0cc457c125e639bbb0633266f4a60dd9775fd7`
- installed-wheel compact summary: `0332a3b9c7a910740e14562075e93f4af23a74a28ad047945dc392448d471983`
