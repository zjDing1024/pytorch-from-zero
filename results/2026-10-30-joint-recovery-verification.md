# Joint shuffled training recovery: verified local scope

Date: 2026-10-30 (Asia/Kuala_Lumpur). Executed by the AI assistant; learner
independent mastery remains unassessed. Public baseline:
`d6da22b470d44f5299e60733b629daa4b20060b8` (271 files, every restored blob verified).
This record reports local verification; a new remote commit/CI is not inferred.

## Actual checks

| Check | Result |
|---|---|
| Main source suite | **1,183 passed**, 307.40 seconds |
| Standalone first-step suite | **59 passed**, 5.13 seconds |
| Standalone saved-forward suite | **45 passed**, 4.86 seconds |
| Standalone delivery suite | **34 passed**, 0.996 seconds |
| Total distinct test cases in those suites | **1,321 passed** |
| New joint tests, included above | 79 contract + 12 CLI + 30 independent = **121** |
| Ruff lint / format | Passed; 116 files formatted correctly (includes Markdown code blocks) |
| Source environment pip check / compileall | Passed |
| New source joint recovery CLI | **12 fault/recovery cases passed** |
| Fresh noneditable wheel | **All 12 installed CLIs passed** |

The three pytest suites each emitted the known optional NumPy-unavailable warning;
these tensor-only checks passed without NumPy. Timings describe this run, not a
performance comparison. Focused reruns are not added to suite totals.

The source and wheel environments used Python 3.12.14 and PyTorch 2.14.1+cpu.
Both were isolated virtual environments with the unchanged ten-wheel CPU runtime
installed offline after verifying every wheel against the existing official-index
hash locks. Source development tooling reused the existing local pytest 9.1.1 /
Ruff 0.16.10 installation; no fresh download of developer tools is claimed.

The installed-wheel check ran from an unrelated directory under Python `-I`.
The package was confirmed noneditable and imported from that fresh environment's
site-packages. All **33** package members matched both current source and installed
bytes. `pip check` passed. An initial wheel caught a concurrent docstring edit in
the source-byte comparison and was superseded before CLI checking; only the rebuilt,
byte-verified final wheel is reported. Its SHA-256 is
`f6d39f01a4853ada0f2e0cc20f78682022120b780e84a56355b0ce7fad6ad344`.
See the [compact wheel summary](2026-10-30-installed-wheel-summary.json) for hashes
and the twelve named results. Bulky reports and local paths are not published.

## Recovery result and controls

The [source experiment](2026-10-30-joint-recovery.json) uses fixture seeds 42/7/123,
shuffle seed 19, and four predeclared failure stages per seed. Its **12 cases**
run across three baseline, three fault and three recovery workers. Every one of
the **147 remaining safe-point comparisons** has the same full typed-state digest
as uninterrupted execution; committed batch-index suffixes also match exactly.
All final parameter maximum absolute differences are **0**. Each baseline
completes three epochs, 15 microbatches and 69 sample visits, with every index
exactly once in each committed epoch.

Fault-stage observations distinguish an advanced issuance cursor from unchanged
committed training progress. Fetch has not changed model/gradients; backward has
changed gradients; update and precommit cases have changed model parameters.
The failed owner refuses snapshot, save, step and train. Previous checkpoint
bytes remain unchanged, unsafe output is absent, and a fresh recovery process
replays the attempted batch from that whole previous snapshot. Fault workers
exit with the declared nonzero status after their checks, not through SIGKILL.

The [independent review](2026-10-30-joint-recovery-independent-review.json) covers
21 configurations / 198 mathematical safe points using explicit affine derivatives
and handwritten momentum/weight-decay equations (`atol=rtol=1e-12`). It separately
checks exact same-runtime replay, six actual mutate-then-raise failures including
`KeyboardInterrupt`, and an isolated fresh-process disk recovery. A stale stream
transplant is rejected. A plausible stale-model transplant with recomputed
checksums is structurally accepted but fails the independent numerical oracle:
hashes are not signatures or proof of training history. These review counts are
not extra pytest tests beyond the 30 included in the main suite.

## Reproduction

From the installed source checkout in the documented CPU environment:

```bash
python -m pytest -q
(cd experiments/wine-first-step-2026-10-13 && python -m pytest -q)
(cd experiments/wine-saved-forward-2026-10-14 && python -m pytest -q)
(cd tools/wine-delivery && python -m unittest discover -s tests -v)
python -I -m pytorch_lab.joint_recovery_cli --output results/local-joint-recovery.json
ruff check --extend-exclude experiments/wine-saved-forward-2026-10-14/docs/source-before-first-capture.py .
ruff format --check --extend-exclude experiments/wine-saved-forward-2026-10-14/docs/source-before-first-capture.py .
```

For the clean wheel workflow, follow [runtime reproducibility](../docs/runtime-reproducibility.md),
build/install the current wheel noneditably in a fresh hash-locked environment,
then run `scripts/check_installed.py` with that interpreter under `-I` from outside
the checkout. Do not reuse an old editable installation for isolated CLI claims.

## Preserved boundaries and remaining work

- Historical input fixtures, results, runtime locks, model science, strict hashes
  and 8-ULP thresholds are unchanged. The previous eleven installed CLI entries
  keep their order; joint recovery is twelfth. Exact inventory tests alone were
  adjusted accordingly. Source/container CI budgets remain 15/20 minutes and
  per-worker/checker limits remain unchanged.
- The new source CLI and the twelve installed-wheel CLIs were run locally.
  No local Docker or full pytest run inside the wheel-only environment is claimed.
  A future exact-commit CI run must verify actual container execution separately.
- Recovery means discard the failed owner and load a prior jointly saved safe
  state. Unsaved computation may run again. No exactly-once external side effects,
  arbitrary asynchronous commit/return interruption, process-kill guarantee or
  durable power-loss transaction is claimed.
- Fixed single-owner CPU float64 affine training only; no arbitrary loader/model,
  workers, prefetch, random transforms, GPU, AMP, scheduler or DDP. Cross-version,
  cross-device and cross-platform snapshot exchange were not tested.
- Only trusted own checkpoints. Size/schema/checksum checks and restricted loading
  are not a malicious-file sandbox or authentication mechanism.
- Next learning step: independently explain the failure trace and reconstruct a
  safe-point state without the assistant. Any later stochastic-forward extension
  should first specify which additional RNG state belongs in the same commit.
