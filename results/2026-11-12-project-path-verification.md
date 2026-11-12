# Project-path integration verification, 2026-11-12

Base: remote engineering commit `21d5997d51b8668a8952065b77ec17be140ffe96`.
The separate learning repository was checked at `5b569329f19553d757d2e5d7022d61946f90663e`.
An independent working copy was reconstructed and every input blob checked against
the remote tree. Old local working directories were left unchanged.

## Delivered behavior

- The homepage now starts with installation and one runnable path rather than a
  dated experiment feed. The previous README is preserved byte-for-byte as
  `README.history.md` at the same directory level, so its relative links still work.
- The new `wine_pipeline_cli` composes existing components, trains once, evaluates
  the same state on reused validation, exports, then predicts in another process.
  No new model, optimizer variant, sweep, source dataset or prediction service.
- [Raw source report](2026-11-12-project-path.json) and the accompanying
  [five real artifacts](2026-11-12-project-path-artifacts/) contain the execution.
  Training: 140 updates / 2,140 sample visits. Prediction: 36/36 successful rows.
  All five artifact/prediction consistency checks passed.
- Validation CE 0.05811863818000037, accuracy 35/36, macro-F1 0.974320987654321.
  This validation has already been used for historical selection. These are
  descriptive scores for the fixed existing model, not fresh generalization,
  improved accuracy, or a pass/fail quality threshold. This new path scores zero
  test rows. Existing regression/legacy CLI checks can rerun historical benchmarks
  with their original test scores; those outputs do not guide this path or tuning.

## Verification scope

- The seven new integration tests pass, including actual subprocess prediction,
  training-only preprocessing, validation row identity, file digests, refusal to
  overwrite old/partial runs and failure without a success report.
- Full source regression and its actual counts are recorded in
  [structured checks](2026-11-12-project-path-checks.json).
- Standalone suites: 59 pytest + 45 pytest + 34 unittest passed. They are separate
  from the main suite, not duplicate runs counted as new tests.
- Ruff check and format check passed using the existing archived-file exclusion.
- A fresh virtual environment was created from the existing official runtime
  wheels with the repository's complete hash lock. A new non-editable wheel was
  built offline, installed and checked using isolated Python (`-I`) outside the
  source tree. `pip check` passed. The full new path passed all five checks and
  produced the same model digest as source. [Installed evidence](2026-11-12-project-path-wheel.json).
- The historical 26 installed CLIs were not rerun locally in this integration.
  CI retains those existing commands and adds the new path in both source and
  the nonroot, read-only, network-disabled CPU container. Container execution is
  not claimed locally because no Docker executable is available here.
- Python 3.12.14 / PyTorch 2.14.1+cpu / Linux x86_64; optional NumPy is absent,
  producing the existing initialization warning. No NumPy interop is used.

The first full-suite attempt without explicit thread limits was interrupted before
completion due to slow progress, without a completed pass/fail claim. The final
whole-suite run uses OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1.
One focused rerun is already included in the final suite and is not counted twice.
These timings are test-run metadata, not performance benchmarks; other task work
may share the host.

The new container step retains its JSON summary only; its temporary intermediate
files disappear when the container exits. The source CI artifact includes all five
intermediate files. Static review found no blocking issue; it did not rerun tests.

## Publication and limits

The exact commit, remote content and CI are checked after publication and recorded
in the learning repository's November 12 daily report. Local checks alone do not
establish a successful remote CI or container run.

Original numerical evidence, failures, protocols, data/license notices and archived
history are retained. The current batch-inference next-step guidance points to this
integration and engineering criteria, without making personal exercises a gate.
No learner-mastery, production, GPU/distributed, authenticity/security-sandbox,
cross-hardware bitwise reproducibility or service latency claim is added.
