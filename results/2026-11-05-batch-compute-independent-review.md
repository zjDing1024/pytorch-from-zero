# Independent review: prepared-batch CPU timing

Fresh execution on 2026-11-04 UTC (2026-11-05 Malaysia reporting date).
Four tests in `tests/test_wine_batch_compute_independent.py` passed in 5.11 s.
Ruff lint and format pass. The optional missing-NumPy warning remains.

## What was directly checked

- Three arm-specific tests replace both clocks with deterministic call spies,
  while actually running model forward, backward and SGD. They verify clock
  ordering wall-start, CPU-start, CPU-end, wall-end; each optimizer step and
  training forward is inside the interval; final evaluation and digest are
  outside. Synthetic durations are exactly 1,000 wall ns and 700 CPU ns. These
  are test values, not performance measurements.
- A complete real experiment poisons all heldout features/labels and guards
  every forward against the standardized training rows. Actual calls count
  1,080 optimizer updates, 24,390 training sample visits and 90 final training
  evaluation forwards. This includes reference, warmup and measured paths.
- Global CPU RNG and caller thread count remain unchanged. Each measured arm
  occurs once in each ordering position; reported medians are recomputed from
  the three measured rows, excluding warmup. Each arm's repeated final parameter
  digests agree, and the experiment's exact reference checks pass.

The guard-instrumented test adds overhead and is not used for runtime results.
The production timer excludes preparation/indexing, model construction, final
scoring and hashing, but includes Python iteration and clock-read overhead.
Wall and process CPU are different quantities; neither is a GPU measurement.
Three within-process samples provide descriptive dispersion, not confidence
intervals or universal speed ranking. No test asserts an arm must be faster.

An initial lint pass reported import ordering and long lines in this independent
file. Formatting corrected those issues and the final tests were rerun. No
scientific thresholds, budgets or production implementation were changed.
