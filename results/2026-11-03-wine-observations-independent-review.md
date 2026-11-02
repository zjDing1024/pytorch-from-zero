# Wine training observation: independent review

Date: 2026-11-03 (UTC+08:00). Scope: frozen two-epoch training-only prefix.

## Evidence

- Frozen plan SHA256 checked before review: `a13ef89e592f988493cd8c8276170fa2de5af8e1e69d7e5848de26a334a025dc`.
- `python -m pytest -q tests/test_wine_observations_independent.py`: **7 passed**, 1 warning, 2.61 s.
- `python -m ruff check tests/test_wine_observations_independent.py`: passed; Ruff formatting applied.
- Warning: installed Torch cannot initialize optional NumPy because NumPy is absent. These tests use native tensors and do not require NumPy.

## Independently checked

Six architecture/seed configurations each execute 14 optimizer steps. The reviewer wraps real `SGD.step`, clones pre-step parameters and raw gradients, and independently computes effective gradient `g + 0.0001 * theta`, initial momentum buffer, subsequent `0.9 * previous_buffer + effective_gradient`, and `theta - 0.05 * buffer`. Actual post-step parameters and buffers agree at absolute/relative tolerance 1e-14. The oracle does not call the observation helper or use its reconstruction result.

Separate clones of actual state verify every reported raw-gradient norm, pre-step parameter norm, post-step momentum norm, actual parameter displacement, and displacement/parameter ratio. Initial biases have zero norm and a null ratio with explicit status. Actual deltas are not replaced by learning-rate-times-raw-gradient.

For all six configurations, observed and diagnostics-disabled audit trajectories agree exactly, and final state tensors and full history/budget dictionaries match the original `wine.fit_candidate` called without the diagnostics wrapper. Diagnostics-disabled still performs audit hashing; only the separate original implementation is the uninstrumented reference.

Each epoch has six 16-row batches and an 11-row tail. Online cross-entropy is checked using sample-weighted sequential accumulation over 107 rows, not an unweighted average of seven batch means.

The isolation test checks scaler mean and population standard deviation directly against the pinned training rows. All validation/test features are replaced with NaN and labels with -100 after loading; training features, labels, and preprocessing remain identical. During the full experiment, guarded forward calls require every row to match a standardized training row; evaluation and fresh split-generation entry points are forbidden. Actual runtime counters, rather than report constants, record **252 updates, 3,852 training sample visits, and 36 full-training epoch evaluations**. These are three passes across six configurations; epoch evaluation forwards are not optimizer visits. Global CPU RNG state is unchanged.

## Review corrections and limits

No production defect was found and no scientific configuration or historical result was changed. The initial reviewer test incorrectly recognized bias names by a `bias` suffix rather than the model's `biases.*` names. Further test-only corrections used sequential accumulation (Python float `sum` has different rounding) and actual momentum snapshots for exact measurement equality; independent mathematical reconstruction retains its original 1e-14 tolerance. These were test-oracle issues, not evidence to change experimental settings.

This proves the stated execution invariants in the tested local environment. It does not establish cross-platform bitwise reproducibility, complete convergence, validation/test accuracy, GPU behavior, instrumentation runtime overhead, or personal skill mastery. The two-epoch prefix is not the historical 120-epoch benchmark.
