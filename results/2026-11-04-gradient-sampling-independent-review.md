# Independent review of reconstructed gradient sampling

This is a fresh review and execution on 2026-11-04, not recovered November 3 code or test evidence.

Command: `python -m pytest tests/test_wine_gradient_sampling_independent.py -q`

Result: **8 passed in 1.40 seconds**, with the existing optional NumPy-unavailable Torch warning. Ruff format and lint passed on this test file. These eight tests belong inside the aggregate suite and must not be added twice. The focused run is not a full-project pass.

## Independent checks

- Four exhaustive finite-population tests enumerate every subset of sizes 1, 2, 3 and 4 from a fixed four-row gradient population. The exact mean of subset means equals the population mean, and the mean squared error matches the finite-population formula. Full-population sampling has zero error.
- A real Wine linear classifier's gradient is independently derived from softmax residuals: weight gradient is residual-transpose times input divided by sample count, and bias gradient is mean residual. It agrees with the produced gradient at `atol=rtol=1e-14`. Pre-existing `.grad` buffers are deliberately set to 3.25 and verified unchanged, along with parameters.
- Two tests capture every actual returned gradient for linear and MLP models. Each sampled batch gradient matches the mean of its corresponding per-example gradients at `1e-14`, its reported digest matches the actual tensor, and squared errors match direct measurement. The weighted partition, including the 11-row tail, matches the full gradient. The deliberately naive equal-batch average differs by more than 1e-3 in these two fixed cases; this is a diagnostic fixture assertion, not a universal lower bound.
- A full six-configuration run replaces heldout features with NaN and labels with -100 after data loading, forbids split regeneration and SGD steps, and guards every forward to ensure it contains only standardized training rows. It directly counts **1,266 gradient forward calls and 18,054 sample visits**, agreeing with reported budgets. CPU global RNG and calling thread count remain unchanged. All reported scientific controls pass.

## Interpretation boundaries

The exact finite-population expectation is checked by exhaustive enumeration on a small population; the Wine sampling experiment has finite draws and its empirical error ratio is descriptive, not a significance test or proof of bias. No optimizer update, convergence improvement or heldout generalization is established. This review covers CPU float64 fixed models without stochastic/stateful layers. It does not certify GPU, distributed or arbitrary model semantics.
