# Independent fixed-batch diagnosis review

The predeclared plan was read before review execution. The production implementation
was not modified. This review is separate from the main regression and installed CLI
runs; its 28 tests belong inside the eventual main suite count, not in addition.

## Actual checks

- `tests/test_training_diagnostics_independent.py`: 28 passed in 2.02 seconds; the post-format rerun also passed all 28 in 1.96 seconds.
- Ruff formatting and lint passed after formatting the new test file.
- Nine configurations (three learning rates × three paths), 108 steps, compared
  against independent scalar closed-form weights, bias, MSE, gradients and SGD
  updates at the declared `atol=rtol=1e-12`. The fixture's mean x is zero and mean
  squared x is 0.5. Thus loss is `0.5*(w-2)^2+(b-0.5)^2`, gradients are `w-2` and
  `2*b-1`, and the correct path has `w_k=2*(1-(1-lr)^k)` and
  `b_k=0.5*(1-(1-2*lr)^k)`. No maximum-error measurement was claimed.
- The diagnosis signature accepts only measured counts and norms, not fault names.
  Replacing SGD.step with a no-op while running the path named correct yields
  gradient_without_update. Replacing step with a genuine parameter mutation while
  omitting backward yields missing_gradients_with_update. This checks that labels
  follow actual parameter changes rather than the selected intervention name.
- Repeated no-op steps preserve the same per-step gradient instead of accumulating
  stale gradients. Early recorded parameter values and gradients stay unchanged
  after later updates, checking cloned measurements rather than aliased storage.
- All eight observational classes were checked, including partial gradients and
  zero gradients with an update. Nonzero gradients with no update remain a symptom,
  not proof of omitted optimizer.step; zero learning rate, optimizer policy and
  floating-point rounding can produce similar observations outside this fixture.
- NaN, infinity, negative, boolean and string norm values were rejected; absent
  gradients with a positive gradient norm were also rejected.
- Two full experiment executions matched exactly and left the default CPU RNG
  unchanged. This is same-runtime repeatability, not cross-platform reproducibility.

The run emitted the existing PyTorch warning that NumPy is absent; these tests use
Torch and Python scalar math. There were no failed tests or production corrections
in this independent run. No model convergence-rate, diagnosis accuracy across
arbitrary faults, performance, GPU or distributed-training claim is made.
