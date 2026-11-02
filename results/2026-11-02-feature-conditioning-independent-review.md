# Independent feature-conditioning review

Date: 2026-11-02 (Asia/Kuala_Lumpur). The frozen
`docs/feature-conditioning-plan.json` was read before this review's execution.
No production code, declared tolerance, or primary experimental configuration was
changed by the reviewer.

## Actual checks

`tests/test_feature_conditioning_independent.py` passed all **33 tests in 2.69 seconds**.
Ruff lint and format checks passed. The existing missing-NumPy warning was emitted;
the checks use PyTorch tensors and Python scalar arithmetic, not NumPy.

- Eighteen configurations (six scales 0.25, 0.5, 1, 1.3, 2, 4 and three policies)
  each ran 12 steps, giving 216 independently checked microsteps. Python scalar
  closed forms check canonical parameters, transformed parameters, coordinate
  and canonical gradients, loss, parameter deltas, predictions and prediction
  update RMS. The declared absolute and relative tolerances remain 1e-12.
- Three separate autograd Hessian calculations evaluate the actual five-row
  loss at a nonzero parameter point and compare against diag(scale², 2).
  Condition numbers and scalar stability ceilings are checked independently.
- Mapped diagonal learning rates preserve canonical states, predictions and
  prediction-update RMS within tolerance, including non-power-of-two scale 1.3.
  The transformed parameter values intentionally differ; raw parameter norms
  are not treated as a comparable cross-coordinate performance metric.
- Equal initial loss does not imply equal scalar-SGD trajectories. The test
  explicitly observes different canonical first updates across scales, and
  retains the counterexample that shrinking this feature to 0.25 worsens its
  coordinate condition number relative to scale 1.
- A real optimizer no-op intervention leaves all actual parameters and losses
  unchanged and causes oracle flags to fail. This checks that execution results
  are measured rather than replaced by the analytic oracle.
- A real optimizer intervention writing infinity is rejected as a nonfinite
  trajectory. Invalid finite/nonfinite scale inputs are rejected. Deterministic
  repeats and caller CPU RNG preservation are also checked.

The scalar oracle is independent of the production analytic-state helper, but
uses the same mathematical fixture. The Hessian check uses PyTorch autograd;
it is not an independent numerical kernel implementation. No maximum-error
measurement, wall-time speedup, generalization improvement, or bitwise
cross-coordinate equality is claimed.

## Outcome and scope

No production defect was found. There were no failed runs or threshold changes
in this independent review. The 33 tests are part of the project suite and must
not be added a second time to the final aggregate count. Full regression,
installed-wheel CLI checks and remote CI are separately evidenced by the
engineering verification and learning report.
