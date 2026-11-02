# Independent learning-rate stability review

Date: 2026-11-02, Asia/Kuala_Lumpur. Scope: fixed five-row CPU float64 affine MSE; frozen plan read before review. Production implementation and plan were not changed by this reviewer.

## Executed checks

`tests/test_lr_stability_independent.py`: **26 passed**, 2.19 seconds in the final focused run. Ruff lint and format pass. These tests belong to the main suite; do not add their count again to aggregate tests.

- Eight learning rates (0, .1, .25, .5, .75, 1, 1.25, 1.5), twelve steps each: 96 closed-form comparisons. Independent equations check both parameters, both gradients, actual deltas, parameter errors, pre/post MSE and gradient/update norms with the frozen atol=rtol=1e-12.
- At lr=1, loss drops from 2.25 to .25 at the first step, then remains .25 while bias alternates between 1 and 0. Weight reaches 2 at the first update. This is not a constant-loss claim from initial state.
- At lr=1.25, loss initially decreases despite an unstable bias direction, then exceeds initial loss. Divergence does not require monotonically increasing total loss from step zero.
- At lr=0, gradients remain nonzero but parameters do not move.
- The 32-step .75 case repeats identically and preserves caller CPU RNG. Invalid rates and oracle steps are rejected.
- Replacing the real optimizer step with a no-op makes the experiment fail and leaves raw parameters at zero while the oracle differs: measured traces are not overwritten by analytical answers.
- An actual optimizer mutation to infinite weight is rejected as non-finite.

## Exploratory stress is a separate result

The valid finite configuration lr=1.5, 32 steps is outside the primary four-rate twelve-step experiment. On this execution it reports `all_oracle_matches=false`, with mismatches at steps 28, 29, 31, 32. Final measured parameters are approximately [1.9999999981373549, -2147483647.500001], while the closed-form values are [1.9999999995343387, -2147483647.5]. The small weight component exceeds the unchanged comparison tolerance during the large-bias trajectory. Cancellation is a plausible mechanism, not a separately established kernel-level cause.

The portable regression test independently recomputes every row's comparison decision at the original tolerance and checks that the reported flags agree. It does not demand that every platform reproduce the same floating-point failure, and it does not turn this local negative result into a passing numerical claim. A valid configuration means bounded finite execution is accepted, not that every admissible trajectory must pass the oracle. The primary frozen experiment still reports its own separate gate.

## Review process and limits

Initial 24-test focused run passed; added non-finite injection and stress-report checks brought the final count to 26. One lint-only missing explicit zip strict argument was corrected before final execution; no numerical test, production behavior or tolerance was altered to obtain a pass. Existing absent-NumPy warning was observed; this review uses Torch/scalar Python, not NumPy.

No arbitrary-model stability guarantee, universal learning-rate recommendation, GPU/AMP/DDP result or cross-runtime equality is established. Full regression, installed wheel and exact-commit CI are separate evidence, not implied by this focused review.
