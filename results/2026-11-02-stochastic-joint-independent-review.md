# Independent stochastic joint-training review

Date: 2026-11-02 (Asia/Kuala_Lumpur). This report concerns assistant-built artifacts, not demonstrated learner mastery.

## Actual execution

Command, from the engineering repository with the source installation active:

```sh
python -m pytest tests/test_stochastic_joint_independent.py -q
```

Final result: **32 passed**, 1 warning, 13.13 seconds. Ruff checks passed; the test file was formatted. The warning says PyTorch could not initialize NumPy because NumPy is absent in this interpreter; these tests use Torch tensors and do not use NumPy. This focused result is not a full regression, wheel or Docker result.

## Independent equations

18 configurations cover model RNG seeds 7/42/123, Dropout probabilities 0.0/0.35/1.0 and sample/microbatch/accumulation settings 23/5/3 and 7/3/2. Two epochs per configuration give 144 equation-checked microbatch steps.

The oracle uses the actual dropped inputs and fixture targets, then explicitly computes affine predictions, summed squared loss, weight and bias gradients, pending raw gradient sums, actual-element normalization, weight decay, momentum and SGD updates without autograd or torch.optim for the oracle. Parameters, pending gradients, momentum and loss satisfy the predeclared atol=rtol=1e-12 gate. Independence is of differentiation and optimizer equations, not the PyTorch Dropout algorithm, data fixture or tensor arithmetic kernels. No maximum-error measurement is claimed.

Every oracle step also checks snapshot restoration exactly by canonical content digest and compares the next step of two restored owners exactly, inserting external default-RNG sampling between their operations. These equality checks have no numerical tolerance and are separate from the mathematical oracle. They are same-process checks; independent-process evidence belongs to the separate CLI experiment.

At p=1.0, the zero-initialized weight remains zero in this fixture while the bias becomes nonzero. This is a tested fixture outcome, not a universal zero-update statement.

## State-boundary tests

- Four genuine operation-then-raise cases: Dropout forward, backward, optimizer step and epoch advance. Each executes the real operation before raising. Caller RNG is unchanged; the failed owner rejects continuation, snapshot and saving. Restoring an earlier whole snapshot produces the same next safe-point state as a separately restored control.
- Constructor, successful step, checkpoint save/load and rejected restoration preserve caller CPU RNG.
- Eight altered snapshots with recomputed checksums are rejected: invalid RNG bytes, inconsistent pending denominator, non-boolean training mode, finite out-of-range Dropout probability, reversed optimizer parameter order, non-finite pending gradient, boolean schema version and wrong architecture.
- An invalid after_update precondition on a non-update step leaves the original safe snapshot unchanged and allows a later valid step.

## Failure record and limits

The initial run had 31 passes and one test-construction failure: a scalar NaN probability cannot be rehashed by the existing strict JSON canonicalizer. The malformed-state test was corrected to a finite out-of-range probability (1.1), so it reaches the intended semantic validation. No production code or acceptance tolerance changed. The final full focused rerun passed all 32.

No production changes were made by this review. There is no evidence here for concurrent execution, arbitrary modules, GPU, process killing, power loss, cross-runtime snapshot interchange or external exactly-once effects. Final full-suite and publication evidence must be recorded separately for the final source tree.
