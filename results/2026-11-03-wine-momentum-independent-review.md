# Independent review: Wine paired momentum experiment

Date: 2026-11-03 (Asia/Kuala_Lumpur). Scope: fixed two-epoch CPU float64 experiment, not optimizer selection or generalization assessment.

## Independent execution

`python -m pytest tests/test_wine_momentum_independent.py -q` completed with **7 passed in 2.36 seconds**, plus the existing optional NumPy initialization warning. Ruff format and lint passed for this file. This is a focused review run, not the full repository test suite; its seven tests are included in the later aggregate and must not be counted twice.

The six architecture/seed cases instrument actual `torch.optim.SGD.step` calls. They clone parameters and raw gradients before each update and parameters after it. An independent recurrence reconstructs coupled weight decay and momentum from these snapshots, not from production success flags. Every actual update and momentum buffer passes `atol=rtol=1e-14`. Plain SGD has no momentum buffer. Each arm executes 14 optimizer updates.

For all six pairs:

- First preparameters, raw gradients, deltas and postparameters are exactly equal.
- The second update starts from exactly equal parameters and raw gradients. Its measured delta difference agrees with `-0.05 * 0.9 * first_effective_gradient` at the independent tolerance, and postparameters differ.
- Third-update raw gradients differ. Later gradients are **not** a controlled equal input, because the parameter trajectories have separated.
- Reported raw-gradient norms and actual-delta norms equal the independent snapshot measurements. Reported paired parameter distance equals the directly measured postparameter difference. First-step distance is exactly zero.
- Both arm models are compared against independent runs of the original, uninstrumented `wine.fit_candidate` at their respective momentum setting. Final tensors, online/final training history, 14 updates and 214 training visits match exactly. This reference checks the shared loop did not silently change either original training trajectory.
- Each epoch uses six 16-row batches and one 11-row tail, preserving sample weighting rather than averaging seven unequal batch means.

The seventh test replaces held-out features with NaN and targets with -100, forbids split regeneration and the held-out scoring helper, and guards every classifier forward to accept only standardized training rows. The complete experiment preserves preprocessing, global CPU RNG and caller thread count. Direct counters measure **168 optimizer updates, 2,568 training sample visits and 24 full-training evaluation forwards**. Evaluation forwards are separately counted, not optimizer sample visits. These counts cover the experiment itself; independent reference reruns and other tests incur additional work.

## Numerical evidence inspected

Frozen plan SHA-256: `ce5abc56ef68aa4d6fd622d01d6c42f4c4e9bac1fd787f7d508a2e2fcbc8c421`.

Raw result SHA-256: `582162de5a620811b02a34c0f83bb18ef53fa4c201897262fe2191cf46ca5c4b`.

All six pairs in the raw result pass the declared controls. Maximum reported actual-update equation residual is `6.245004513516506e-17`. Momentum yields lower final training CE in all six fixed two-epoch pairs; this is descriptive evidence at one nominal learning rate and budget, not a performance threshold in the tests.

## Review correction and limits

Review found that the initial paired-distance expression reconstructed postparameters using a different subtraction order. It was changed to measure the actual post-update tensors directly before final result generation. The preliminary output was retained outside the published evidence; no learning rate, budget, scientific tolerance or historical result was changed to pass review.

CLI review covers existing-file/directory/live/dangling-symlink refusal before execution, invalid arguments, exact success flag, non-finite JSON rejection, experiment failure and exclusive-create race protection. Source and installed command lists retain the earlier 18 commands and append this experiment as the nineteenth. Actual aggregate source, installed-wheel and remote container execution are reported separately, not inferred from this inspection.

Equal nominal learning rate does not mean equal effective displacement. There is no matched-effective-step-size experiment, converged optimum, validation/test improvement, GPU/AMP/distributed test or observation-overhead benchmark here. The inherited whole-repository regression may separately execute the historical Wine benchmark; its held-out results do not select or tune this experiment.

## Official semantic reference

The versioned [PyTorch 2.14 SGD documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html) specifies the order of decay and momentum and the first-buffer initialization. The tests and explanation above are original derivations, not copied implementation. PyTorch's BSD-style license and fixed source references are recorded in the accompanying experiment documentation.
