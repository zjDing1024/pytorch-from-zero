# Wine: paired momentum at a fixed nominal learning rate

## Question and scope

On the pinned 107-row training partition, how does SGD momentum 0.9 change
optimization versus momentum 0 over a fixed two-epoch prefix? This is a matched
initialization, batch-order and update-budget intervention, not an optimizer
leaderboard. The learning rate is equal; effective displacement is not equal.
The protocol was frozen in [wine-momentum-plan.json](wine-momentum-plan.json)
before observing these results (SHA-256
`ce5abc56ef68aa4d6fd622d01d6c42f4c4e9bac1fd787f7d508a2e2fcbc8c421`).

## Original implementation and official sources

Study [PyTorch SGD documentation, version 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html)
and the fixed [official SGD source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/optim/sgd.py).
PyTorch is maintained by its team and contributors under its
[BSD-style license](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE).
We derived these checks and wrote the paired loop ourselves; no upstream code was
copied. Reuse our pinned `prepare_train` and digest helper, leaving historical
Wine training and observation code unchanged. UCI Wine retains its bundled
CC BY 4.0 attribution and data/split hashes.

For raw batch gradient g and weight decay λ, effective gradient is q=g+λθ.
With default dampening 0 and no Nesterov, momentum starts with b₁=q₁ and then
bₜ=μbₜ₋₁+qₜ. The actual update is −ηb (or −ηq without momentum).
Both arms therefore share the first step. At step 2 they still share the
pre-update parameters and raw gradient, so their delta difference is −ημq₁.
After that their parameter trajectories and gradients generally differ; later
loss differences cannot be decomposed as a pure same-gradient momentum effect.

## Design and cost

- Linear 13→3 and MLP 13→16→3; seeds 42, 7, 123.
- Existing train-only standardization, float64 CPU, one thread.
- Momentum 0 and 0.9; lr 0.05, decay 1e-4, batch 16 with 11-row tail.
- One common DataLoader stream per pair; separately initialized equal models and
  distinct real SGD optimizers. No heldout forward, scoring or parameter selection.
- Each arm: 14 optimizer updates and 214 training sample visits. Six pairs:
  **168 updates / 2,568 sample visits**. Epoch-end full training CE adds
  **24 evaluation forwards / 2,568 evaluation sample visits**, not optimizer visits.
- No initial full-partition evaluation or audit replay inside this report.
  Independent tests and other delivery checks have their own additional costs.

Each step records actual cloned parameter delta norms, raw/effective gradient
norms, momentum buffer state, cumulative path length, and actual post-update
paired parameter distance. Raw audit hashes identify parameter/gradient states.
No-momentum buffer is absent (`null`), not a fabricated zero-valued momentum.
An independent optimizer-step wrapper checks these measurements with captured
real tensors; both arms also match the original uninstrumented trainer exactly.

## Reproduction

Follow the root README installation steps, then run:

```bash
python -m pytorch_lab.wine_momentum_cli --output results/my-wine-momentum.json
```

The output must be a new path; existing files/directories/symlinks are refused.
JSON rejects NaN/Infinity. Failed scientific checks return nonzero while retaining
structured evidence. The nineteenth installed-package CLI is included in both
source CI and the existing offline/read-only CPU container delivery check.

## Actual result

[Full raw result](../results/2026-11-03-wine-momentum.json) records all 84 paired
batches / 168 arm updates. Eight controls pass in all six pairs. Final training
CE after exactly two epochs:

| Model | Seed | Momentum 0 | Momentum 0.9 |
|---|---:|---:|---:|
| linear | 42 | 0.475531263 | 0.091532547 |
| linear | 7 | 0.496512743 | 0.110625801 |
| linear | 123 | 0.442215929 | 0.094449431 |
| mlp16 | 42 | 0.866991545 | 0.274933430 |
| mlp16 | 7 | 0.805984798 | 0.273297940 |
| mlp16 | 123 | 0.814937705 | 0.200184599 |

Momentum 0.9 reaches lower training CE in this fixed short experiment; it also
changes displacement and accumulated path length. This is not evidence of better
generalization, optimal hyperparameters, convergence, or a universal ranking.
Online CE mixes pre-update models and is sample-weighted; epoch-final CE uses one
common final state per arm and is a different statistic. We do not tune learning
rates to equalize outcomes or use historical test scores to choose a winner.

## Limits and next useful question

This small float64 CPU task does not cover GPU, AMP, stateful/stochastic layers,
distributed optimization or monitoring overhead. The existing delivery suite
separately reruns the historical full Wine benchmark, including heldout scores;
those outputs are not inputs to this experiment. The next bounded question is minibatch gradient estimation/noise at a fixed
parameter state, with a predeclared budget of zero optimizer updates. Compare
minibatch gradients against the full training gradient to separate sampling
noise from changing parameter trajectories; do not tune on heldout scores.

## Verification

[Local aggregate and wheel evidence](../results/2026-11-03-wine-momentum-verification.md)
and [independent review](../results/2026-11-03-wine-momentum-independent-review.md)
separate scientific controls, test assertions, and actual installed CLI execution.
