# Wine batch grouping under two budgets

This original experiment asks two distinct questions rather than ranking batch
sizes without a resource contract. The frozen [plan](wine-batch-budget-plan.json)
was written before outcomes. No heldout scores select settings or winners.

## Design

Use pinned107 training rows, train-only standardization, linear/MLP models and
seeds42/7/123. SGD lr0.05,momentum0.9,decay1e-4; CPU float64 singlethread.
Four explicit permutations from a local generator are shared across all arms.
Batch grouping does not consume extra RNG draws. This ordering deliberately
is not asserted equal to historical DataLoader trajectories with the same seed.

Each configuration runs three unique trajectories:

- batch16,14updates,214sample visits: shared baseline for both comparisons.
- batch32,2epochs,8updates,214visits: equal sample visits.
- batch32,14updates,385visits: equal updates, not equal data exposure.

Epoch tails are retained (11rows) and batches never cross epoch boundaries.
The baseline is executed once, not twice. Sixconfigurations total216updates and
4878training sample visits. One initial fulltrain CE per configuration plus one
endpoint per trajectory adds24evaluation forwards/2568evaluation sample visits.
OnlineCE weights pre-update batch means by actual sample counts and combines
changing parameter states; endpointCE uses one final state. They are not identical
statistics. Trace records training positions, batch sizes, actual parameter delta
norms and post-update hashes.

## Official research and original implementation

[PyTorch2.14 DataLoader](https://docs.pytorch.org/docs/2.14/data.html)
explains samplers, batching and drop_last. We use explicit permutations to make
order invariant auditable instead of inferring it from equal seeds.
[PyTorch2.14 SGD](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html)
and [fixed official source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/optim/sgd.py)
explain per-update momentum and coupled weight decay. Upstream uses its
[BSD-style license](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE).
All experiment/control code is original; no upstream code copied. Bundled UCI
Wine data retains its CC BY4.0 attribution and unchanged data/split hashes.

## Reproduce

After the root README installation:

```bash
python -m pytorch_lab.wine_batch_budget_cli --output results/my-batch-budget.json
```

Choose a new file; existing outputs and symlinks are refused. Scientific checks
and JSON finiteness are checked, with nonzero exit on failed controls. The CLI
is also executed as the21st source and installed-package delivery command.

## Actual outcome

[Raw evidence](../results/2026-11-04-wine-batch-budget.json):

| Model | Seed | batch16,214visits/14steps | batch32,214visits/8steps | batch32,385visits/14steps |
|---|---:|---:|---:|---:|
|linear|42|0.089011770|0.192385676|0.092114797|
|linear|7|0.110586373|0.220041389|0.100067593|
|linear|123|0.100532319|0.200448639|0.096599128|
|mlp16|42|0.263910925|0.669581090|0.262344683|
|mlp16|7|0.261625596|0.607787215|0.259725422|
|mlp16|123|0.201472918|0.625797923|0.212725598|

Entries are endpoint trainingCE. Batch16 is lower in all six equal-visits
comparisons. Equal-update outcomes are mixed: batch32 lower in four of six, while
using more samples. No optimization settings or endpoint were changed in response.
All five per-configuration controls pass. Independent tests reconstruct actual SGD
updates with real row sequences and actual tail denominators, verify endpoints
and report hashes, and guard poisoned heldout inputs; see
[independent review](../results/2026-11-04-batch-budget-independent-review.md).

## Limits

Neither comparison equalizes wall-clock compute. Momentum and decay apply per
update, so changes are not attributable purely to gradient noise. These short CPU
trajectories do not establish convergence, generalization or optimal batch size.
The legacy delivery suite separately reruns heldout benchmark scores; those
outputs are not inputs to this experiment. The next useful training question is a
predeclared learning-rate schedule/control under an explicit update clock, not
an unbounded batch-size grid or post-hoc learning-rate tuning.
