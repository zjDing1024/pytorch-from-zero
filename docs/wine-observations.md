# Wine training-only optimization observations

## Purpose and frozen protocol

Return from scalar examples to the existing real-data classifier. The question is
whether optimization telemetry measures actual SGD behavior without changing the
training trajectory, not whether a hyperparameter improves test accuracy.
The [plan](wine-observations-plan.json) was frozen before execution, SHA-256
`a13ef89e592f988493cd8c8276170fa2de5af8e1e69d7e5848de26a334a025dc`.
Historical `wine.py`, dataset, split, results and numerical thresholds are unchanged.

Reuse the packaged UCI Wine CSV integrity check and pinned 107 training IDs from
the original 107/36/35 split. Standardization uses only training population
statistics (`correction=0`). Parsing the bundled dataset is necessary to load it;
no validation/test example enters preprocessing, forward calls or scoring here.
The partition is not recalculated from changed held-out labels.

Retain `linear` (42 parameters), `mlp16` (275), seeds 42/7/123, local initialization
and DataLoader generators, batch size 16, shuffle, no workers, no drop-last, CPU
float64, one thread. SGD stays at learning rate 0.05, momentum 0.9 and coupled
weight decay 0.0001. No search, selection or retuning is performed.

The fixed **two-epoch prefix** has six 16-row batches and one 11-row tail per epoch:
14 updates / 214 visits per configuration. Six observed configurations total 84
updates / 1,284 visits. Two additional reference passes per configuration make
the actual experiment **252 updates / 3,852 training sample visits**. The original
120-epoch benchmark (840 updates / 12,840 visits per configuration) is not rerun
by this observation CLI. Separately, unchanged delivery checks still execute the
historical Wine benchmark, including held-out scoring. Those regression outputs
are not used to tune this probe.

```bash
python -m pytorch_lab.wine_observations_cli --output results/my-wine-observations.json
```

Use a new path. File and stdout agree. Nonfinite JSON and overwrites are rejected;
failed scientific checks produce a nonzero exit and retained failure report.

## Measurements and interpretation

Each named weight/bias tensor has raw gradient presence/L2 norm, pre-step
parameter L2 norm, actual cloned post-minus-pre parameter delta L2 norm,
delta-to-pre-parameter ratio, post-step momentum L2 norm, and the residual of
actual delta versus negative learning rate times momentum. Batch mean CE is
measured before the update. Ratios with zero denominators are null with an
explicit status, never epsilon-smoothed or silently zero. Initial biases naturally
have undefined ratios even when training is correct.

For raw CE gradient `g_t` and pre-step parameter `p_t`, coupled decay gives
`d_t = g_t + 0.0001 p_t`. The first buffer is `b_1 = d_1`; subsequently
`b_t = 0.9 b_(t-1) + d_t`; then `p_(t+1) = p_t - 0.05 b_t`.
Raw gradient norm therefore does not determine update norm by learning rate alone.
Norms depend on parameter coordinates and tensor sizes; they neither rank layer
quality nor uniquely establish a failure's cause.

Snapshots use `detach().clone()`: detach alone still aliases the in-place update.
Telemetry adds no backward call or random sampling. Epoch online CE is weighted
by actual sample count, including the tail. Epoch-end full-training CE describes
a different model/evaluation point. No monotonic minibatch-loss threshold is used.

## Noninterference

Enabled and disabled diagnostics share minimal audit hashing of batch tensors,
logits, loss, raw gradients, pre/post parameters, momentum and loader RNG. All
14-step traces must match exactly. Disabled diagnostics omit metric clones and
reductions; the common hash verifier is not a zero-overhead benchmark. A third,
genuinely uninstrumented `wine.fit_candidate(epochs=2)` must match final tensors,
histories, update counts and sample counts exactly.

This triangulates the new loop against legacy training, not arbitrary hooks,
concurrent code or devices. The report retains hashes rather than every tensor
element; fresh execution recomputes them. All 84 observed batches and 252
per-parameter measurement records remain available. SGD-delta reconstruction uses
the frozen atol=rtol=1e-12, separate from exact trajectory equality. It checks the
observed momentum buffer, not an independent proof of buffer construction;
independent tests reconstruct decay/momentum separately.

## Actual local results

[Raw report](../results/2026-11-03-wine-observations.json): all six runs pass
trajectory, legacy-reference, budget and reconstruction checks. The largest
actual-delta reconstruction residual is 6.245004513516506e-17.

| Model | Seed | Epoch 1 final train CE | Epoch 2 final train CE |
|---|---:|---:|---:|
| linear | 42 | 0.2341595792615103 | 0.09153254696650664 |
| linear | 7 | 0.27037404102211315 | 0.11062580053243792 |
| linear | 123 | 0.24248734290815616 | 0.09444943051791349 |
| mlp16 | 42 | 0.748265852805488 | 0.2749334296977856 |
| mlp16 | 7 | 0.680234255645997 | 0.2732979401873286 |
| mlp16 | 123 | 0.681939142080543 | 0.20018459862973095 |

These are prefix training losses, not validation selection or improved accuracy.
No full convergence, throughput, memory, GPU/AMP/DDP or personal mastery claim is
implied. [Verification](../results/2026-11-03-wine-observations-verification.md)
and [independent review](../results/2026-11-03-wine-observations-independent-review.md)
describe the actual acceptance scope.

## Official research and licenses

- [PyTorch](https://github.com/pytorch/pytorch), maintained by its open-source
  community: optimizer state transitions and tensor storage are directly relevant
  to this phase, unlike an unrelated popular application framework.
- [Pinned SGD source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/optim/sgd.py):
  gradient collection, coupled decay, momentum and final in-place update explain
  why raw gradients and actual deltas must be measured separately.
- [PyTorch 2.14 SGD](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html),
  [detach](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html),
  [clone](https://docs.pytorch.org/docs/2.14/generated/torch.clone.html).
- [Upstream BSD-style three-clause license](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE).
  This implementation is original; no upstream code is copied.
- [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine), CC BY 4.0; existing
  packaged attribution remains. Dataset bytes are not claimed as original work.

Next: predeclare momentum versus no momentum on the same Wine training partition,
with fixed initialization, batch order, learning rate and equal update budgets.
Freeze the causal question, training metrics and evaluation budget first; use
independent update equations. Do not select with validation scores, retune from
test scores, or claim either choice is universally better. This is future work.
