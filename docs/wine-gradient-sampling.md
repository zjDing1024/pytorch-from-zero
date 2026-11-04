# Fixed-parameter Wine gradient sampling, reconstructed November 4

The November 3 uncommitted source and raw evidence were unavailable after workspace
recovery. This is a newly implemented, newly executed replacement, not a recovery
of those original bytes. In particular the new protocol uses 32 draws per size,
so prior run counts and metrics must not be carried forward as this run's evidence.
The existing published data, model code, results and prior experiments are unchanged.

## Question and frozen protocol

At fixed initialization, does a sample-count-weighted partition reproduce the full
training mean-CE gradient? How does random subset gradient error compare with the
finite-population expectation, without moving the parameters?

[Plan](wine-gradient-sampling-plan.json) was written before the first run.
The pinned 107 training rows and train-fitted scaler are reused. Models are linear
13→3 and MLP13→16→3, seeds42/7/123, CPU float64 single thread, eval mode.
No optimizer, momentum, weight decay or parameter updates occur in this experiment.
The static partition has six16-row batches and one11-row tail. Independent random
subsets of sizes4/16/64 each have32draws, uniform without replacement inside a draw,
with dedicated seed `model_seed * 1000 + batch_size`. Draws may overlap one another.

Each model costs1full+107single-row+7partition+96sampled gradient calls=211;
sample visits are107+107+107+32×(4+16+64)=3009. Six models cost
**1266 gradient calls / 18054 sample visits / zero optimizer updates**.
Independent tests and the delivery suite are additional verification work.

## Original derivation and upstream study

For per-example gradient hᵢ at one unchanged parameter vector, the full gradient
is g=(1/N)Σhᵢ. A partition batch mean gⱼ must have weight nⱼ/N; equal weighting
of batch means changes the weight of examples when a tail batch is smaller.
This identity relies on the fixed parameters and sample-additive mean loss;
it does not combine gradients measured along a changing optimization trajectory.

For a uniformly chosen subset of b distinct examples, expanding the squared norm
of the mean using inclusion probabilities b/N and b(b−1)/(N(N−1)) gives
E||ĝ−g||²=(N−b)/(b(N−1)) × (1/N)Σ||hᵢ−g||². The formula includes the finite-population
correction, tends to zero at b=N, and is verified by independent exhaustive
enumeration on a small population. A finite Monte Carlo mean need not equal its
expectation. No empirical-ratio threshold is used to force a pass.

Official sources studied: [PyTorch2.14 autograd.grad](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html)
returns gradients rather than accumulating them in leaf `.grad`; [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
defines reduction behavior. This study uses neither ignored labels nor class
weights, which would change the effective denominator. PyTorch is maintained by
its team and contributors under its [BSD-style license](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE).
Implementation and derivation here are original, not copied from upstream code.
UCI Wine's bundled CC BY4.0 attribution remains intact.

## Run

Install using root README, then:

```bash
python -m pytorch_lab.wine_gradient_sampling_cli --output results/my-gradient-sampling.json
```

Existing targets, directories and symlinks are refused. Scientific failures retain
structured evidence and exit nonzero. Serialization rejects NaN/Infinity.
The twentieth CLI is included in source CI and installed-package checks.

## New actual observations

The [raw Nov4 report](../results/2026-11-04-wine-gradient-sampling.json) contains all
indices, gradient hashes and squared errors, full reference gradients, and exact
configuration. Raw vectors are calculated in memory; hashes keep serialized
output bounded. Independent tests capture the real tensors and recompute metrics,
so hashes are not used as a substitute for numerical correctness checks.

All six configurations pass deterministic controls. Weighted partition maximum
absolute error ≤2.220446049250313e-16. Equal-weight partition gradient error L2 is
0.07795060920469293–0.09396175119916425. Across18model/size combinations, empirical
mean squared error divided by its analytic expectation ranges0.762392084431612–
1.2375402944172407. These finite32draw estimates are descriptive, not confidence
intervals, significance tests, guaranteed monotonic observations or evidence of bias.

Independent checks include analytic linear-model CE gradients, exhaustive subset
variance, true per-example/subset tensor comparisons, poisoned-heldout/forward
isolation, no optimizer calls, and unchanged parameters, existing gradient buffers,
CPU RNG and caller thread count.

## Limits and next step

This studies initial fixed parameters, not trained representations, SGD convergence,
wall-clock efficiency or generalization. No validation/test scoring happens here;
the existing delivery regression separately reruns the historical Wine benchmark.
CPU float64 evidence does not cover GPU/AMP/DDP. Next: real batch-size training
comparisons with equal sample-visits and equal optimizer updates reported as
separate protocols, not conflated into a single fairness claim.

See [local validation](../results/2026-11-04-gradient-sampling-verification.md) and
[independent review](../results/2026-11-04-gradient-sampling-independent-review.md).
