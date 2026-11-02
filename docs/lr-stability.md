# Learning-rate stability with an analytic reference

This original CPU float64 experiment extends training observations into optimization
stability. It does not alter the prior diagnostic learning-rate bound or checkpoints.
The fixed five-row fixture is x=(-1,-.5,0,.5,1), y=2x+.5, with zero weight/bias,
mean squared error and plain SGD (no momentum/weight decay).

## Derivation and frozen contract

Let ew=w-2 and eb=b-.5. Since mean(x)=0 and mean(x²)=.5,
L=.5 ew²+eb² and Hessian=diag(1,2). Each update gives ew'=(1-lr)ew,
eb'=(1-2lr)eb. For k updates, ew=-2(1-lr)^k and eb=-.5(1-2lr)^k.
The worst-case contraction condition is 0<lr<1. At lr=1, this initialization
leaves a bias two-cycle; lr>1 excites a growing bias error. Spectral instability
need not be observed if an initialization has zero error along the unstable axis.

[The plan](lr-stability-plan.json) was written before results: rates .1,.75,1,1.25,
12 updates each, fixed atol=rtol=1e-12 analytic comparison. These choices distinguish
monotone component error, sign-alternating contraction, a critical two-cycle and
finite growing error. lr=1 loss drops from 2.25 to .25 on the first update, then stays
.25; it is not constant from initialization. lr=1.25 initially lowers total loss
before its unstable direction dominates, so one declining step cannot certify stability.

## Usage

```bash
python -m pytorch_lab.lr_stability_cli --output results/my-lr-stability.json
```

The output must not already exist. Report retains every before/after parameter,
loss, gradient, actual update, parameter error and analytic comparison. The pure
`spectral_regime` function describes this known Hessian, not an inferred universal
root cause. `run_rate` bounds lr to [0,1.5] and steps to 1..32 and rejects nonfinite
measurements. Those execution bounds do not guarantee every analytic comparison
passes in floating point. `run_experiment` applies the narrower frozen four-case gate.

## Evidence and numerical limits

[Main raw report](../results/2026-11-02-lr-stability.json) contains 48 measured updates.
[Additional stress observation](../results/2026-11-02-lr-stability-stress.json) retains
lr=1.5 at 32 steps: finite computation but analytic weight comparison fails at some
late steps under the unchanged 1e-12 gate. It was found by an extra broad-bound test,
not selected as a new main acceptance case. Exact cause has not been isolated;
large bias versus small residual weight error is consistent with cancellation but
is not proven causal. Do not assert this same failure pattern on every runtime.
No tolerance was widened and no unsuccessful raw comparison was changed to true.

Correct learning can oscillate in parameters while loss contracts. A nonzero
update does not guarantee convergence; decreasing loss for a finite prefix does
not establish asymptotic convergence. These results do not recommend a learning
rate for arbitrary neural networks, GPUs, data order, momentum or stochastic losses.

## Source study and originality

PyTorch SGD source explains the plain update used here:
[sgd.py](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/optim/sgd.py).
[autograd.backward](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/autograd/__init__.py)
accumulates leaf gradients, so this experiment clears them each step.
[BSD-style three-clause license](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE).
The fixture, derivation, instrumentation and tests are original, with no copied code.
