# Fixed-batch training diagnostics

This original CPU float64 probe records actual loss, gradient presence and norm,
parameter values and measured update norm. It distinguishes symptoms without reading
an injected fault name. It is not a general root-cause classifier or convergence test.

## Frozen protocol

[Predeclared configuration](training-diagnostics-plan.json) was written before the
first run: five dyadic inputs, y=2x+0.5, zero initial weight/bias, SGD lr=0.1,
no momentum or weight decay, 12 steps. Compare the correct loop with omitted backward
and omitted optimizer.step, all with explicit zero_grad(set_to_none=True) each step.
Each report retains all before/after losses, parameters, gradients and deltas.

For this symmetric fixture, mean(x)=0 and mean(x²)=1/2, so
L=(w-2)²/2+(b-0.5)², dL/dw=w-2 and dL/db=2(b-0.5).
The test oracle independently uses these scalar equations and SGD updates at
atol=rtol=1e-12. This tolerance is not a general floating-point error bound.

## Run

```bash
python -m pytorch_lab.training_diagnostics_cli --output results/my-diagnostics.json
```

The output must not exist. Output JSON and stdout contain the same report. Invalid
arguments, existing files and dangling symlinks are refused. A failed write may leave
a partial file; this CLI does not promise transactional publication or durability.

## Observations and limits

- No gradients/no update: gradients were absent at measurement. Omitted backward is
  one explanation, but so are detached graphs or parameters not participating in loss.
- Nonzero gradients/no update: no parameter change was observed. It does not uniquely
  prove omitted step; zero learning rate, excluded optimizer parameters and rounding
  can also produce that symptom.
- Zero gradients are distinct from None. With momentum or weight decay, a parameter
  with a zero gradient can still change; a parameter with None is skipped by SGD.
- Partial gradient presence is reported rather than silently treating it as complete.
- Nonzero updates are observations, not proof of correct descent. A growing loss can
  coexist with an update. Exact zero is not used as a convergence threshold.

The controlled labels remain outside diagnose(). Experiment assertions know the
intervention for checking this fixture; the diagnostic function does not. No claimed
classifier accuracy, throughput improvement, arbitrary-model convergence, GPU or
multi-process monitoring is derived from these three controlled paths.

## Source study and attribution

PyTorch provides the tensor/autograd/SGD APIs. The following pinned official source
was studied; implementation and scalar oracle here are original, not copied:

- [Optimizer.zero_grad](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/optim/optimizer.py): None versus zero semantics.
- [SGD](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/optim/sgd.py): skip None; apply weight decay before momentum.
- [Autograd](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/autograd/__init__.py): backward accumulates leaf gradients.
- [BSD-style license](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE).

## Results

The [complete source report](../results/2026-11-02-training-diagnostics.json) records
36 measured steps. Correct training reduces loss from 2.25 to 0.16071347777446246;
both omitted-operation paths retain 2.25. These values describe this fixture only.
Software, installed-wheel and remote-CI verification are separately documented in
the verification record and learning report, not inferred from these loss values.
