# CPU input-Dropout joint training recovery

This original implementation closes the bounded stochastic-training integration
specified in the learning roadmap on 2026-11-01. It is not the earlier fixed-weight
Dropout replay: parameters now receive real SGD updates, and pending accumulated
gradients survive a completed-microbatch checkpoint.

## Fixed model and state ownership

`StochasticJointTrainer` uses CPU float64 input Dropout (non-inplace), followed by
the existing zero-initialized 3-input, 2-output affine model. Default configuration:
23 samples, microbatch 5, accumulation 3, p=0.35, learning rate 0.03, momentum 0.8,
weight decay 0.01. The short final microbatch flushes the pending window. Raw SUM-MSE
gradients accumulate; divide once by the actual number of target elements before
SGD. Zero the gradients only after an update, never at each microbatch entrance.

The shuffle stream owns its generator; the trainer separately owns default-CPU RNG
bytes for model Dropout. The affine parameters start at zero, so model construction
does not consume random initialization draws. `fork_rng(devices=[])` restores the
caller's CPU state on both success and failure. This temporarily borrows a global
resource and is only supported in a single-threaded exclusive-owner setting.

The new format is `pytorch-lab-stochastic-joint-training`, schema version 1. It
binds the architecture, training mode, p/configuration, model RNG, dataset fingerprint,
shuffle state, weights, SGD momentum, pending gradients, actual denominator and
committed progress. Existing affine-joint and fixed-weight replay formats are not
silently reinterpreted. Same-version, same-environment trusted own files only.

## Failure and persistence contract

Invalid preconditions leave a safe owner unchanged. Once an operation begins, it
marks the owner unsafe before fetching indices. Exposed failpoints are after_fetch,
after_forward, after_backward, after_update, before_commit. The after_update point
is allowed only on an update-triggering batch. Any exception after beginning leaves
the old owner unusable for continue, snapshot or save; reload the whole last safe
checkpoint. Caller RNG restoration does not undo model, optimizer, gradient or
shuffle mutation. No internal rollback or external-effects exactly-once claim.

Only after forward, backward, optional SGD update and next-epoch permutation
preparation succeed are private model RNG and progress committed. `last_trace` is a
copied diagnostic of the most recent successful step, not persisted training state;
it is None on construction/restoration until another step succeeds.

Save retains the bounded `weights_only` loading and no-clobber publication protocol.
A safe point is an in-memory boundary, not a durable acknowledgement. No guarantee
covers arbitrary asynchronous interruption between final assignments and return,
process kill, power loss, hostile file decoding, concurrency, worker prefetch,
random augmentation, GPU, AMP, DDP or cross-runtime snapshot exchange. Checksums
detect accidental corruption, not authenticated provenance or fabricated history.

## Reproduction and evidence

```sh
python -m pytorch_lab.stochastic_joint_cli --output results/my-stochastic-joint.json
python -m pytest -q tests/test_stochastic_joint.py tests/test_stochastic_joint_math.py
python -m pytest -q tests/test_stochastic_joint_cli.py
```

Use an absent output path. The CLI launches new processes and compares every resumed
safe-point digest and complete step trace, not just a final loss. Seed 7, 42, 123;
normal cuts 1, 3, 5; three epochs. It separately exercises five failure boundaries.
The full saved baseline traces include outputs, losses and raw accumulated gradients.
Exact restoration is checked inside each execution environment, not between hosts.

Independent affine-gradient and momentum equations use actual dropped inputs with
atol=rtol=1e-12. This is independent math, not an independent RNG algorithm. Exact
resume checks use no numerical tolerance. p=0 and p=1 are separate boundary tests;
p=1 can still update bias and decay/momentum, and need not advance RNG. A concatenated
large batch can consume different Dropout masks than microbatches, so it is not an
unconditional stochastic accumulation oracle.

Actual execution counts and limitations are recorded in
[local verification](../results/2026-11-02-stochastic-joint-verification.md) and
[raw subprocess evidence](../results/2026-11-02-stochastic-joint.json).
Remote commit and exact-SHA CI are recorded only after publication in the learning
report. Local checks do not establish remote Docker execution.

## Official sources and original implementation

- [PyTorch random.py](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/random.py): caller state restoration in finally.
- [PyTorch Dropout](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/dropout.py): p, training and inplace determine behavior beyond parameter state.
- [PyTorch SGD](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/optim/sgd.py): decay, momentum and update order. [BSD-style license](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE).
- [Accelerate checkpointing](https://github.com/huggingface/accelerate/blob/90b233c86202a5ded4bf66407e75222c5fb007fc/src/accelerate/checkpointing.py): restore object state before RNG; framework RNG-error logging is not adopted as silent success in this strict contract. [Apache-2.0 license](https://github.com/huggingface/accelerate/blob/90b233c86202a5ded4bf66407e75222c5fb007fc/LICENSE).

No upstream code was copied. This extends this repository's original affine-joint
implementation, retaining the old implementation and tests as compatibility checks.
The artifact demonstrates tested behavior, not the learner's personal mastery.

## Stop condition and next direction

Close this integration with full regression and exact-commit CI. Then move to a
predeclared training-diagnostics/optimization question rather than another isolated
checkpoint demo. Learner self-check: derive the actual-denominator update, explain
why clearing pending gradients breaks recovery, and reconstruct a seed-only failure.
