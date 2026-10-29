# Joint shuffled training safe points

## Problem and deliberately small scope

The earlier index stream commits **issuance**, while the fixed-order trainer
commits completed microbatches. Saving them independently can pair a newer
cursor with older gradients/model parameters and silently skip work. This
increment introduces one owner and one jointly validated checkpoint. It does
not change those earlier APIs.

`JointConfig` retains the bounded affine fixture/training settings and adds a
separate `shuffle_seed`. Default: 23 samples, 3 input / 2 output dimensions,
microbatch size 5, accumulation 3, SGD learning rate 0.03, momentum 0.8 and
weight decay 0.01. The fixture seed and shuffle seed are separate local CPU
generators. Training is single-process CPU float64, without stochastic forward
operations. `drop_last=False` is fixed: each committed epoch visits every sample.

This is original engineering code, not a TorchData/Accelerate wrapper and not
a generic training framework. No workers, prefetch, random transforms, GPU,
AMP, scheduler, concurrent consumers, or DDP are implemented here.

## The operation and its commit point

`JointTrainer.step_microbatch()` performs these stages under exclusive ownership:

1. Check arguments and lower the safe-point guard **before** requesting indices.
2. Fetch a batch from the owned stream. Its issuance cursor has now advanced;
   the trainer's committed counters have not.
3. Compute squared-error **SUM** and backward, accumulating unnormalized gradients.
4. At an accumulation boundary or epoch tail, divide by the actual accumulated
   target-element count, run SGD, and clear pending gradients. For a 5+3-sample
   tail window with two outputs, the denominator is 16, not 20 or a fixed number
   of microbatches. Momentum and weight decay are applied once per SGD update.
5. If the epoch is exhausted, flush its tail and explicitly prepare the next
   stream permutation. This canonicalizes every completed epoch to next epoch,
   cursor zero, with no pending gradients.
6. Commit the training counters and reopen the guard. Return immutable committed
   indices for a caller's observation; this return is not a durable acknowledgement.

The failpoints `after_fetch`, `after_backward`, `after_update` and `before_commit`
exercise different partial mutations. `after_update` is accepted only on a batch
that will update; bad failpoint arguments leave the safe object unchanged.

Any tested pre-commit failure leaves the owner **tainted**, not rolled back.
Snapshot, save, step and train then refuse. This includes an optimizer or epoch
advance which mutates and subsequently raises. Recovery constructs a new owner
from the last whole checkpoint and replays work after that saved point. Discard
the failed owner. A checkpoint can be older than the last successful operation,
so all subsequent work, including successful but unsaved work, may be recomputed.

No claim is made for arbitrary asynchronous interruption at the commit/return
instruction boundary, process-kill handling, or exactly-once logging, external
effects or sample execution. The committed *recovered trajectory* can match the
uninterrupted trajectory even though discarded computations ran more than once.

## One snapshot, not two independently selected files

The joint snapshot binds:

- Complete training configuration, exact PyTorch version, regenerated fixture
  fingerprint, format and schema version.
- Stream configuration/identity, current permutation, epoch, issuance cursor,
  and post-permutation generator state.
- Committed epoch/cursor/microbatch/update counts.
- Model parameters, SGD parameter groups and momentum buffers.
- Pending loss-SUM gradients, microbatch count, sample count and target-element
  denominator.
- A typed content digest over the complete joint state, including the stream.

Restore validates the actual fixture, derives the expected stream config from it,
and checks cursor/counter/window alignment before returning a new owner. A stream
from another seed or another progress point is rejected, even if its own nested
digest is recomputed. There is no API accepting an independently chosen stream
plus independently chosen trainer.

These checks detect accidental damage or incompatible components, not provenance.
An attacker who replaces numerically plausible gradients/parameters or creates a
consistent forged stream and recomputes all hashes can defeat history checks.
The loader does not replay the entire training history to authenticate a snapshot.
Use `expected_config` from independently known settings, and only load trusted
own files. File bounds and `weights_only=True` are not a hostile-file sandbox.

Snapshots are deep detached copies. Disk publication uses the existing same-directory
temporary-file, fsync and no-clobber hard-link helper. An existing checkpoint name
is not replaced. This requires a compatible local filesystem and does not promise
directory persistence after power loss. Save I/O failures leave the training owner
at its existing safe point; they do not imply a checkpoint was published.

## Reproduce and interpret the experiment

Install the project using the hash-locked CPU environment described in
[runtime reproducibility](runtime-reproducibility.md), then use a new output name:

```bash
python -I -m pytorch_lab.joint_recovery_cli --output results/local-joint-recovery.json
python -m pytest tests/test_joint_recovery.py tests/test_joint_recovery_cli.py tests/test_joint_recovery_independent.py -q
```

The predeclared CLI uses fixture seeds 42/7/123, shuffle seed 19, three epochs,
and failure cuts 1/3/4/7 for fetch/backward/update/precommit respectively. Each seed
uses separate baseline, fault and recovery worker processes: **12 checkpoint
recovery cases, 3 fresh recovery processes, 9 total workers**, not 12 processes.
Fault workers observe the injection and refused unsafe operations before returning
an expected nonzero status. This is controlled Python exception injection followed
by process exit, not SIGKILL or filesystem fault simulation.

Success requires exact full typed-state digests at every remaining safe point,
identical committed index suffixes, replay of the attempted batch, unchanged prior
checkpoint bytes, and refusal to save/continue a tainted owner. Baseline coverage
also checks 15 committed microbatches and 69 sample visits with all 23 indices once
per epoch. Numerical state equality is within the same fixed runtime; no
cross-runtime or serialized archive-byte equality is inferred.

The independent test separately calculates affine derivatives and momentum/decay
updates rather than using this trainer's backward/SGD implementation as its oracle.
Its floating-point comparison is separate from exact resumed-state comparison;
neither changes earlier Wine strict-hash or 8-ULP gates.

See [compact experiment](../results/2026-10-30-joint-recovery.json) and
[actual verification record](../results/2026-10-30-joint-recovery-verification.md).
Test counts, worker cases, sample visits and source/wheel execution are different
units and are never added together. Learner independent mastery remains unassessed.

## Official source study and design choice

**TorchData**, by Meta/PyTorch contributors, addresses resumable data iteration.
Its [0.11 tutorial](https://meta-pytorch.org/data/0.11/stateful_dataloader_tutorial.html)
describes saving loader state and custom sampler/dataset state. The
[fixed StatefulDataLoader source](https://github.com/meta-pytorch/data/blob/377e64c1be69a9be6649d14c9e3664070323e464/torchdata/stateful_dataloader/stateful_dataloader.py)
organizes iterator state around data retrieval. This makes explicit data progress
worth studying, but a loader's progress alone cannot assert that an optimizer
update completed. The current project deliberately excludes its worker machinery.
Its [license is BSD-3-Clause](https://github.com/meta-pytorch/data/blob/377e64c1be69a9be6649d14c9e3664070323e464/LICENSE).

**Hugging Face Accelerate** organizes broader training-state persistence.
The [official guide](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)
identifies model, optimizer and random-state coverage and supports custom state.
In [checkpointing.py at 90b233c](https://github.com/huggingface/accelerate/blob/90b233c86202a5ded4bf66407e75222c5fb007fc/src/accelerate/checkpointing.py),
`save_accelerator_state` handles component files, including a conditional stateful
loader branch. This inspires a state-inventory review, not a claim that those
separate writes supply our transaction protocol. Its
[license is Apache-2.0](https://github.com/huggingface/accelerate/blob/90b233c86202a5ded4bf66407e75222c5fb007fc/LICENSE).

Our inference from these responsibilities is to couple the small owned stream
and trainer at a single safe point, using fail-closed recovery rather than a new
pending-token acknowledgement system. This smaller design makes mutation/commit
boundaries testable without introducing distributed transactions. Source study
used official references; no upstream implementation, comments or excerpts were
copied, and neither project is added as a runtime dependency.
