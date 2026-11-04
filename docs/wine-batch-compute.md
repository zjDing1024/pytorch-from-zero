# Wine batch budgets: bounded CPU update-loop measurement

This original measurement harness answers a narrower question than end-to-end
training speed: what wall and process CPU time do the existing three prepared
batch update loops consume on this invocation? It does not modify the scientific
batch-size protocol or tune from timing results.

## Official research and design

We studied [PyTorch 2.14 benchmark utilities](https://docs.pytorch.org/docs/2.14/benchmark_utils.html)
for explicit thread settings, warmup and repeated observations. PyTorch is
maintained by its team/community under its [BSD-style license](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE).
Our code is original and does not use/copy Timer or its adaptive repetition logic.
[Python clocks](https://docs.python.org/3.12/library/time.html) distinguish elapsed
`perf_counter_ns` from process CPU `process_time_ns`. Nanosecond units do not imply
nanosecond accuracy. Clock implementation/resolution are retained in the report.

The new lean loop pre-indexes tensor batches and initializes each model/optimizer
before timing. Timing includes Python iteration, zero_grad, forward, CE, backward
and SGD.step. It excludes imports, data preparation, initialization, indexing,
evaluation, tensor hashing and reference checks. The CPU-clock reads are nested
inside wall-clock reads, so the scopes differ by small clock-read overhead.
There is no claim of pure kernel time or total application cost.

## Frozen protocol

See [plan](wine-batch-compute-plan.json), written before results. For each of two
architectures and three seeds, preserve the three arms from the prior batch
budget experiment. Run the historical instrumented implementation once per arm
outside the timers, then one warmup and three measured repetitions with rotating
arm order. Every invocation starts from a newly initialized model/optimizer.
These are within-process repeats, not independent machine/process replicates.

- Reference: 18 trajectories, 216 updates, 4,878 training visits.
- Warmup: 18 trajectories, 216 updates, 4,878 training visits.
- Measured: 54 trajectories, 648 updates, 14,634 training visits.
- Total: 90 trajectories, 1,080 updates, 24,390 training visits; 90 endpoint
  full-training evaluations add 9,630 evaluation visits outside timers.

Actual final parameters and training CE must match the established reference
exactly, and sample order/budget must match. There is no performance ranking or
latency threshold in acceptance. Min/median/max wall time and median process CPU
time are descriptive; keep all raw observations including outliers.

## Run and interpretation

```bash
python -m pytorch_lab.wine_batch_compute_cli --output results/new-compute.json
```

The CLI refuses existing outputs and serializes finite JSON before writing.
[Raw local evidence](../results/2026-11-05-wine-batch-compute.json) contains six
configurations and all 72 timed loop observations, including warmups. Reference
states and all budgets passed. For example, linear/seed42 wall medians were
2.333 ms (batch16/14 steps), 1.544 ms (batch32/8 steps), and 2.530 ms
(batch32/14 steps); these are this host/run's narrow update-loop costs, not a
portable speedup. No statistical significance is asserted from three repeats.

The timing loop does not alter final learning outcomes; it separates audit
measurement overhead from the computation being measured. Equal sample visits
and equal updates remain different scientific questions. Shared-host load,
frequency, caches and clocks remain uncontrolled. No GPU/AMP/distributed or
heldout generalization claim is made. Historic full Wine delivery checks may
score heldout data separately, but never feed this experiment's configuration.

Next return to a predeclared learning-rate schedule on an optimizer-update clock,
without another timing variant or searching for a nicer timing ratio.
