# Independent batch-budget review

New execution on 2026-11-04; these checks do not reuse prior experiment results.

## Actually executed

`PYTHONPATH=src python -m pytest -q tests/test_wine_batch_budget_independent.py`

7 tests passed (2.23 seconds), with the existing optional NumPy-unavailable Torch warning. Ruff formatting and lint passed. The six parametrized tests cover both architectures and all three seeds, each auditing all three arms. A seventh test audits the complete experiment's isolation and actual costs.

## Independent checks

- Capture actual model forward rows and actual SGD post-update tensors, rather than accepting reported budget booleans.
- Independently slice each of four explicit permutations, retain every 11-row epoch tail, and check actual sequence and reported row IDs/batch sizes. Equal-visits arms consume the same 214 row positions; the 14-update batch32 arm extends this prefix to 385 visits.
- Compute reference gradients with cross entropy `sum / actual_batch_count`, then apply a handwritten coupled-weight-decay and momentum recurrence. All 18 trajectories match actual optimizer states at every update under `atol=rtol=1e-14`. This detects a fixed full-batch denominator applied to the tail as well as incorrect updates.
- Independently verify recorded loss, actual-update norm, post-update tensor hash, final parameters, and final full-training CE against those actual/reference tensors.
- Replace heldout features with NaN and labels with -100; forbid new splits and evaluation through the legacy metrics helper; guard every forward to accept only standardized training rows.
- Count actual operations across the complete experiment: 216 optimizer updates, 4,878 training sample visits, and 24 full-training evaluation forwards. Evaluation visits are separately 2,568. The shared batch16 baseline is not rerun for each comparison.
- Caller global CPU RNG and thread count are unchanged after completion.

## Interpretation and limits

Equal sample visits do not give equal updates, and equal updates do not give equal sample visits. Momentum and weight decay act per update. No test requires either batch size to win; these controls establish implementation and budget integrity, not universal optimizer quality, convergence, generalization, or matched wall-clock cost. The independent handwritten reference itself performs additional training beyond the main report's stated budget.

Only this independent test file and review document were authored by the reviewer. A finite initial/endpoint-loss guard was recommended and added by the implementation owner. Ruff initially identified unbound loop-closure references in the test wrappers; explicit default bindings resolved them before the final passing run. No experiment configuration or tolerance was adjusted.
