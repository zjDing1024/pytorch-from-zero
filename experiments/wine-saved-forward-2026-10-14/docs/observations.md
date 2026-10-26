# Saved-forward observations

Date: 2026-10-14, Asia/Kuala_Lumpur. Primary capture began 2026-10-13T16:09:29.382136+00:00.

## Prospectively fixed controls

- Protocol SHA256: `2e18232b25b56ca76227604e2d00de99781048db755f3b60d8899c5b443ed27f`
- Fixture SHA256: `4db70a0c4f889e1a0adb5d3ed8c4c17e0eb36871b4c85e82885672e77fcfddf7`
- CPU: AMD EPYC 9V74 80-Core Processor
- Capability order: AVX512, DEFAULT, DEFAULT, AVX512
- All seven primary capture controls passed; all seven per-worker controls passed.
- All 19 original-step records match before/with/after hooks within every worker.
- All scientific records strictly repeat within A and within B.

## Saved values and matched-input experiment

Four CE saved slots were recorded at forward pack time, and unpacked in order [1, 2, 3, 0]. All were unchanged at unpack and exact across A/B.

| Slot | Shape | dtype | Values | Cross-condition |
|---|---|---|---:|---|
| 0 | [16, 3] | float64 | 48 | byte-identical |
| 1 | [16, 3] | float64 | 48 | byte-identical |
| 2 | [16] | int64 | 16 | byte-identical |
| 3 | [] | float64 | 1 | byte-identical |

Slots 0 and 1 both match explicit log-softmax values byte for byte. Their duplicate bytes do not establish distinct storage or uniquely name their consumers. Source rules support the qualified association with the CE log-softmax/NLL path. Slot 2 equals the targets; scalar slot 3 is 16.0.

Separate explicit log_softmax + nll_loss reproduces the original scalar CE and original dL/dlogits exactly within each worker. Its dL/dlog_probs equals the preregistered upstream (target = −0.0625, other = +0.0), including signed-zero bytes.

Another public log-softmax VJP uses identical logits, identical saved log-probs and identical upstream gradient across A/B, yet reproduces exactly the same five unequal gradient entries as original CE:

| Row | Class | A | B | Absolute difference |
|---:|---:|---|---|---|
| 2 | 0 | 0.020275821280621192 | 0.020275821280621196 | 3.469446951953614e-18 |
| 3 | 2 | 0.01913008659227004 | 0.019130086592270036 | 3.469446951953614e-18 |
| 6 | 0 | 0.02445097784878127 | 0.024450977848781266 | 3.469446951953614e-18 |
| 8 | 2 | 0.017291771656884038 | 0.01729177165688404 | 3.469446951953614e-18 |
| 12 | 0 | 0.021688540259568654 | 0.02168854025956865 | 3.469446951953614e-18 |

All five are 1 ULP apart; none exceeds 8 ULP. Row/class indices are zero-based. All 48 values and hex representations for the unequal entries are preserved in `results/findings.json` and full captures.

## Retained failures and null findings

- No mismatch was observed in any saved CE slot, visible forward boundary, explicit upstream or same-condition repeat.
- The original 19-stage comparison remains strictly unequal, with 263 changed stage entries and 22 beyond 8 ULP. Maximum 640 ULP occurs near zero; maximum absolute difference at that entry is 5.421010862427522e-19. No threshold changes were made.
- The three repeated baseline/observer records and downstream copies are repeated observations of related values, not independent statistical samples. Do not add their mismatch counts as independent evidence.
- Explicit decomposition is a separate public calculation; it is not an instruction-level trace of the original CE call.

## Interpretation

For this exact host/build/fixture, the newly observed saved-forward values rule out a difference at those CE saved slots. Equal explicit upstream and equal log-softmax output combined with the matched public VJP mismatch narrows the observed discrepancy to the log-softmax backward operation. Source reading explains what that operation consumes; the experiment does not trace individual native arithmetic steps.

No claim is made about a specific exp implementation, SIMD lane, reduction/FMA choice, microkernel bug, mathematical correctness ranking, historical CI, other hardware, full-training convergence, general error bounds or learner mastery. Public hooks bypass normal saved-version checks, so parity is an empirical fixture-specific control, not a general transparency guarantee.

## Final verification

Final code SHA256: `3362f69e70217fd2c48359dda01a8e1166d39b74201c75ad0ce6370f8e9f0426`. The final A-B-B-A verification passed every capture control; each of its four scientific hashes equals the corresponding primary-capture hash exactly. All 45 focused tests, Ruff check, Ruff format check, compileall and the existing environment pip check passed. Complete records are in `results/abba-reviewed.json` and `results/verification.json`. No new installation, package integration, full-project test run, container run or remote CI is claimed.

Independent review also passed 46 separately authored checks of raw evidence and failure/schema handling. Their scenarios overlap with the primary suite; counts are reported separately, not combined as unique evidence. No scientific or tested-harness blocker remained for this local package.
