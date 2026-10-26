# Wine CE saved-forward and matched-backward probe

Original, standalone CPU float64 investigation prepared for **2026-10-14, Asia/Kuala_Lumpur**. It extends the previous frozen Wine first-step experiment using its exact fixture and original step function. **Prepared locally; not integrated or published.**

## Finding

On AMD EPYC 9V74, the AVX512 / DEFAULT / DEFAULT / AVX512 capture found:

- Every CE saved slot was byte-identical across conditions: two 16×3 float64 tensors, 16 integer targets and one scalar total weight. Both 16×3 records exactly matched the explicit log-softmax output.
- Original logits, scalar loss, explicit log-softmax output, and the upstream gradient were byte-identical.
- Public log-softmax backward, with matched input and upstream gradient, reproduced the same **5 of 48 gradient differences, each 1 ULP**, as original CE. Maximum absolute difference: 3.469446951953614e-18.
- Before / with / after the observer, all 19 original-step boundaries matched exactly within each worker. A repeated exactly, and B repeated exactly.

This narrows this fixture's discrepancy to the **public log-softmax backward operation with matched recorded inputs**. It does not identify a native microkernel defect, prove either answer incorrect, or explain historical cross-host CI. The saved-forward mismatch hypothesis was not observed at any captured CE saved slot. Unrecorded transient arithmetic remains outside the trace.

The original whole-step 8-ULP diagnostic still fails: 22 stage entries exceed 8, with maximum 640 ULP near zero. The newly isolated VJP difference is within 8 ULP. Neither result changes the preregistered budget; software checks passing does not turn the scientific comparison into strict equality.

## What runs

The exact Oct 13 fixture is pinned at SHA256 `4db70a0c4f889e1a0adb5d3ed8c4c17e0eb36871b4c85e82885672e77fcfddf7`. It contains the original 16-row TRAIN minibatch, TRAIN-derived normalization, seed-42 MLP16 initialization and target values. It is loaded unchanged; no preprocessing, random initialization, fit loop, validation or test evaluation runs.

Each fresh worker performs:

1. One original first step without hooks.
2. The same first-step sequence, observing only the original F.cross_entropy forward with public saved_tensors_hooks.
3. One original first step without hooks again.
4. A separate explicit log_softmax → nll_loss calculation on the original frozen logits.
5. A separate public autograd.grad log-softmax vector-Jacobian product with upstream −1/16 at target classes and +0 elsewhere.

The three steps start from identical initial parameters; they are not three sequential training updates. Four primary workers therefore perform 12 independent first-step updates. Verification reruns and unit-test work are reported separately in the change log. This is a diagnostic experiment, not a training-performance benchmark.

The hook records a detached clone's bytes, returns a detached original-value payload, and returns that payload unchanged at unpack. No tensor is mutated between pack/unpack. Public hooks alter autograd machinery and skip usual version-counter checks; observed parity supports only this fixture. Saved slots are indexed observations, not uniquely identified consumer operations. Original stride/layout metadata is not recorded.

## Run

Tested existing environment: Python 3.12.14, PyTorch 2.14.1+cpu, pytest 9.1.1 and Ruff 0.16.10. No dependencies were installed or upgraded. Existing target-platform hash locks are retained, but no new clean installation or portability claim is made.

From this directory with that environment active:

```bash
python -I saved_probe.py --output results/my-capture.json
python -m pytest -q
ruff check saved_probe.py tests
ruff format --check saved_probe.py tests
python -m compileall -q saved_probe.py tests
```

Output paths must be new. Each worker has a maximum 60-second timeout. Exit 0 means capture controls passed, not strict or 8-ULP equality. JSON size limits are checked after subprocess collection; they are not hard memory sandboxes. No private backward APIs are invoked. The code reads no user credentials and performs no network calls.

## Evidence and structure

- `docs/plan.json`: unchanged local prospective protocol, declared before new numerical work
- `reference/oct13/`: exact original implementation, plan, freeze manifest, fixture and data attribution
- `saved_probe.py`, `tests/`: original observer, matched-gradient experiment and failure-boundary tests
- `results/abba-initial.json`: full primary four-worker values, hashes, controls and comparisons
- `docs/source-before-first-capture.py`: initial source retained for primary-capture attribution
- `docs/observations.md`: exact values and interpretation
- `docs/source-study.md`: pinned official source/license reading
- `docs/post-observation-changes.md`: hardening and verification work after observing results
- `daily-learning/2026-10-14/README.md`: daily report

The reference module retains its historical fixture-construction function for provenance; this package uses only its loading, first-step and comparison support. Do not invoke refreeze or regenerate the fixture. Its historic fixture-generation data source and recipe are documented in the included provenance.

## Data, authorship and limits

Wine: Aeberhard, S. & Forina, M. (1992), [UCI Wine](https://doi.org/10.24432/C5PC7J), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The unchanged fixture is the previous TRAIN-standardized derivative; no new dataset transformations were introduced. Pinned scikit-learn source provenance and its BSD notice are retained. No author endorsement is implied.

Implementation and tests are original. PyTorch source was read and cited, not copied; PyTorch and dependencies retain their own licenses. No new project license is selected here. This package supplies inspectable assistant-prepared engineering evidence; the learner's independent mastery is unverified. Package integration, full-project regression, wheel/container execution and exact-commit remote CI have not been performed for this increment.

## Verification

45 focused tests passed, as did Ruff, formatting, compileall and the existing environment pip check. The hardened A-B-B-A rerun retained all controls and reproduced every primary scientific hash exactly. See `results/verification.json`.

Independent review also passed 46 separately authored checks of raw evidence and failure/schema handling. Their scenarios overlap with the primary suite; counts are reported separately, not combined as unique evidence. No scientific or tested-harness blocker remained for this local package.
