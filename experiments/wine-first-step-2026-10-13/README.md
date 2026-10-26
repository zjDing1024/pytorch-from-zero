# Frozen Wine first-step probe

A standalone, original CPU float64 investigation prepared for `pytorch-from-zero` on **2026-10-13, Asia/Kuala_Lumpur**. It freezes a real TRAIN minibatch and initial MLP16 weights, then records exactly one forward/loss/backward/SGD update per worker. It is not a training benchmark or a model-quality experiment.

**Current state: locally prepared and tested, not integrated or published.** The earlier fixed-logits increment remains separately frozen pending publication approval. No prior files, manifests, thresholds or remote repositories were changed by this preparation.

## Main observation

On AMD EPYC 9V74, A-B-B-A changed the observed ATen capability AVX512 → DEFAULT → DEFAULT → AVX512. The four recorded forward boundaries were byte-identical. The first recorded mismatch was `d(loss)/d(logits)`: 5/48 entries differed by 1 ULP. Across 19 stages / 1,946 values, 263 entries differed and 22 exceeded the preregistered 8-ULP budget. Both conditions repeated exactly.

The maximum was 640 ULP in a momentum-buffer entry (absolute difference **5.421010862427522e-19**, values about 4.37934e-6). Across every stage the maximum absolute difference was **2.7755575615628914e-17**. All updated parameters were within 8 ULP. Strict and numerical failures remain failures; small absolute differences do not change the chosen contract.

**This does not locate a backward kernel defect.** Identical logits and scalar loss do not establish identical unrecorded saved forward intermediates, such as log-softmax output. The first recorded mismatch is an observation boundary, not necessarily the origin. It does not explain the earlier Intel-host full-Wine result or historical CI.

## Files

- `docs/plan.json`: immutable prospective local protocol, before fixture construction or model evaluation
- `data/frozen_fixture.json`, `docs/freeze.json`: all input/weight values and byte hashes, freeze-time controls
- `probe.py`: standalone original fixture/capture/validation/comparison CLI
- `tests/test_probe.py`: mathematical sanity checks, integrity and failure-path tests
- `results/abba-initial.json`, `results/abba-reviewed.json`: complete four-worker captures before/after validation hardening
- `context/probe-initial.py`: exact initial source retained for provenance
- `docs/observations.md`: detailed outcomes and interpretation
- `docs/source-study.md`: official PyTorch sources and license boundaries
- `docs/post-observation-changes.md`: disclosed hardening, no scientific changes
- `docs/integration-plan.md`: bounded future integration and blockers
- `daily-learning/2026-10-13/README.md`: partial daily report
- `VALIDATION.md`: executed checks and remaining verification limits

## Run

Existing tested environment: Python 3.12.14, PyTorch 2.14.1+cpu, pytest 9.1.1, Ruff 0.16.10, Linux x86_64. The `requirements/` directory preserves the existing target-platform hash locks; no dependencies were installed or upgraded for this task. They are not a portability guarantee or evidence of a new clean install.

From this directory with the existing CPU environment active:

```bash
python -I probe.py --output results/my-abba.json
python -m pytest -q
ruff check probe.py tests
ruff format --check probe.py tests
python -m compileall -q probe.py tests
```

Use a new output path. Capture is fixed A-B-B-A with one step per worker, one intra/inter-op thread, three thread environment variables set before import, deterministic algorithms enabled, and 60 seconds maximum per worker. `--timeout` can lower this bound. Exit 0 means valid capture controls, not strict or 8-ULP equality. Numerical failures are retained under `comparisons_to_A1`.

The included frozen fixture is authoritative. `--freeze` is the original provenance recipe, refuses to overwrite existing fixture/manifest files, and does no model forward or update. Do not regenerate or replace inputs to search for a preferred outcome. The source pins the approved frozen fixture hash; a different environment's refreeze may be byte-different and requires a new protocol rather than changing this one.

The 2 MiB JSON limit is checked after worker output collection and is not a hard memory sandbox. No GPU, paid compute, downloads, fit loop, validation evaluation, test evaluation or test-set tuning is performed by the probe.

## Data and authorship

Wine: Aeberhard, S. & Forina, M. (1992), [UCI Wine](https://doi.org/10.24432/C5PC7J), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original pinned scikit-learn CSV and BSD notice are retained in `data/`; full provenance describes its header/label-format changes. This experiment selects TRAIN rows, standardizes from TRAIN statistics once, and creates a frozen derivative fixture. It does not claim ownership of the data or endorsement by its authors.

Implementation/tests are original. PyTorch source was read for understanding, not copied. No new license is selected for this standalone preparation. Assistant execution provides inspectable engineering evidence; the learner's independent mastery remains unverified.
