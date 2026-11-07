# Wine inference artifact: a bounded JSON contract

Predeclared plan: [wine-model-bundle-plan.json](wine-model-bundle-plan.json).
The previous validation experiment selected mlp16/constant/epoch20 across seeds.
This increment fixes seed42 because it is the first existing declared seed, **not**
because of its validation score. It retrains only the original 107 training rows,
with train-fitted population standardization, 140 updates / 2140 sample visits.
No validation or test score is calculated here. Loading/validating the complete
vendored CSV is not equivalent to never reading test bytes.

## Contract and implementation

Original `wine_model_bundle.py` serializes float64 tensors as bounded JSON arrays,
not executable pickle. The payload contains exact ordered feature names, class
index names, mean/scale, architecture/seed/epoch, training row IDs and source digest.
The schema fixes this small model, rather than pretending to be a general model registry.
`state_dict` is converted into independent Python lists before serialization.
Inference uses eval + no_grad, raw measurements in the vendored UCI Wine units,
and training-only preprocessing. It returns logits, probabilities and class labels.

Limits: at most 1,000,000 JSON bytes, 1–1024 rows, exactly 13 declared ordered
columns, finite numeric values with absolute magnitude <=1,000,000, scale >=1e-12.
These bounds prevent simple numerical/resource errors; they are **not** scientific
plausibility or out-of-distribution detection. A sender can lie about column labels.
The parser rejects duplicate keys, NaN constants, excessive nesting and malformed
JSON. Schema/state shape mismatch and modified payload digests fail closed.
SHA256 detects accidental changes, **not authenticity**: anyone changing the payload
can recompute it. Consume only artifacts with independently trusted provenance.
JSON avoids pickle execution, but this is not a hostile-file security sandbox.

## Reproduce and use

```bash
python -m pytorch_lab.wine_model_bundle_cli --output results/local-bundle-check.json
python -m pytorch_lab.wine_model_bundle_cli \
  --bundle results/2026-11-08-wine-model.json \
  --request results/2026-11-08-wine-request.json \
  --output results/local-predictions.json
```

The first command trains the frozen configuration, writes an intermediate bundle,
invokes a separate Python interpreter to reload and predict, then compares original
logits and all reload predictions. Its persistent report contains the bundle and
prediction evidence. The checked-in standalone model and request files are extracted
from that same run. The request is the 107 public training-feature rows, not private
user input. No output is overwritten. Input artifacts and requests are strictly read-only.
No server, external upload, dependency on credentials, or paid service is needed.

## Evidence and interpretation

[Raw report](../results/2026-11-08-model-bundle.json) and
[verification](../results/2026-11-08-model-bundle-verification.md).
Predictions before/after reload match within the observed same-runtime experiment;
this does not promise bitwise agreement across hardware/PyTorch versions. There is
no new accuracy, throughput, calibration, robustness or generalization claim.
Inference artifacts omit optimizer state and are **not training-resume checkpoints**.

Next useful work is a small batch-inference interface with explicit per-request
errors and measured end-to-end costs, keeping preprocessing and validation separate
from model-only timings. Do not optimize based on already reused validation scores.

## Source study and licenses

- [PyTorch Module.state_dict](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/nn/modules/module.py): shallow state references explain why independent lists are materialized.
- [Official save/load tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html): architecture plus state and eval mode, not a serialized live trainer.
- [PyTorch serialization notes](https://docs.pytorch.org/docs/2.14/notes/serialization.html): trusted provenance and loading boundaries; this implementation uses JSON instead of pickle.
- [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/v2.14.1/LICENSE): BSD-style and component notices. Studied interfaces and implemented the artifact contract originally; no upstream code copied.
- Existing UCI Wine CC BY 4.0 / scikit-learn BSD-3-Clause attribution remains in package data. This repository has not selected its own open-source license.
