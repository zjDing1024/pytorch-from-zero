# Wine batch inference: partial errors and measured latency

[Plan fixed before execution](wine-batch-inference-plan.json). This increment uses the
existing mlp16/seed42/constant/epoch20 artifact contract. It changes delivery, not model
selection: there is no new validation/test scoring, hyperparameter search, or accuracy claim.
The implementation and tests are original, not copied from a serving framework.

## Input and failure contract

`WineBatchPredictor(bundle)` validates and loads the bounded JSON artifact once.
`predict({"features": [...], "rows": [...]})` then validates the exact ordered feature
names and a list of 1–1024 rows. Each row must contain 13 finite int/float values
with magnitude <=1,000,000. Booleans, numeric strings, nonfinite values and enormous
integers are rejected. The existing raw units and train-fitted preprocessing apply.

Each response item preserves its zero-based input index. Valid rows have `ok: true`
and logits, probabilities, class index/name. Invalid rows have `ok: false` and a stable
`invalid_shape` or `invalid_number` code, without echoing the raw input. All valid
rows are gathered into **one model forward call**, then scattered to original order.
If all rows are invalid there is no tensor/model work. Summary counts reconcile with
all requested rows. No input or bundle mutation is intended or required.

Global faults (wrong features/fields, empty/oversized batch, malformed JSON, invalid
bundle or model-output failure) abort the entire request. Partial row errors do not
make the CLI fail: exit 0 means the request was processed; inspect `summary.failed`
and each item, including the all-invalid case. Global input/file errors exit 2.
JSON parsing remains bounded to 1,000,000 bytes and rejects duplicate keys and invalid
constants before row isolation. `NaN` JSON is therefore a global parse error, whereas
an in-memory NaN value is a per-row numeric error. Outputs use exclusive creation and
never overwrite an existing file, but are not crash-atomic/durably fsynced writes.

This is a Python API and local JSON CLI, not an HTTP server. No network listener,
credentials, private data, paid service, concurrent queue or service SLA is introduced.
SHA256 checks integrity, not authenticity; only use independently trusted model
provenance. Bounds are neither scientific plausibility checks nor OOD detection.
Softmax probabilities are not calibrated confidence estimates.

## Reproduce

```bash
# Verify and measure the predefined batch sizes, retaining raw timings.
python -m pytorch_lab.wine_batch_inference_cli --output results/local-batch-inference.json
# Reuse the already published public model and 107 training-feature rows.
python -m pytorch_lab.wine_batch_inference_cli \
  --bundle results/2026-11-08-wine-model.json \
  --request results/2026-11-08-wine-request.json \
  --output results/local-batch-response.json
```

For a partial-success example, replace `--request` with
`examples/wine-batch-mixed-request.json`; the request contains two valid public
training rows, one shape error and one boolean-value error.
[Executed example response](../results/2026-11-10-mixed-batch-response.json) retains
all four indices with two successes and two failures. Its timings are one
functional demonstration, not an additional benchmark repeat.

The verification command first trains the unchanged fixture (140 updates /2140
training visits), explicitly **outside all latency measurements**. It then uses only
the public 107 training-feature rows, cyclically repeated for 1024. This is not a new
data sample. Every measured warm/fresh output is checked against the prior bundle
predictor (float64 logits rtol=atol=1e-12; exact class index). Mixed valid/invalid rows
are separately checked. The old full-suite Wine tests still execute their historical
held-out protocol; those scores do not select or evaluate this delivery increment.

## Cost boundaries

One CPU thread, float64; batch sizes 1,16,107,1024; one discarded warmup and three
retained repetitions per size. Samples, median, min and max are retained, with no
performance threshold or universal ranking. Measurements are sequential, not a
throughput or concurrent-load benchmark. Each repetition runs warm then fresh-process
in a fixed order, not a randomized causal comparison. CPU model/host load are not
recorded by the runner, which further limits comparison across machines.

- Warm in-memory total starts at request-envelope validation and ends after Python
  response items have been built. Model construction/load and JSON I/O are excluded.
- Validation covers envelope and per-row validation plus collection of accepted rows.
- Preprocessing covers tensor creation, train-fitted standardization and entering
  inference mode. Model-forward time covers the single model call only.
- Postprocessing includes finite-output verification, softmax, class extraction and
  assembling per-row predictions. Residual interpreter/timer overhead is not subtracted.
- Fresh-process file-to-file total is measured by the **parent** from subprocess launch
  through successful process exit, after the output file has been closed. It includes
  Python/PyTorch startup, imports, setting threads, read/parse, validation/load,
  preprocessing, inference, encoding/writing and exit. It excludes preparing input
  files, training, reading the completed output back for oracle checks and fsync.
- Child diagnostics end **before output encoding** and are deliberately not labeled
  total file-to-file time. Do not subtract independent warm and cold samples to claim
  a precise startup cost. The first discard does not clear OS caches; fresh process
  does not mean cold disk/page cache.

These are measured time/byte costs, not cloud billing estimates. The tiny model is
expected to make application overhead significant, but conclusions must follow actual
samples. Three repeats give a narrow local observation, not a latency distribution SLA.
[Raw final evidence](../results/2026-11-10-batch-inference-final.json) and
[verification record](../results/2026-11-10-batch-inference-verification.md) contain
measured values and the exact validation scope.

## Source study and learning value

Actual source reading: PyTorch v2.14.1
[`Module.train/eval`](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/nn/modules/module.py#L2894-L2935)
and [`inference_mode`](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/autograd/grad_mode.py#L213-L280).
`eval()` calls `train(False)` and changes module behavior; it does not disable AD.
`inference_mode()` disables AD and extra tracking but does not switch evaluation
behavior. The bundle loader sets eval, and the serving-only forward uses inference
mode. Its resulting tensors do not escape into a later training graph; output is
plain JSON-compatible data. No claim that inference_mode is faster than no_grad was
made or measured here. [PyTorch license](https://github.com/pytorch/pytorch/blob/v2.14.1/LICENSE)
and existing UCI/scikit-learn data notices remain applicable; no upstream code copied.

The engineering evidence is isolation of bad rows, preservation of ordering, exact
preprocessing semantics, bounded file handling, dependency-free error contracts,
installed package delivery and honest measurement boundaries. It is assistant-built
and tested work, not proof that the learner independently masters serving systems.
Current project entry: [train → evaluate → export → predict](project-path.md).
Any later HTTP work needs explicit timeout, backpressure and failure-handling tests;
avoid adding a framework without a concrete need.
