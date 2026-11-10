"""Bounded, index-preserving partial batch inference on the fixed Wine bundle."""

import math
from time import perf_counter

import torch

from pytorch_lab import wine
from pytorch_lab.wine_model_bundle import LIMIT, MAX_ROWS, load_payload


def _row_error(row):
    if type(row) is not list or len(row) != 13:
        return "invalid_shape"
    # Test magnitude before isfinite: arbitrarily large Python ints cannot overflow conversion.
    if any(type(v) not in (int, float) or abs(v) > LIMIT or not math.isfinite(v) for v in row):
        return "invalid_number"
    return None


class WineBatchPredictor:
    """Load/validate a trusted bundle once; valid rows share one forward call."""

    def __init__(self, bundle):
        self.model, self.scaler = load_payload(bundle)

    def predict(self, request):
        start = perf_counter()
        if type(request) is not dict or set(request) != {"features", "rows"}:
            raise ValueError("invalid request fields")
        if request["features"] != list(wine.FEATURE_NAMES):
            raise ValueError("feature order mismatch")
        rows = request["rows"]
        if type(rows) is not list or not 1 <= len(rows) <= MAX_ROWS:
            raise ValueError("batch size must be in [1, 1024]")
        items, valid, indices = [], [], []
        for index, row in enumerate(rows):
            error = _row_error(row)
            items.append({"index": index, "ok": error is None})
            if error:
                items[-1]["error"] = {"code": error}
            else:
                valid.append(row)
                indices.append(index)
        validated = perf_counter()
        preprocess_seconds = model_seconds = postprocess_seconds = 0.0
        if valid:
            with torch.inference_mode():
                x = self.scaler.transform(torch.tensor(valid, dtype=torch.float64))
                preprocessed = perf_counter()
                logits = self.model(x)
                forwarded = perf_counter()
                if not torch.isfinite(logits).all():
                    # This is a model/runtime failure, not a row validation failure.
                    raise ValueError("nonfinite model output")
                probabilities = logits.softmax(1).tolist()
                labels = logits.argmax(1).tolist()
                for index, logit, probability, label in zip(
                    indices, logits.tolist(), probabilities, labels, strict=True
                ):
                    items[index]["prediction"] = {
                        "logits": logit,
                        "probabilities": probability,
                        "class_index": label,
                        "class": f"class_{label}",
                    }
                postprocessed = perf_counter()
                preprocess_seconds = preprocessed - validated
                model_seconds = forwarded - preprocessed
                postprocess_seconds = postprocessed - forwarded
        return {
            "schema": "wine-batch-response-v1",
            "items": items,
            "summary": {
                "requested": len(rows),
                "succeeded": len(valid),
                "failed": len(rows) - len(valid),
            },
            "timings_seconds": {
                "validation": validated - start,
                "preprocessing": preprocess_seconds,
                "model_forward": model_seconds,
                "postprocessing": postprocess_seconds,
                "in_memory_total": perf_counter() - start,
            },
        }
