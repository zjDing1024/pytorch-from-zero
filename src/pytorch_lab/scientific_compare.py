"""Bounded, schema-aware diagnostic comparison of complete Wine result JSON.

The strict canonical digest is unchanged. A separate scientific comparison permits
at most eight adjacent finite binary64 representable steps only at explicitly
listed continuous-output paths. This is a chosen diagnostic budget, not a proven
error bound for iterative training or a cross-platform reproducibility guarantee.
"""

import argparse
import json
import math
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path

from pytorch_lab.runtime_probe import result_digest

MAX_ULPS = 8
MAX_EXAMPLES = 20
MAX_JSON_BYTES = 16 * 1024 * 1024
_IGNORED = {"created_at_utc", "environment"}
_ARCHITECTURES = ("linear", "mlp16")
_PARTITIONS = {"train": 107, "validation": 36, "test": 35}
_DATA_SHA256 = "10e8a802908b34f86e5da8ce962f3c806694bc98450a18f61851af59f324bede"
_CHECKS = (
    "all_rows_disjoint_and_covered",
    "no_cross_split_exact_feature_duplicates",
    "train_scaled_mean_zero",
    "train_scaled_variance_one",
    "equal_update_and_sample_budgets",
)


@dataclass(frozen=True)
class _Array:
    element: object
    minimum: int
    maximum: int


def _array(element: object, size: int) -> _Array:
    return _Array(element, size, size)


def _shape(value: object, spec: object, path: str = "$") -> None:
    """Require known Wine fields; extensions remain allowed and exact by default."""
    if isinstance(spec, dict):
        if type(value) is not dict or not spec.keys() <= value.keys():
            raise ValueError(f"missing or invalid Wine object at {path}")
        for key, child in spec.items():
            _shape(value[key], child, f"{path}.{key}")
    elif isinstance(spec, _Array):
        if type(value) is not list or not spec.minimum <= len(value) <= spec.maximum:
            raise ValueError(f"invalid Wine array at {path}")
        for item in value:
            _shape(item, spec.element, f"{path}[]")
    elif type(value) not in (spec if isinstance(spec, tuple) else (spec,)):
        raise ValueError(f"invalid Wine scalar type at {path}")


def _scan_json(value: object) -> None:
    pending, nodes = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 200_000 or depth > 32:
            raise ValueError("Wine input exceeds structural limits")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("JSON object keys must be strings")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite JSON number")
        elif type(item) not in (str, int, bool, type(None)):
            raise ValueError("non-JSON value")


def _metrics_schema(size: int) -> dict:
    return {
        "cross_entropy": float,
        "accuracy": float,
        "macro_f1": float,
        "confusion_matrix": _array(_array(int, 3), 3),
        "per_class": _array(
            {"class": int, "support": int, "precision": float, "recall": float, "f1": float},
            3,
        ),
        "errors": _Array(
            {
                "row_id": int,
                "true": int,
                "predicted": int,
                "true_probability": float,
                "predicted_probability": float,
            },
            0,
            size,
        ),
        "row_ids": _array(int, size),
        "predictions": _array(int, size),
        "probabilities": _array(_array(float, 3), size),
    }


def validate_result(value: object) -> None:
    """Check the bounded Wine-v1 contract, not retrain or rederive its metrics."""
    if (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.max_exp) != (2, 53, 1024):
        raise ValueError("IEEE binary64 Python floats required")
    _scan_json(value)
    summary = {
        name: {"mean": float, "sample_std": float, "values": _array(float, 3)}
        for name in ("accuracy", "macro_f1", "cross_entropy")
    }
    run = {
        "architecture": str,
        "seed": int,
        "parameters": int,
        "updates": int,
        "samples_seen": int,
        "history": _Array(
            {"epoch": int, "online_cross_entropy": float, "final_train_cross_entropy": float},
            1,
            51,
        ),
        **{name: _metrics_schema(size) for name, size in _PARTITIONS.items()},
    }
    _shape(
        value,
        {
            "schema_version": int,
            "created_at_utc": str,
            "environment": {
                "python": str,
                "torch": str,
                "device": str,
                "dtype": str,
                "threads": int,
            },
            "dataset": {
                "name": str,
                "sha256": str,
                "source": str,
                "license": str,
                "features": _array(str, 13),
            },
            "protocol": {
                "config": {
                    "epochs": int,
                    "batch_size": int,
                    "learning_rate": (int, float),
                    "momentum": (int, float),
                    "weight_decay": (int, float),
                },
                "seeds": _array(int, 3),
                "split_salt": str,
                "split_row_ids": {name: _array(int, size) for name, size in _PARTITIONS.items()},
                "selection": str,
                "selected_architecture": str,
                "test_usage": str,
                "seed_variation": str,
            },
            "audit": {
                "sizes": dict.fromkeys(_PARTITIONS, int),
                "class_counts": {name: _array(int, 3) for name in _PARTITIONS},
                "duplicate_feature_rows": int,
                "source_rows_partitioned_once": bool,
                "cross_split_feature_duplicates": int,
            },
            "preprocessing": {
                "fit_partition": str,
                "correction": int,
                "mean": _array(float, 13),
                "scale": _array(float, 13),
            },
            "training_prior_baseline": {"priors": _array(float, 3), "test": _metrics_schema(35)},
            "models": {
                name: {"runs": _array(run, 3), "validation": summary, "test": summary}
                for name in _ARCHITECTURES
            },
            "checks": dict.fromkeys(_CHECKS, bool),
            "all_checks_passed": bool,
            "limits": _Array(str, 1, 100),
        },
    )
    environment, protocol, dataset = value["environment"], value["protocol"], value["dataset"]
    if value["schema_version"] != 1 or dataset["name"] != "UCI Wine":
        raise ValueError("unsupported Wine schema or dataset")
    if dataset["sha256"] != _DATA_SHA256 or len(set(dataset["features"])) != 13:
        raise ValueError("invalid Wine dataset identity")
    if (
        environment["device"] != "cpu"
        or environment["dtype"] != "float64"
        or environment["threads"] != 1
        or not environment["python"].strip()
        or not environment["torch"].strip()
    ):
        raise ValueError("invalid Wine runtime metadata")
    timestamp = datetime.fromisoformat(value["created_at_utc"])
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError("Wine timestamp must specify UTC")
    config = protocol["config"]
    if (
        not 1 <= config["epochs"] <= 1000
        or not 1 <= config["batch_size"] <= 107
        or config["learning_rate"] <= 0
        or not 0 <= config["momentum"] < 1
        or config["weight_decay"] < 0
        or protocol["seeds"] != [42, 7, 123]
        or protocol["selected_architecture"] not in _ARCHITECTURES
        or set(value["models"]) != set(_ARCHITECTURES)
    ):
        raise ValueError("invalid Wine protocol")
    split = protocol["split_row_ids"]
    if sorted(row for name in _PARTITIONS for row in split[name]) != list(range(178)):
        raise ValueError("invalid Wine source-row partition")
    if value["all_checks_passed"] is not True or any(
        value["checks"][key] is not True for key in _CHECKS
    ):
        raise ValueError("failed Wine protocol checks")
    for name in _ARCHITECTURES:
        for index, item in enumerate(value["models"][name]["runs"]):
            if item["architecture"] != name or item["seed"] != protocol["seeds"][index]:
                raise ValueError("invalid Wine run identity or order")
            for partition in _PARTITIONS:
                _validate_metric_indices(item[partition], split[partition])
    _validate_metric_indices(value["training_prior_baseline"]["test"], split["test"])


def _validate_metric_indices(metric: dict, ids: list[int]) -> None:
    if metric["row_ids"] != ids or any(p not in (0, 1, 2) for p in metric["predictions"]):
        raise ValueError("invalid Wine metric rows or predictions")
    if [item["class"] for item in metric["per_class"]] != [0, 1, 2]:
        raise ValueError("invalid Wine metric class order")
    if any(count < 0 for row in metric["confusion_matrix"] for count in row):
        raise ValueError("invalid Wine confusion count")
    for error in metric["errors"]:
        if (
            error["row_id"] not in ids
            or error["true"] not in (0, 1, 2)
            or error["predicted"] not in (0, 1, 2)
        ):
            raise ValueError("invalid Wine error identity")


def ulp_distance(left: float, right: float) -> int:
    """Count adjacent binary64 steps, collapsing the two zero encodings to zero."""
    if any(type(value) is not float or not math.isfinite(value) for value in (left, right)):
        raise ValueError("ULP distance requires finite binary64 floats")

    def rank(value: float) -> int:
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]
        magnitude = bits & ((1 << 63) - 1)
        return -magnitude if bits >> 63 else magnitude

    return abs(rank(left) - rank(right))


def _continuous(path: tuple) -> bool:
    """Match complete paths; a familiar leaf name elsewhere gets no tolerance."""
    if len(path) == 3 and path[:2] in (("preprocessing", "mean"), ("preprocessing", "scale")):
        return type(path[2]) is int
    if path[:2] == ("training_prior_baseline", "test"):
        tail = path[2:]
    elif (
        len(path) >= 5
        and path[0] == "models"
        and path[1] in _ARCHITECTURES
        and path[2] == "runs"
        and type(path[3]) is int
    ):
        if path[4] == "history":
            return (
                len(path) == 7
                and type(path[5]) is int
                and path[6] in {"online_cross_entropy", "final_train_cross_entropy"}
            )
        if path[4] not in _PARTITIONS:
            return False
        tail = path[5:]
    elif (
        len(path) >= 5
        and path[0] == "models"
        and path[1] in _ARCHITECTURES
        and path[2] in {"validation", "test"}
        and path[3] == "cross_entropy"
    ):
        return (len(path) == 5 and path[4] in {"mean", "sample_std"}) or (
            len(path) == 6 and path[4] == "values" and type(path[5]) is int
        )
    else:
        return False
    return (
        tail == ("cross_entropy",)
        or (
            len(tail) == 3
            and tail[0] == "probabilities"
            and all(type(index) is int for index in tail[1:])
        )
        or (
            len(tail) == 3
            and tail[0] == "errors"
            and type(tail[1]) is int
            and tail[2] in {"true_probability", "predicted_probability"}
        )
    )


def _path(parts: tuple) -> str:
    return "$" + "".join(
        f"[{part}]" if type(part) is int else f"[{json.dumps(part)}]" for part in parts
    )


def _shown(value: object) -> object:
    if type(value) in (dict, list):
        return {"type": type(value).__name__, "length": len(value)}
    return value[:200] + "..." if type(value) is str and len(value) > 200 else value


def compare_results(left: dict, right: dict) -> dict:
    """Return strict and bounded-scientific outcomes; reject malformed inputs."""
    validate_result(left)
    validate_result(right)
    exact = {
        "passed": result_digest(left) == result_digest(right),
        "left_result_sha256": result_digest(left),
        "right_result_sha256": result_digest(right),
        "mismatch_count": 0,
        "mismatch_categories": {},
        "examples": [],
    }
    approximate = {
        "passed": True,
        "exact_required_mismatch_count": 0,
        "continuous_ulp_failure_count": 0,
        "examples": [],
    }
    statistics = dict.fromkeys(
        (
            "compared_numeric_leaves",
            "compared_float_leaves",
            "continuous_float_leaves",
            "float_value_difference_count",
            "float_bit_difference_count",
            "signed_zero_difference_count",
        ),
        0,
    )
    statistics.update(max_absolute_difference=None, max_ulp_distance=None)
    categories = Counter()

    def mismatch(
        path: tuple,
        a: object,
        b: object,
        category: str,
        failure: str | None,
        details: dict | None = None,
    ) -> None:
        exact["mismatch_count"] += 1
        categories[category] += 1
        example = {"path": _path(path), "category": category, "left": _shown(a), "right": _shown(b)}
        example.update(details or {})
        if len(exact["examples"]) < MAX_EXAMPLES:
            exact["examples"].append(example)
        if failure is not None:
            approximate[failure] += 1
            approximate["passed"] = False
            if len(approximate["examples"]) < MAX_EXAMPLES:
                approximate["examples"].append(example)

    def visit(a: object, b: object, path: tuple) -> None:
        if type(a) in (int, float) and type(b) in (int, float):
            statistics["compared_numeric_leaves"] += 1
        if type(a) is not type(b):
            mismatch(path, a, b, "type", "exact_required_mismatch_count")
        elif type(a) is dict:
            for key in sorted(a.keys() | b.keys()):
                if not path and key in _IGNORED:
                    continue
                if key not in a or key not in b:
                    mismatch(
                        path + (key,),
                        key in a,
                        key in b,
                        "key_presence",
                        "exact_required_mismatch_count",
                    )
                else:
                    visit(a[key], b[key], path + (key,))
        elif type(a) is list:
            if len(a) != len(b):
                mismatch(path, a, b, "list_length", "exact_required_mismatch_count")
            for index, (x, y) in enumerate(zip(a, b, strict=False)):
                visit(x, y, path + (index,))
        elif type(a) is float:
            statistics["compared_float_leaves"] += 1
            continuous = _continuous(path)
            statistics["continuous_float_leaves"] += int(continuous)
            distance, delta = ulp_distance(a, b), abs(a - b)
            overflow = not math.isfinite(delta)
            if overflow:
                with localcontext() as context:
                    context.prec = 1100
                    delta = abs(Decimal.from_float(a) - Decimal.from_float(b))
            maximum = statistics["max_absolute_difference"]
            previous = (
                None
                if maximum is None
                else (Decimal(maximum["value"]) if maximum["overflow"] else maximum["value"])
            )
            if previous is None or delta > previous:
                statistics["max_absolute_difference"] = {
                    "value": str(delta) if overflow else delta,
                    "overflow": overflow,
                    "path": _path(path),
                    "left": a,
                    "right": b,
                }
            maximum = statistics["max_ulp_distance"]
            if maximum is None or distance > maximum["value"]:
                statistics["max_ulp_distance"] = {
                    "value": distance,
                    "path": _path(path),
                    "left": a,
                    "right": b,
                }
            if struct.pack(">d", a) != struct.pack(">d", b):
                statistics["float_bit_difference_count"] += 1
                if a == b:
                    statistics["signed_zero_difference_count"] += 1
                    mismatch(
                        path,
                        a,
                        b,
                        "signed_zero",
                        None if continuous else "exact_required_mismatch_count",
                    )
                else:
                    statistics["float_value_difference_count"] += 1
                    failure = (
                        ("continuous_ulp_failure_count" if distance > MAX_ULPS else None)
                        if (continuous)
                        else "exact_required_mismatch_count"
                    )
                    mismatch(
                        path,
                        a,
                        b,
                        "continuous_float" if continuous else "exact_float",
                        failure,
                        {
                            "ulp_distance": distance,
                            "absolute_difference": str(delta) if overflow else delta,
                            "absolute_difference_overflow": overflow,
                            "allowed_ulps": MAX_ULPS if continuous else 0,
                        },
                    )
        elif a != b:
            mismatch(path, a, b, "exact_scalar", "exact_required_mismatch_count")

    visit(left, right, ())
    exact["mismatch_categories"] = dict(sorted(categories.items()))
    approximate["exact_fields_passed"] = approximate["exact_required_mismatch_count"] == 0
    approximate["continuous_floats_passed"] = approximate["continuous_ulp_failure_count"] == 0
    excluded_left = {key: left[key] for key in sorted(_IGNORED)}
    excluded_right = {key: right[key] for key in sorted(_IGNORED)}
    return {
        "schema_version": 1,
        "policy": {
            "name": "wine-float64-lastbits-v1",
            "max_ulps": MAX_ULPS,
            "absolute_tolerance": 0.0,
            "continuous_paths": [
                "preprocessing.{mean,scale}[*]",
                "training_prior_baseline.test.{cross_entropy,probabilities[*][*]}",
                "training_prior_baseline.test.errors[*].{true_probability,predicted_probability}",
                "models.{linear,mlp16}.runs[*].{train,validation,test}.cross_entropy",
                "models.{linear,mlp16}.runs[*].{train,validation,test}.probabilities[*][*]",
                "models.{linear,mlp16}.runs[*].{train,validation,test}.errors[*]."
                "{true_probability,predicted_probability}",
                "models.{linear,mlp16}.runs[*].history[*]."
                "{online_cross_entropy,final_train_cross_entropy}",
                "models.{linear,mlp16}.{validation,test}.cross_entropy.{mean,sample_std,values[*]}",
            ],
            "default": "Exact types, structure, nonfloats and all unlisted floats; priors and "
            "count-derived metrics/summaries are exact. Object key order is irrelevant; "
            "array order is significant. Signed zeros are numerically equal only on "
            "allow-listed continuous paths; all unlisted floats retain encoding equality.",
            "strict_digest": "Unmodified runtime_probe.result_digest; excludes only root "
            "created_at_utc and environment; signed zero encodings differ.",
            "excluded_root_fields": sorted(_IGNORED),
            "max_examples_per_outcome": MAX_EXAMPLES,
            "limits": "At most 8 adjacent finite binary64 representable steps is a predeclared "
            "diagnostic budget, not a derived error bound for iterative training, "
            "a count of differing mantissa bits, or a general platform guarantee. "
            "Validation checks the Wine schema and reported checks, not metric truth.",
        },
        "exact": exact,
        "approximate": approximate,
        "statistics": statistics,
        "excluded_metadata": {
            "identical": json.dumps(excluded_left, sort_keys=True)
            == json.dumps(excluded_right, sort_keys=True),
            "left": excluded_left,
            "right": excluded_right,
        },
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON constant")


def load_result(path: Path) -> dict:
    with path.open("rb") as source:
        payload = source.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("Wine input exceeds byte limit")
    result = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    validate_result(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="New report path; never overwrite"
    )
    parser.add_argument("--require-exact", action="store_true", help="Gate on the strict digest")
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("report already exists")
        report = compare_results(load_result(args.left), load_result(args.right))
        report["gate"] = {
            "mode": "exact" if args.require_exact else "approximate",
            "passed": report["exact" if args.require_exact else "approximate"]["passed"],
        }
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        # Exclusive creation also rejects a concurrent file or dangling symlink.
        with args.output.open("x", encoding="utf-8") as target:
            target.write(encoded)
        print(encoded, end="")
        if not report["gate"]["passed"]:
            parser.exit(1, "Wine comparison failed; the report was retained.\n")
    except (ValueError, TypeError, KeyError, OSError, RecursionError, OverflowError):
        parser.exit(2, "Wine comparison rejected: invalid input or unavailable new report path.\n")


if __name__ == "__main__":
    main()
