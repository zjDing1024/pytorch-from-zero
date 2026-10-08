"""Independent data, math and isolation checks for the fixed Wine protocol."""

import copy
import csv
import hashlib
import io
import json
import math
from dataclasses import FrozenInstanceError, asdict, replace
from importlib.resources import files

import pytest
import torch

import pytorch_lab.wine as wine

PINNED_SHA256 = "10e8a802908b34f86e5da8ce962f3c806694bc98450a18f61851af59f324bede"
VALIDATION_IDS = [
    8,
    13,
    14,
    24,
    25,
    28,
    31,
    34,
    38,
    48,
    53,
    57,
    61,
    63,
    64,
    69,
    84,
    90,
    92,
    95,
    102,
    107,
    113,
    115,
    124,
    128,
    130,
    135,
    143,
    146,
    157,
    162,
    164,
    167,
    170,
    175,
]
TEST_IDS = [
    5,
    10,
    11,
    33,
    35,
    39,
    42,
    43,
    44,
    50,
    55,
    58,
    60,
    68,
    70,
    71,
    72,
    75,
    78,
    81,
    85,
    94,
    108,
    111,
    121,
    123,
    134,
    137,
    150,
    152,
    155,
    156,
    159,
    160,
    172,
]


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


@pytest.fixture
def data():
    x, y = wine.load_wine()
    return x, y, wine.build_split(y)


@pytest.fixture(scope="module")
def smoke():
    return wine.run_wine_experiment(wine.WineConfig(epochs=1))


def test_pinned_offline_bytes_schema_and_class_counts():
    payload = files("pytorch_lab").joinpath("data/wine.csv").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PINNED_SHA256 == wine.DATA_SHA256
    rows = list(csv.reader(io.StringIO(payload.decode())))
    assert rows[0] == ["178", "13", "class_0", "class_1", "class_2"]
    assert len(rows) == 179 and {len(row) for row in rows[1:]} == {14}
    x, y = wine.load_wine()
    assert x.shape == (178, 13) and y.shape == (178,)
    assert x.dtype == torch.float64 and y.dtype == torch.int64
    assert x.device.type == y.device.type == "cpu"
    assert torch.isfinite(x).all()
    assert torch.bincount(y).tolist() == [59, 71, 48]
    assert len(set(wine.FEATURE_NAMES)) == 13
    assert x[0].tolist() == [
        14.23,
        1.71,
        2.43,
        15.6,
        127.0,
        2.8,
        3.06,
        0.28,
        2.29,
        5.64,
        1.04,
        3.92,
        1065.0,
    ]


def test_loader_rejects_changed_bytes_and_independently_checks_schema(tmp_path, monkeypatch):
    original = files("pytorch_lab").joinpath("data/wine.csv").read_bytes()
    folder = tmp_path / "data"
    folder.mkdir()
    path = folder / "wine.csv"
    path.write_bytes(original + b"\n")
    monkeypatch.setattr(wine, "files", lambda package: tmp_path)
    with pytest.raises(ValueError, match="integrity"):
        wine.load_wine()

    rows = list(csv.reader(io.StringIO(original.decode())))
    bad_header, bad_width, bad_count, bad_classes = [copy.deepcopy(rows) for _ in range(4)]
    bad_header[0][0] = "177"
    bad_width[1].pop()
    bad_count.pop()
    bad_classes[1][-1] = "1"
    for changed in (bad_header, bad_width, bad_count, bad_classes):
        buffer = io.StringIO()
        csv.writer(buffer).writerows(changed)
        payload = buffer.getvalue().encode()
        path.write_bytes(payload)
        # Re-sign intentionally so the structural contract is exercised after integrity.
        monkeypatch.setattr(wine, "DATA_SHA256", hashlib.sha256(payload).hexdigest())
        with pytest.raises(ValueError, match="unexpected Wine"):
            wine.load_wine()


def test_fixed_split_matches_independent_ids_manifest_and_exact_counts(data):
    x, y, split = data
    held_out = set(VALIDATION_IDS + TEST_IDS)
    assert split == {
        "train": [i for i in range(178) if i not in held_out],
        "validation": VALIDATION_IDS,
        "test": TEST_IDS,
    }
    manifest = json.loads(files("pytorch_lab").joinpath("data/wine_split.json").read_text())
    assert manifest["data_sha256"] == PINNED_SHA256
    assert manifest["salt"] == wine.SPLIT_SALT == "wine-split-v1:20261009"
    assert manifest["row_ids"] == split == wine.build_split(y.clone())
    assert wine.audit_split(x, y, split) == {
        "sizes": {"train": 107, "validation": 36, "test": 35},
        "class_counts": {"train": [35, 43, 29], "validation": [12, 14, 10], "test": [12, 14, 9]},
        "duplicate_feature_rows": 0,
        "source_rows_partitioned_once": True,
        "cross_split_feature_duplicates": 0,
    }


def test_split_audit_rejects_overlap_omissions_bad_ids_and_missing_classes(data):
    x, y, split = data
    overlap, omitted, bool_id, bad_keys, missing_class = [copy.deepcopy(split) for _ in range(5)]
    overlap["test"][0] = overlap["train"][0]
    omitted["train"].pop()
    bool_id["train"][0] = False
    bad_keys["valid"] = bad_keys.pop("validation")
    moved = [i for i in missing_class["validation"] if y[i] == 0]
    missing_class["validation"] = [i for i in missing_class["validation"] if y[i] != 0]
    missing_class["train"].extend(moved)
    for invalid in (overlap, omitted, bool_id, bad_keys, missing_class):
        with pytest.raises(ValueError):
            wine.audit_split(x, y, invalid)


def test_cross_split_duplicates_rejected_within_split_duplicates_reported(data):
    x, y, split = data
    source, within = split["train"][:2]
    duplicate = x.clone()
    duplicate[within] = duplicate[source]
    audit = wine.audit_split(duplicate, y, split)
    assert audit["duplicate_feature_rows"] == 1
    assert audit["cross_split_feature_duplicates"] == 0
    duplicate[split["validation"][0]] = duplicate[source]
    with pytest.raises(ValueError, match="cross split"):
        wine.audit_split(duplicate, y, split)


def test_standardizer_population_statistics_constants_and_detached_state():
    x = torch.tensor(
        [[1.0, 5.0, 2.0], [3.0, 5.0, 4.0], [5.0, 5.0, 6.0]], dtype=torch.float64, requires_grad=True
    )
    scaler = wine.Standardizer.fit(x)
    torch.testing.assert_close(scaler.mean, torch.tensor([3.0, 5.0, 4.0], dtype=torch.float64))
    expected_scale = torch.tensor([math.sqrt(8 / 3), 1.0, math.sqrt(8 / 3)], dtype=torch.float64)
    torch.testing.assert_close(scaler.scale, expected_scale)
    assert not scaler.mean.requires_grad and not scaler.scale.requires_grad
    transformed = scaler.transform(x)
    torch.testing.assert_close(transformed.mean(0), torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(
        transformed.var(0, correction=0), torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    )
    assert not torch.equal(scaler.scale, x.detach().std(0, correction=1))
    with torch.no_grad():
        x.fill_(100)
    assert scaler.mean.tolist() == [3.0, 5.0, 4.0]


def test_standardizer_train_only_statistics_unchanged_by_heldout_perturbation(data):
    x, _, split = data
    train = x[split["train"]]
    original = wine.Standardizer.fit(train)
    changed = x.clone()
    changed[split["validation"]] += 1e6
    changed[split["test"]] *= -1e6
    fitted = wine.Standardizer.fit(changed[split["train"]])
    assert torch.equal(original.mean, fitted.mean)
    assert torch.equal(original.scale, fitted.scale)
    torch.testing.assert_close(original.mean, train.mean(0))
    torch.testing.assert_close(original.scale, train.std(0, correction=0))
    assert not torch.equal(original.mean, x.mean(0))
    assert torch.equal(original.transform(train), fitted.transform(train))


def test_feature_and_target_contracts_reject_malformed_training_inputs(data):
    x, y, _ = data
    bad_features = [x.float(), x[:0], x[:, :0], x[0], x.clone()]
    bad_features[-1][0, 0] = float("nan")
    for invalid in bad_features:
        with pytest.raises(ValueError):
            wine.Standardizer.fit(invalid)
        with pytest.raises(ValueError):
            wine.fit_candidate(invalid, y, "linear", 42, wine.WineConfig(epochs=1))
    for invalid in (y.float(), y[:, None], y[:-1], y - 1, y + 1):
        with pytest.raises(ValueError):
            wine.fit_candidate(x, invalid, "linear", 42, wine.WineConfig(epochs=1))
    with pytest.raises(ValueError, match="13 features"):
        wine.fit_candidate(x[:, :-1], y, "linear", 42, wine.WineConfig(epochs=1))
    with pytest.raises(ValueError, match="width"):
        wine.Standardizer.fit(x).transform(x[:, :-1])


def test_cross_entropy_uses_raw_logits_manual_logsumexp_and_shift_invariance():
    logits = torch.tensor(
        [[1000.0, 1001.0, 999.0], [-4.0, 2.0, 0.0], [0.1, 0.2, 0.3]], dtype=torch.float64
    )
    labels = torch.tensor([2, 1, 0])
    losses = []
    for row, label in zip(logits.tolist(), labels.tolist(), strict=True):
        maximum = max(row)
        losses.append(maximum + math.log(sum(math.exp(v - maximum) for v in row)) - row[label])
    actual = wine.metrics(logits, labels, [10, 11, 12])
    shifted = wine.metrics(logits + 2048, labels, [10, 11, 12])
    assert actual["cross_entropy"] == pytest.approx(sum(losses) / 3, abs=1e-12)
    assert shifted["cross_entropy"] == pytest.approx(actual["cross_entropy"], abs=1e-12)
    assert shifted["predictions"] == actual["predictions"]
    torch.testing.assert_close(
        torch.tensor(shifted["probabilities"]), torch.tensor(actual["probabilities"])
    )
    # Passing already-normalized probabilities must not accidentally match this objective.
    wrong = wine.metrics(logits.softmax(1), labels, [10, 11, 12])
    assert abs(actual["cross_entropy"] - wrong["cross_entropy"]) > 0.1


def test_confusion_orientation_macro_f1_and_error_row_ids_have_known_answers():
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
    predicted = torch.tensor([0, 0, 1, 0, 1, 2, 2, 2])
    logits = torch.zeros(8, 3, dtype=torch.float64)
    logits[torch.arange(8), predicted] = 2
    report = wine.metrics(logits, labels, list(range(8, 0, -1)))
    assert report["confusion_matrix"] == [[2, 1, 0], [1, 1, 1], [0, 0, 2]]
    assert report["accuracy"] == 5 / 8
    assert report["macro_f1"] == pytest.approx(28 / 45)
    assert [item["f1"] for item in report["per_class"]] == pytest.approx([2 / 3, 2 / 5, 4 / 5])
    assert [item["support"] for item in report["per_class"]] == [3, 3, 2]
    assert [item["row_id"] for item in report["errors"]] == [3, 5, 6]
    for error in report["errors"]:
        assert error["predicted_probability"] == pytest.approx(math.exp(2) / (math.exp(2) + 2))
        assert error["true_probability"] == pytest.approx(1 / (math.exp(2) + 2))


def test_metrics_zero_denominators_and_invalid_shapes():
    logits = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    report = wine.metrics(logits, labels, [0, 1])
    assert report["macro_f1"] == pytest.approx(2 / 9)
    assert report["per_class"][2] == {
        "class": 2,
        "support": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    assert report["per_class"][1]["precision"] == report["per_class"][1]["f1"] == 0
    for values, targets, ids in (
        (logits[:, :2], labels, [0, 1]),
        (logits, labels, [0]),
        (logits, labels + 2, [0, 1]),
    ):
        with pytest.raises(ValueError):
            wine.metrics(values, targets, ids)


def test_classifier_parameter_counts_outputs_and_seed_contracts():
    for architecture, count in (("linear", 42), ("mlp16", 275)):
        model = wine.WineClassifier(architecture, 42)
        assert sum(p.numel() for p in model.parameters()) == count
        assert all(p.dtype == torch.float64 and p.device.type == "cpu" for p in model.parameters())
        logits = model(torch.zeros(4, 13, dtype=torch.float64))
        assert logits.shape == (4, 3) and torch.equal(logits, torch.zeros_like(logits))
        assert not torch.allclose(logits.sum(1), torch.ones(4, dtype=torch.float64))
    for architecture, seed in (
        ("unknown", 42),
        ("linear", True),
        ("linear", -1),
        ("mlp16", 2**63),
        ("mlp16", 1.2),
    ):
        with pytest.raises(ValueError):
            wine.WineClassifier(architecture, seed)


def test_local_initialization_and_minibatching_preserve_global_rng(data):
    x, y, split = data
    train = wine.Standardizer.fit(x[split["train"]]).transform(x[split["train"]])
    before = torch.random.get_rng_state().clone()
    for architecture in wine.ARCHITECTURES:
        wine.WineClassifier(architecture, 123)
        wine.fit_candidate(train, y[split["train"]], architecture, 7, wine.WineConfig(epochs=1))
    assert torch.equal(torch.random.get_rng_state(), before)


def test_both_architectures_receive_equal_full_tail_update_and_sample_budgets(data):
    x, y, split = data
    train = wine.Standardizer.fit(x[split["train"]]).transform(x[split["train"]])
    config = wine.WineConfig(epochs=2, batch_size=16)
    reports = []
    for architecture in wine.ARCHITECTURES:
        model, report = wine.fit_candidate(train, y[split["train"]], architecture, 42, config)
        assert report["updates"] == 14  # Six full batches and one 11-row tail per epoch.
        assert report["samples_seen"] == 214
        assert [row["epoch"] for row in report["history"]] == [1, 2]
        assert not model.training and all(p.grad is None for p in model.parameters())
        reports.append(report)
    assert reports[0]["updates"] == reports[1]["updates"]
    assert reports[0]["samples_seen"] == reports[1]["samples_seen"]


def test_training_repeatability_is_exact_and_different_seeds_change_parameters(data):
    x, y, split = data
    train = wine.Standardizer.fit(x[split["train"]]).transform(x[split["train"]])
    config = wine.WineConfig(epochs=2)
    for architecture in wine.ARCHITECTURES:
        first, first_report = wine.fit_candidate(train, y[split["train"]], architecture, 42, config)
        repeat, repeat_report = wine.fit_candidate(
            train, y[split["train"]], architecture, 42, config
        )
        assert first_report == repeat_report
        assert all(
            torch.equal(value, repeat.state_dict()[name])
            for name, value in first.state_dict().items()
        )
        changed = wine.WineClassifier(architecture, 7)
        initial = wine.WineClassifier(architecture, 42)
        assert not torch.equal(changed.weights[0], initial.weights[0])


def test_selection_uses_three_seed_mean_and_parameter_count_tiebreak():
    assert wine.select_architecture({"linear": [0.1, 0.9, 0.9], "mlp16": [0.5] * 3}) == "mlp16"
    assert wine.select_architecture({"mlp16": [0.5] * 3, "linear": [0.5] * 3}) == "linear"
    for values in (
        {"linear": [0.1] * 3},
        {"linear": [0.1], "mlp16": [0.2] * 3},
        {"linear": [float("nan")] * 3, "mlp16": [0.2] * 3},
        {"linear": [-0.1] * 3, "mlp16": [0.2] * 3},
    ):
        with pytest.raises(ValueError):
            wine.select_architecture(values)


def test_selection_happens_before_test_scoring_even_when_test_prefers_other_model(
    data, monkeypatch
):
    _, y, split = data
    events = []
    original_select, original_metrics = wine.select_architecture, wine.metrics

    def fake_fit(train_x, train_y, architecture, seed, config):
        assert len(train_x) == len(train_y) == 107
        events.append(("fit", architecture, seed))

        class ScriptedModel(torch.nn.Module):
            def forward(self, features):
                assert not self.training and not torch.is_grad_enabled()
                partition = {107: "train", 36: "validation", 35: "test"}[len(features)]
                targets = y[split[partition]]
                preferred = "mlp16" if partition == "test" else "linear"
                predicted = targets if architecture == preferred else (targets + 1) % 3
                logits = torch.zeros(len(features), 3, dtype=torch.float64)
                logits[torch.arange(len(features)), predicted] = 5
                return logits

        return ScriptedModel(), {
            "architecture": architecture,
            "seed": seed,
            "updates": 7,
            "samples_seen": 107,
            "history": [],
        }

    def tracked_select(losses):
        assert len([event for event in events if event[0] == "fit"]) == 6
        events.append(("select",))
        return original_select(losses)

    def tracked_metrics(logits, targets, row_ids):
        if row_ids == split["test"]:
            assert ("select",) in events
        events.append(("metrics", len(row_ids)))
        return original_metrics(logits, targets, row_ids)

    monkeypatch.setattr(wine, "fit_candidate", fake_fit)
    monkeypatch.setattr(wine, "select_architecture", tracked_select)
    monkeypatch.setattr(wine, "metrics", tracked_metrics)
    result = wine.run_wine_experiment(wine.WineConfig(epochs=1))
    assert result["protocol"]["selected_architecture"] == "linear"
    assert result["models"]["mlp16"]["test"]["accuracy"]["mean"] == 1
    assert result["models"]["linear"]["test"]["accuracy"]["mean"] == 0
    assert (
        len([event for event in events[: events.index(("select",))] if event == ("metrics", 36)])
        == 6
    )


def test_config_is_frozen_explicit_and_rejects_malformed_values():
    config = wine.WineConfig()
    assert asdict(config) == {
        "epochs": 120,
        "batch_size": 16,
        "learning_rate": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.0001,
    }
    config.validate()
    with pytest.raises(FrozenInstanceError):
        config.epochs = 1
    invalid = (
        [{"epochs": value} for value in (0, -1, 1001, True, 1.5)]
        + [{"batch_size": value} for value in (0, 108, False, 1.5)]
        + [
            {"learning_rate": 0},
            {"learning_rate": float("nan")},
            {"learning_rate": "0.1"},
            {"momentum": -0.1},
            {"momentum": 1},
            {"momentum": True},
            {"weight_decay": -1},
            {"weight_decay": float("inf")},
        ]
    )
    for values in invalid:
        with pytest.raises(ValueError):
            replace(config, **values).validate()
    replace(config, epochs=1000, batch_size=107, momentum=0, weight_decay=0).validate()


def test_one_epoch_full_protocol_all_seeds_models_baseline_and_json(smoke, data):
    x, y, split = data
    assert smoke["schema_version"] == 1 and smoke["all_checks_passed"]
    assert smoke["protocol"]["seeds"] == [42, 7, 123]
    assert smoke["protocol"]["config"]["epochs"] == 1
    assert smoke["protocol"]["split_row_ids"] == split
    assert smoke["audit"]["sizes"] == {"train": 107, "validation": 36, "test": 35}
    assert smoke["environment"]["threads"] == 1
    assert smoke["preprocessing"]["fit_partition"] == "train"
    assert smoke["preprocessing"]["mean"] == x[split["train"]].mean(0).tolist()
    assert all(smoke["checks"].values())
    for architecture in wine.ARCHITECTURES:
        runs = smoke["models"][architecture]["runs"]
        assert [run["seed"] for run in runs] == [42, 7, 123]
        for run in runs:
            assert run["architecture"] == architecture
            assert run["updates"] == 7 and run["samples_seen"] == 107
            for partition in ("train", "validation", "test"):
                assert run[partition]["row_ids"] == split[partition]
                assert len(run[partition]["predictions"]) == len(split[partition])
                assert 0 <= run[partition]["accuracy"] <= 1
                assert 0 <= run[partition]["macro_f1"] <= 1
        values = [run["test"]["accuracy"] for run in runs]
        mean = sum(values) / 3
        sample_std = math.sqrt(sum((value - mean) ** 2 for value in values) / 2)
        assert smoke["models"][architecture]["test"]["accuracy"]["sample_std"] == sample_std
    baseline = smoke["training_prior_baseline"]
    assert baseline["priors"] == pytest.approx([35 / 107, 43 / 107, 29 / 107])
    assert baseline["test"]["predictions"] == [1] * 35
    assert baseline["test"]["accuracy"] == pytest.approx(14 / 35)
    assert json.loads(json.dumps(smoke, allow_nan=False)) == smoke


def test_full_experiment_test_feature_perturbation_cannot_change_selection_or_training(
    smoke, data, monkeypatch
):
    x, y, split = data
    changed = x.clone()
    changed[split["test"]] += 1000
    monkeypatch.setattr(wine, "load_wine", lambda: (changed, y))
    before = torch.random.get_rng_state().clone()
    result = wine.run_wine_experiment(wine.WineConfig(epochs=1))
    assert torch.equal(torch.random.get_rng_state(), before)
    assert result["protocol"]["selected_architecture"] == smoke["protocol"]["selected_architecture"]
    assert result["preprocessing"] == smoke["preprocessing"]
    changed_test = False
    for architecture in wine.ARCHITECTURES:
        actual = result["models"][architecture]["runs"]
        reference = smoke["models"][architecture]["runs"]
        for run, original in zip(actual, reference, strict=True):
            assert run["train"] == original["train"]
            assert run["validation"] == original["validation"]
            assert run["history"] == original["history"]
            changed_test |= run["test"] != original["test"]
    assert changed_test


def test_experiment_restores_callers_thread_count_on_success_and_failure(monkeypatch):
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)

        def completed(config):
            assert torch.get_num_threads() == 1
            return {"done": True}

        monkeypatch.setattr(wine, "_run_wine_experiment", completed)
        assert wine.run_wine_experiment() == {"done": True}
        assert torch.get_num_threads() == 2

        def failed(config):
            assert torch.get_num_threads() == 1
            raise ValueError("deliberate failure")

        monkeypatch.setattr(wine, "_run_wine_experiment", failed)
        with pytest.raises(ValueError, match="deliberate failure"):
            wine.run_wine_experiment()
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(original_threads)


def test_evaluation_disables_autograd_and_restores_mode_even_after_error(data, monkeypatch):
    x, y, split = data
    model = wine.WineClassifier("linear", 42)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    original_forward = model.forward
    observed = []

    def check_forward(features):
        observed.append((model.training, torch.is_grad_enabled()))
        return original_forward(features)

    monkeypatch.setattr(model, "forward", check_forward)
    for was_training in (True, False):
        model.train(was_training)
        wine.evaluate_classifier(model, x[split["test"]], y[split["test"]], split["test"])
        assert model.training is was_training
        with pytest.raises(ValueError):
            wine.evaluate_classifier(model, x[split["test"]], y[split["test"]], [])
        assert model.training is was_training
    assert observed == [(False, False)] * 4
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
