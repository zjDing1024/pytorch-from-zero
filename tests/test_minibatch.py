"""Contracts for the Module, Dataset/DataLoader, and SGD learning increment."""

import json
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lab.minibatch import (
    AffineRegressor,
    MinibatchConfig,
    RegressionDataset,
    evaluate,
    fit_minibatches,
    make_loader,
    make_splits,
    run_minibatch_experiment,
    train_one_epoch,
)
from pytorch_lab.regression import analytical_gradients


def make_dataset(n=5, d=3, k=2, dtype=torch.float64):
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(n, d, generator=generator, dtype=dtype)
    target = torch.randn(n, k, generator=generator, dtype=dtype)
    return RegressionDataset(x, target)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_module_registers_exact_zero_initialized_parameters(dtype):
    model = AffineRegressor(3, 2, dtype=dtype)
    assert isinstance(model, nn.Module)
    assert set(dict(model.named_parameters())) == {"weight", "bias"}
    assert set(model.state_dict()) == {"weight", "bias"}
    assert model.weight.shape == (3, 2)
    assert model.bias.shape == (2,)
    for parameter in model.parameters():
        assert isinstance(parameter, nn.Parameter)
        assert parameter.requires_grad and parameter.is_leaf
        assert parameter.dtype == dtype
        assert parameter.device.type == "cpu"
        assert torch.count_nonzero(parameter).item() == 0
    assert model(torch.ones(4, 3, dtype=dtype)).shape == (4, 2)


@pytest.mark.parametrize("shape", [(1, 1, 1), (5, 3, 1), (5, 3, 2), (7, 2, 4)])
def test_module_backward_matches_existing_multioutput_analytical_gradients(shape):
    dataset = make_dataset(*shape)
    model = AffineRegressor(shape[1], shape[2])
    with torch.no_grad():
        model.weight.copy_(torch.arange(model.weight.numel()).reshape_as(model.weight) / 7)
        model.bias.fill_(0.3)
    loss = (model(dataset.features) - dataset.targets).square().mean()
    loss.backward()
    expected = analytical_gradients(dataset.features, dataset.targets, model.weight, model.bias)
    for parameter, gradient in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "dimensions", [(0, 1), (1, 0), (-1, 1), (True, 1), (1, False), (1.5, 1), (1, 2.5)]
)
def test_module_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError):
        AffineRegressor(*dimensions)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16])
def test_module_rejects_unsupported_dtypes(dtype):
    with pytest.raises(ValueError):
        AffineRegressor(3, 2, dtype=dtype)


@pytest.mark.parametrize(
    "features",
    [
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(2, 3, 1, dtype=torch.float64),
        torch.zeros(0, 3, dtype=torch.float64),
        torch.zeros(2, 2, dtype=torch.float64),
        torch.zeros(2, 3, dtype=torch.float32),
        torch.zeros(2, 3, dtype=torch.int64),
        torch.full((2, 3), float("nan"), dtype=torch.float64),
        torch.full((2, 3), float("inf"), dtype=torch.float64),
    ],
)
def test_module_rejects_invalid_features(features):
    with pytest.raises(ValueError):
        AffineRegressor(3, 2)(features)


def test_dataset_snapshots_and_detaches_inputs():
    x = torch.arange(15, dtype=torch.float64).reshape(5, 3).requires_grad_()
    target = torch.arange(10, dtype=torch.float64).reshape(5, 2).requires_grad_()
    expected_x, expected_target = x.detach().clone(), target.detach().clone()
    dataset = RegressionDataset(x, target)
    assert len(dataset) == 5
    assert dataset.features.data_ptr() != x.data_ptr()
    assert dataset.targets.data_ptr() != target.data_ptr()
    assert not dataset.features.requires_grad and dataset.features.grad_fn is None
    assert not dataset.targets.requires_grad and dataset.targets.grad_fn is None
    with torch.no_grad():
        x.fill_(100)
        target.fill_(-100)
    torch.testing.assert_close(dataset.features, expected_x)
    torch.testing.assert_close(dataset.targets, expected_target)
    for index in range(len(dataset)):
        sample_x, sample_target = dataset[index]
        torch.testing.assert_close(sample_x, expected_x[index])
        torch.testing.assert_close(sample_target, expected_target[index])


@pytest.mark.parametrize(
    "x_shape,target_shape",
    [
        ((0, 3), (0, 2)),
        ((5, 0), (5, 2)),
        ((5, 3), (5, 0)),
        ((5,), (5, 2)),
        ((5, 3), (5,)),
        ((5, 3), (1, 2)),
        ((5, 3, 1), (5, 2)),
    ],
)
def test_dataset_rejects_empty_or_inconsistent_shapes(x_shape, target_shape):
    with pytest.raises(ValueError):
        RegressionDataset(
            torch.zeros(x_shape, dtype=torch.float64),
            torch.zeros(target_shape, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    "x_dtype,target_dtype",
    [
        (torch.int64, torch.int64),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float64),
    ],
)
def test_dataset_rejects_unsupported_or_inconsistent_dtypes(x_dtype, target_dtype):
    with pytest.raises(ValueError):
        RegressionDataset(torch.zeros(5, 3, dtype=x_dtype), torch.zeros(5, 2, dtype=target_dtype))


@pytest.mark.parametrize("field", ["features", "targets"])
@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -float("inf")])
def test_dataset_rejects_nonfinite_inputs(field, bad):
    x, target = torch.zeros(5, 3, dtype=torch.float64), torch.zeros(5, 2, dtype=torch.float64)
    (x if field == "features" else target)[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        RegressionDataset(x, target)


def collect_order(loader):
    return [int(value) for features, _ in loader for value in features[:, 0]]


def test_loader_replays_shuffle_but_changes_epoch_order_and_keeps_last_batch():
    dataset = RegressionDataset(
        torch.arange(23, dtype=torch.float64).reshape(-1, 1),
        torch.zeros(23, 1, dtype=torch.float64),
    )
    first = make_loader(dataset, batch_size=5, seed=17, shuffle=True)
    recreated = make_loader(dataset, batch_size=5, seed=17, shuffle=True)
    assert isinstance(first, DataLoader)
    assert first.num_workers == 0 and first.drop_last is False
    assert isinstance(first.generator, torch.Generator)
    assert first.generator.device.type == "cpu"
    epochs = [collect_order(first), collect_order(first)]
    assert epochs == [collect_order(recreated), collect_order(recreated)]
    assert epochs[0] != epochs[1]
    assert all(sorted(order) == list(range(23)) for order in epochs)
    assert [len(x) for x, _ in first] == [5, 5, 5, 5, 3]


def test_unshuffled_loader_keeps_order_and_has_its_own_generator():
    dataset = RegressionDataset(
        torch.arange(7, dtype=torch.float64).reshape(-1, 1),
        torch.zeros(7, 1, dtype=torch.float64),
    )
    loader = make_loader(dataset, batch_size=3)
    assert isinstance(loader.generator, torch.Generator)
    before = torch.random.get_rng_state().clone()
    assert collect_order(loader) == list(range(7))
    assert collect_order(loader) == list(range(7))
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": True},
        {"batch_size": 1.5},
        {"batch_size": 2, "seed": -1},
        {"batch_size": 2, "seed": True},
        {"batch_size": 2, "seed": 2**63},
    ],
)
def test_loader_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make_loader(make_dataset(), **kwargs)


@pytest.mark.parametrize("was_training", [True, False])
def test_evaluate_weights_uneven_batches_and_restores_mode_without_gradients(was_training):
    target = torch.tensor([[1, 2], [3, 4], [9, 10]], dtype=torch.float64)
    dataset = RegressionDataset(torch.zeros(3, 2, dtype=torch.float64), target)
    model = AffineRegressor(2, 2)
    model.train(was_training)
    observed = []
    handle = model.register_forward_pre_hook(
        lambda module, inputs: observed.append((module.training, torch.is_grad_enabled()))
    )
    try:
        actual = evaluate(model, make_loader(dataset, batch_size=2))
    finally:
        handle.remove()
    assert isinstance(actual, float)
    assert actual == pytest.approx(target.square().mean().item(), rel=1e-14)
    assert observed == [(False, False), (False, False)]
    assert model.training is was_training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_training_one_full_batch_matches_handwritten_gradient_step():
    dataset = make_dataset(n=7, k=3)
    model = AffineRegressor(3, 3)
    learning_rate = 0.07
    gradients = analytical_gradients(dataset.features, dataset.targets, model.weight, model.bias)
    expected = [
        parameter.detach() - learning_rate * gradient
        for parameter, gradient in zip(model.parameters(), gradients, strict=True)
    ]
    metrics = train_one_epoch(
        model,
        make_loader(dataset, batch_size=100),
        torch.optim.SGD(model.parameters(), lr=learning_rate),
    )
    assert metrics["samples_seen"] == 7 and metrics["batches"] == 1
    assert metrics["mse"] == pytest.approx(dataset.targets.square().mean().item())
    for parameter, reference in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter, reference, rtol=1e-14, atol=1e-14)
        assert parameter.grad is None


def test_training_resets_gradients_and_reports_sample_weighted_online_loss():
    dataset = make_dataset(n=5, k=2)
    model = AffineRegressor(3, 2)
    model.eval()
    learning_rate = 0.03
    expected_weight, expected_bias = model.weight.detach().clone(), model.bias.detach().clone()
    weighted_loss = 0.0
    for x, target in make_loader(dataset, batch_size=2):
        weighted_loss += (
            x @ expected_weight + expected_bias - target
        ).square().mean().item() * len(x)
        dw, db = analytical_gradients(x, target, expected_weight, expected_bias)
        expected_weight -= learning_rate * dw
        expected_bias -= learning_rate * db
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 1000)
    observed = []
    handle = model.register_forward_pre_hook(
        lambda module, inputs: observed.append((module.training, torch.is_grad_enabled()))
    )
    try:
        metrics = train_one_epoch(
            model,
            make_loader(dataset, batch_size=2),
            torch.optim.SGD(model.parameters(), lr=learning_rate),
        )
    finally:
        handle.remove()
    assert observed == [(True, True)] * 3
    assert model.training is True
    assert metrics["samples_seen"] == 5 and metrics["batches"] == 3
    assert metrics["mse"] == pytest.approx(weighted_loss / 5, rel=1e-14)
    torch.testing.assert_close(model.weight, expected_weight, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(model.bias, expected_bias, rtol=1e-14, atol=1e-14)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_training_rejects_nonfinite_loss():
    dataset = RegressionDataset(
        torch.ones(2, 1, dtype=torch.float64),
        torch.full((2, 1), 1e200, dtype=torch.float64),
    )
    model = AffineRegressor(1, 1)
    with pytest.raises(ValueError, match="non-finite"):
        train_one_epoch(model, make_loader(dataset, 2), torch.optim.SGD(model.parameters(), lr=0.1))


def test_training_rejects_nonfinite_gradients_before_update():
    dataset = make_dataset()
    model = AffineRegressor(3, 2)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    handle = model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    try:
        with pytest.raises(ValueError, match="non-finite"):
            train_one_epoch(
                model, make_loader(dataset, 5), torch.optim.SGD(model.parameters(), lr=0.1)
            )
    finally:
        handle.remove()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_training_rejects_nonfinite_parameter_update():
    dataset = RegressionDataset(
        torch.ones(2, 1, dtype=torch.float64),
        torch.full((2, 1), 1e100, dtype=torch.float64),
    )
    model = AffineRegressor(1, 1)
    with pytest.raises(ValueError, match="non-finite"):
        train_one_epoch(
            model, make_loader(dataset, 2), torch.optim.SGD(model.parameters(), lr=1e300)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"epochs": -1},
        {"epochs": True},
        {"epochs": 1.5},
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": True},
        {"batch_size": 1.5},
        {"learning_rate": 0},
        {"learning_rate": -1},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"noise_std": -1},
        {"noise_std": float("nan")},
        {"noise_std": float("inf")},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**63},
    ],
)
def test_fit_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        fit_minibatches(make_dataset(), MinibatchConfig(**kwargs))


def test_splits_match_original_seeded_synthetic_problem_exactly():
    seed, noise_std = 7, 0.05
    splits = make_splits(seed=seed, noise_std=noise_std)
    assert {name: len(dataset) for name, dataset in splits.items()} == {
        "train": 96,
        "validation": 32,
        "test": 32,
    }
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(160, 3, generator=generator, dtype=torch.float64)
    noise = torch.randn(160, 1, generator=generator, dtype=torch.float64)
    targets = features @ torch.tensor([[1.75], [-2.0], [0.5]], dtype=torch.float64) - 0.3
    targets += noise_std * noise
    for name, start, stop in [("train", 0, 96), ("validation", 96, 128), ("test", 128, 160)]:
        torch.testing.assert_close(splits[name].features, features[start:stop], rtol=0, atol=0)
        torch.testing.assert_close(splits[name].targets, targets[start:stop], rtol=0, atol=0)


def test_fitting_only_depends_on_training_dataset_and_converges():
    splits = make_splits()
    config = MinibatchConfig()
    model, history = fit_minibatches(splits["train"], config)
    for name in ("validation", "test"):
        splits[name].features.fill_(1e6)
        splits[name].targets.fill_(-1e6)
    repeated_model, repeated_history = fit_minibatches(splits["train"], config)
    assert history == repeated_history
    for actual, repeated in zip(model.parameters(), repeated_model.parameters(), strict=True):
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    assert len(history) == config.epochs + 1
    assert history[0]["epoch"] == 0 and history[-1]["epoch"] == config.epochs
    assert history[-1]["train_mse"] < history[0]["train_mse"] * 0.01
    expected_final = evaluate(model, make_loader(splits["train"], batch_size=17))
    assert history[-1]["train_mse"] == pytest.approx(expected_final, rel=1e-14)
    assert all(math.isfinite(row["online_train_mse"]) for row in history[1:])


def test_fitting_history_distinguishes_online_from_fixed_model_loss():
    dataset = make_dataset(n=5, k=2)
    config = MinibatchConfig(epochs=1, batch_size=2, learning_rate=0.03)
    reference = AffineRegressor(3, 2)
    online = train_one_epoch(
        reference,
        make_loader(dataset, batch_size=2, seed=config.seed, shuffle=True),
        torch.optim.SGD(reference.parameters(), lr=config.learning_rate),
    )
    final = evaluate(reference, make_loader(dataset, batch_size=2))
    model, history = fit_minibatches(dataset, config)
    assert history[1]["online_train_mse"] == pytest.approx(online["mse"], rel=1e-14)
    assert history[1]["train_mse"] == pytest.approx(final, rel=1e-14)
    assert history[1]["online_train_mse"] != pytest.approx(history[1]["train_mse"])
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_complete_experiment_preserves_global_rng_and_replays_training():
    before = torch.random.get_rng_state().clone()
    first = run_minibatch_experiment()
    assert torch.equal(torch.random.get_rng_state(), before)
    second = run_minibatch_experiment()
    assert torch.equal(torch.random.get_rng_state(), before)
    json.dumps(first, allow_nan=False)
    assert first["training"] == second["training"]


@pytest.mark.parametrize("parameter_name", ["weight", "bias"])
def test_module_rejects_nonfinite_parameters(parameter_name):
    model = AffineRegressor(3, 2)
    with torch.no_grad():
        getattr(model, parameter_name).fill_(float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        model(torch.ones(5, 3, dtype=torch.float64))


def test_module_rejects_prediction_overflow_from_finite_inputs():
    model = AffineRegressor(1, 1)
    with torch.no_grad():
        model.weight.fill_(1e200)
    with pytest.raises(ValueError, match="non-finite"):
        model(torch.full((2, 1), 1e200, dtype=torch.float64))


def test_cpu_contract_rejects_non_cpu_dataset_and_features():
    features = torch.empty(5, 3, dtype=torch.float64, device="meta")
    with pytest.raises(ValueError, match="CPU"):
        RegressionDataset(features, torch.zeros(5, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="CPU"):
        AffineRegressor(3, 2)(features)


@pytest.mark.parametrize(
    "bad_target",
    [
        torch.zeros(1, 1, dtype=torch.float64),
        torch.zeros(2, 1, dtype=torch.float32),
        torch.full((2, 1), float("nan"), dtype=torch.float64),
    ],
)
def test_evaluate_rejects_malformed_batch_and_restores_model_mode(bad_target):
    model = AffineRegressor(3, 1)
    model.train()
    # A generic loader can bypass RegressionDataset validation; loss must revalidate.
    batches = [(torch.zeros(2, 3, dtype=torch.float64), bad_target)]
    with pytest.raises(ValueError):
        evaluate(model, batches)
    assert model.training is True


def test_empty_evaluation_loader_rejected_and_mode_restored():
    model = AffineRegressor(3, 2)
    model.train()
    with pytest.raises(ValueError, match="no samples"):
        evaluate(model, [])
    assert model.training is True


def test_fit_supports_float32_multioutput_data():
    dataset = make_dataset(n=7, k=3, dtype=torch.float32)
    model, history = fit_minibatches(dataset, MinibatchConfig(epochs=2, batch_size=3))
    assert model.weight.dtype == model.bias.dtype == torch.float32
    assert model(dataset.features).shape == (7, 3)
    assert [row["samples_seen"] for row in history[1:]] == [7, 7]
    assert [row["batches"] for row in history[1:]] == [3, 3]
    assert history[-1]["train_mse"] < history[0]["train_mse"]


def test_experiment_reports_actual_work_counts_and_comparable_full_batch_baseline():
    result = run_minibatch_experiment(MinibatchConfig(epochs=10, batch_size=13))
    training, comparison = result["training"], result["comparison"]
    assert result["all_checks_passed"] is True
    assert all(result["checks"].values())
    assert training["optimizer_steps"] == 80
    assert training["training_sample_visits"] == 960
    assert training["last_batch_size"] == 5
    assert comparison["full_batch_optimizer_steps"] == 10
    assert comparison["full_batch_training_sample_visits"] == 960
    assert comparison["handwritten_full_batch"]["config"] == {
        "seed": 42,
        "steps": 10,
        "learning_rate": 0.05,
        "noise_std": 0.05,
    }
    assert comparison["module_full_batch_vs_handwritten_max_parameter_gap"] < 1e-12
    assert comparison["module_full_batch_final_train_mse"] == pytest.approx(
        comparison["handwritten_full_batch"]["final_train_mse"],
        rel=1e-12,
    )
    assert training["final_train_mse"] < training["initial_train_mse"] * 0.01
    assert training["test_mse"] < training["test_mean_baseline_mse"] * 0.01
