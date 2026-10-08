import pytest
import torch

from pytorch_lab.regression import (
    TrainingConfig,
    affine_mse,
    analytical_gradients,
    finite_difference_gradients,
    run_training,
)


def make_problem(n=5, d=3, k=2):
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(n, d, generator=generator, dtype=torch.float64)
    target = torch.randn(n, k, generator=generator, dtype=torch.float64)
    weight = torch.randn(d, k, generator=generator, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(k, generator=generator, dtype=torch.float64, requires_grad=True)
    return x, target, weight, bias


@pytest.mark.parametrize("shape", [(1, 1, 1), (5, 3, 1), (5, 3, 2), (7, 2, 4)])
def test_gradients_match_three_independent_routes(shape):
    x, target, weight, bias = make_problem(*shape)
    automatic = torch.autograd.grad(affine_mse(x, target, weight, bias), (weight, bias))
    manual = analytical_gradients(x, target, weight, bias)
    numerical = finite_difference_gradients(x, target, weight, bias)
    for actual, expected, finite in zip(manual, automatic, numerical, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(finite, expected, rtol=1e-6, atol=1e-7)
    assert weight.grad is None  # autograd.grad returns gradients without accumulating into .grad
    assert bias.grad is None


def test_finite_difference_preserves_inputs_even_when_noncontiguous():
    x, target, weight, bias = make_problem()
    noncontiguous = weight.detach().T.contiguous().T.requires_grad_()
    assert not noncontiguous.is_contiguous()
    before = noncontiguous.detach().clone()
    numerical = finite_difference_gradients(x, target, noncontiguous, bias)
    analytical = analytical_gradients(x, target, noncontiguous, bias)
    for actual, expected in zip(numerical, analytical, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(noncontiguous, before, rtol=0, atol=0)


def test_torch_gradcheck():
    x, target, weight, bias = make_problem()
    assert torch.autograd.gradcheck(lambda w, b: affine_mse(x, target, w, b), (weight, bias))


def test_forbid_silent_target_broadcasting():
    x, _, weight, bias = make_problem(k=1)
    with pytest.raises(ValueError, match="expected"):
        affine_mse(x, torch.zeros(5, dtype=torch.float64), weight, bias)
    with pytest.raises(ValueError, match="broadcasting"):
        affine_mse(x, torch.zeros(1, 1, dtype=torch.float64), weight, bias)


@pytest.mark.parametrize("shape", [(0, 3, 2), (5, 0, 2), (5, 3, 0)])
def test_empty_dimensions_rejected(shape):
    with pytest.raises(ValueError, match="nonzero"):
        affine_mse(*make_problem(*shape))


def test_integer_dtype_rejected():
    x, target, weight, bias = make_problem()
    with pytest.raises(ValueError, match="floating"):
        affine_mse(x.to(torch.int64), target, weight, bias)


def test_mixed_dtype_rejected():
    x, target, weight, bias = make_problem()
    with pytest.raises(ValueError, match="same dtype"):
        affine_mse(x.float(), target, weight, bias)


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -float("inf")])
def test_nonfinite_inputs_rejected(bad):
    x, target, weight, bias = make_problem()
    x[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        affine_mse(x, target, weight, bias)


@pytest.mark.parametrize("epsilon", [0, -1e-6, float("nan"), float("inf")])
def test_bad_finite_difference_step_rejected(epsilon):
    with pytest.raises(ValueError):
        finite_difference_gradients(*make_problem(), epsilon=epsilon)


def test_float32_finite_difference_rejected():
    inputs = [tensor.float() for tensor in make_problem()]
    with pytest.raises(ValueError, match="float64"):
        finite_difference_gradients(*inputs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": -1},
        {"steps": True},
        {"steps": 1.5},
        {"learning_rate": 0},
        {"learning_rate": -1},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"noise_std": -1},
        {"noise_std": float("inf")},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**63},
    ],
)
def test_invalid_training_configuration(kwargs):
    with pytest.raises(ValueError):
        run_training(TrainingConfig(**kwargs))


def test_training_converges_and_beats_holdout_baseline():
    result = run_training()
    assert result["final_train_mse"] < result["initial_train_mse"] * 0.01
    assert result["test_mse"] < result["test_mean_baseline_mse"] * 0.01
    assert result["gradient_descent_lstsq_max_parameter_gap"] < 1e-10
    assert result["split_sizes"] == {"train": 96, "validation": 32, "test": 32}


def test_local_generator_does_not_modify_global_rng_and_replays():
    before = torch.random.get_rng_state().clone()
    first = run_training()
    after = torch.random.get_rng_state()
    second = run_training()
    assert torch.equal(before, after)
    # LAPACK lstsq may have last-bit differences; SGD values replay bit for bit here.
    for key in ("history", "learned_weight", "learned_bias", "test_mse"):
        assert first[key] == second[key]


def test_noiseless_fit_recovers_ground_truth():
    result = run_training(TrainingConfig(noise_std=0))
    assert result["test_mse"] < 1e-20
    torch.testing.assert_close(
        torch.tensor(result["learned_weight"]), torch.tensor(result["true_weight"])
    )


def test_changed_seed_changes_data_and_metrics():
    first = run_training(TrainingConfig(seed=42))
    second = run_training(TrainingConfig(seed=7))
    assert first["initial_train_mse"] != second["initial_train_mse"]


def test_finite_inputs_whose_loss_overflows_fail_explicitly():
    x, target, weight, bias = make_problem()
    x.fill_(1e200)
    with pytest.raises(ValueError, match="loss became non-finite"):
        affine_mse(x, target, weight, bias)


def test_overflowing_training_fails_explicitly():
    with pytest.raises(ValueError, match="non-finite"):
        run_training(TrainingConfig(steps=1, learning_rate=1e200))
