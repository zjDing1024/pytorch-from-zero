"""Hand calculations, per-step torch equivalence, and strict input contracts."""

import json

import pytest
import torch

from pytorch_lab.momentum import momentum_sgd_step, run_momentum_reference_trace


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def step(parameter, gradient, buffer=None, **options):
    options = {"learning_rate": 0.1, "momentum": 0.8, **options}
    parameters, buffers = momentum_sgd_step([parameter], [gradient], [buffer], **options)
    return parameters[0], buffers[0]


def test_first_buffer_is_undampened_and_uses_first_present_gradient():
    parameter, buffer = step(tensor([2.0]), None, dampening=0.5)
    assert buffer is None
    parameter, buffer = step(parameter, tensor([3.0]), buffer, dampening=0.5)
    torch.testing.assert_close(buffer, tensor([3.0]))
    torch.testing.assert_close(parameter, tensor([1.7]))
    parameter, buffer = step(parameter, tensor([4.0]), buffer, dampening=0.5)
    torch.testing.assert_close(buffer, tensor([4.4]))  # 0.8 * 3 + 0.5 * 4
    torch.testing.assert_close(parameter, tensor([1.26]))


def test_none_gradient_and_explicit_zero_have_different_semantics():
    parameter, buffer = tensor([2.0]), tensor([3.0])
    skipped, frozen = step(parameter, None, buffer, weight_decay=0.5)
    moved, changed = step(parameter, tensor([0.0]), buffer, weight_decay=0.5)
    assert torch.equal(skipped, parameter) and torch.equal(frozen, buffer)
    torch.testing.assert_close(changed, tensor([3.4]))  # momentum + coupled decay
    torch.testing.assert_close(moved, tensor([1.66]))
    _, new = step(parameter, tensor([0.0]), weight_decay=0.0)
    assert new is not None and torch.equal(new, tensor([0.0]))


def test_learning_rate_change_does_not_rescale_buffer_and_zero_lr_still_updates_it():
    parameter, buffer = step(tensor([2.0]), tensor([3.0]))
    parameter, buffer = step(parameter, tensor([1.0]), buffer, learning_rate=0.0)
    torch.testing.assert_close(parameter, tensor([1.7]))
    torch.testing.assert_close(buffer, tensor([3.4]))
    parameter, buffer = step(parameter, tensor([0.0]), buffer, learning_rate=0.01)
    torch.testing.assert_close(buffer, tensor([2.72]))
    torch.testing.assert_close(parameter, tensor([1.6728]))


def test_maximize_negates_loss_gradient_but_not_weight_decay():
    parameter, buffer = step(tensor([2.0]), tensor([3.0]), maximize=True, weight_decay=0.5)
    torch.testing.assert_close(buffer, tensor([-2.0]))  # -3 + 0.5 * 2
    torch.testing.assert_close(parameter, tensor([2.2]))


def test_nesterov_uses_current_gradient_plus_new_momentum_buffer():
    parameter, buffer = step(tensor([2.0]), tensor([3.0]), nesterov=True)
    torch.testing.assert_close(buffer, tensor([3.0]))
    torch.testing.assert_close(parameter, tensor([1.46]))  # 2 - 0.1 * (3 + 0.8 * 3)
    parameter, buffer = step(parameter, tensor([1.0]), buffer, nesterov=True)
    torch.testing.assert_close(buffer, tensor([3.4]))
    torch.testing.assert_close(parameter, tensor([1.088]))


def test_zero_momentum_ignores_but_preserves_existing_buffer():
    parameter, buffer = step(tensor([2.0]), tensor([3.0]), tensor([9.0]), momentum=0.0)
    torch.testing.assert_close(parameter, tensor([1.7]))
    assert torch.equal(buffer, tensor([9.0]))
    _, no_buffer = step(tensor([2.0]), tensor([3.0]), momentum=0.0)
    assert no_buffer is None
    _, resumed = step(parameter, tensor([1.0]), buffer)
    torch.testing.assert_close(resumed, tensor([8.2]))


@pytest.mark.parametrize(
    "options",
    [
        {"momentum": 0.0},
        {"momentum": 0.9},
        {"momentum": 0.9, "dampening": 0.4, "weight_decay": 0.2},
        {"momentum": 0.9, "dampening": 1.0},
        {"momentum": 0.9, "nesterov": True, "weight_decay": 0.2},
        {"momentum": 0.9, "nesterov": True, "maximize": True, "weight_decay": 0.2},
    ],
)
def test_multistep_independent_quadratic_gradients_match_torch_parameters_and_buffers(options):
    manual = [tensor([[1.0, -2.0], [0.3, 0.4]]).t(), tensor(0.5)]
    reference = [value.clone().requires_grad_() for value in manual]
    buffers = [None, None]
    oracle = torch.optim.SGD(reference, lr=0.1, foreach=False, fused=False, **options)
    for iteration in range(12):
        rate = (0.1, 0.04, 0.0, 0.01)[iteration // 3]
        # Each path obtains its own derivative from its own current parameters.
        gradients = [2.0 * value for value in manual]
        reference_gradients = list(
            torch.autograd.grad(sum(value.square().sum() for value in reference), reference)
        )
        if iteration in (0, 5):
            gradients[1] = reference_gradients[1] = None
        manual, buffers = momentum_sgd_step(
            manual, gradients, buffers, learning_rate=rate, **options
        )
        oracle.param_groups[0]["lr"] = rate
        for parameter, gradient in zip(reference, reference_gradients, strict=True):
            parameter.grad = gradient
        oracle.step()
        for actual, expected, buffer in zip(manual, reference, buffers, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
            expected_buffer = oracle.state.get(expected, {}).get("momentum_buffer")
            assert (buffer is None) == (expected_buffer is None)
            if buffer is not None:
                torch.testing.assert_close(buffer, expected_buffer, rtol=0, atol=1e-12)


@pytest.mark.parametrize("gradient_present", [False, True])
def test_inputs_are_unchanged_and_outputs_are_independent_detached_snapshots(gradient_present):
    parameter = tensor([[1.0, 2.0], [3.0, 4.0]]).t().requires_grad_()
    gradient = parameter.square() if gradient_present else None
    buffer = tensor([[0.1, 0.2], [0.3, 0.4]]).t().requires_grad_()
    saved = [
        value.detach().clone() if value is not None else None
        for value in (parameter, gradient, buffer)
    ]
    new_parameter, new_buffer = step(parameter, gradient, buffer, weight_decay=0.2)
    for original, expected in zip((parameter, gradient, buffer), saved, strict=True):
        if original is not None:
            assert torch.equal(original, expected)
    assert not new_parameter.requires_grad and new_parameter.grad_fn is None
    assert not new_buffer.requires_grad and new_buffer.grad_fn is None
    new_parameter.fill_(100)
    new_buffer.fill_(100)
    assert torch.equal(parameter, saved[0]) and torch.equal(buffer, saved[2])


@pytest.mark.parametrize("name", ["learning_rate", "momentum", "dampening", "weight_decay"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "0.1", tensor(0.1)])
def test_rejects_invalid_hyperparameters(name, value):
    with pytest.raises(ValueError, match=name):
        step(tensor([1.0]), tensor([2.0]), **{name: value})


@pytest.mark.parametrize(
    "options",
    [
        {"dampening": 1.1},
        {"nesterov": 1},
        {"maximize": 0},
        {"nesterov": True, "momentum": 0},
        {"nesterov": True, "dampening": 0.1},
    ],
)
def test_rejects_invalid_option_combinations(options):
    with pytest.raises(ValueError):
        step(tensor([1.0]), tensor([2.0]), **options)


@pytest.mark.parametrize("slot", ["parameters", "gradients", "buffers"])
@pytest.mark.parametrize(
    "invalid",
    [
        [1.0],
        tensor([]),
        torch.ones(1, dtype=torch.float32),
        torch.ones(1, dtype=torch.int64),
        torch.ones(1, dtype=torch.complex128),
        torch.ones(1, dtype=torch.float64, device="meta"),
        tensor([float("nan")]),
        tensor([float("inf")]),
        tensor([1.0]).to_sparse(),
    ],
)
def test_rejects_unsupported_tensors_in_every_slot(slot, invalid):
    values = {"parameters": [tensor([1.0])], "gradients": [tensor([2.0])], "buffers": [None]}
    values[slot] = [invalid]
    with pytest.raises(ValueError):
        momentum_sgd_step(**values, learning_rate=0.1, momentum=0.8)


@pytest.mark.parametrize("slot", ["gradients", "buffers"])
def test_shape_mismatch_is_rejected_even_when_broadcasting_would_work(slot):
    values = {"parameters": [tensor([1.0, 2.0])], "gradients": [None], "buffers": [None]}
    values[slot] = [tensor([1.0])]
    with pytest.raises(ValueError, match="shape"):
        momentum_sgd_step(**values, learning_rate=0.1, momentum=0.8)


@pytest.mark.parametrize(
    "parameters, gradients, buffers",
    [
        ([], [], []),
        ([tensor([1.0])], [], [None]),
        ([tensor([1.0])], [None], []),
        (tensor([1.0]), [None], [None]),
    ],
)
def test_rejects_empty_or_mismatched_or_nonsequence_collections(parameters, gradients, buffers):
    with pytest.raises(ValueError):
        momentum_sgd_step(parameters, gradients, buffers, learning_rate=0.1, momentum=0.8)


def test_shared_parameter_storage_is_outside_the_contract():
    base = tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="distinct storage"):
        momentum_sgd_step(
            [base[:1], base[1:]], [None, None], [None, None], learning_rate=0.1, momentum=0.8
        )


@pytest.mark.parametrize(
    "parameter, gradient, buffer, options",
    [
        (tensor([1e308]), tensor([1e308]), None, {"weight_decay": 2.0}),
        (tensor([1.0]), tensor([1.0]), tensor([1e308]), {"momentum": 2.0}),
        (tensor([1.0]), tensor([1e308]), None, {"learning_rate": 2.0}),
        (tensor([1.0]), tensor([1e308]), None, {"momentum": 1.0, "nesterov": True}),
    ],
)
def test_arithmetic_overflow_fails_without_changing_inputs(parameter, gradient, buffer, options):
    before_parameter, before_gradient = parameter.clone(), gradient.clone()
    before_buffer = None if buffer is None else buffer.clone()
    with pytest.raises(ValueError, match="finite"):
        step(parameter, gradient, buffer, **options)
    assert torch.equal(parameter, before_parameter) and torch.equal(gradient, before_gradient)
    if buffer is not None:
        assert torch.equal(buffer, before_buffer)


def test_reference_trace_records_every_step_parameter_and_buffer_and_preserves_rng():
    rng = torch.random.get_rng_state().clone()
    result = run_momentum_reference_trace()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["all_checks_passed"] and all(result["checks"].values())
    assert result["max_parameter_abs_difference"] <= 1e-12
    assert result["max_buffer_abs_difference"] <= 1e-12
    assert len(result["cases"]) == 6
    for case in result["cases"]:
        assert len(case["steps"]) == 6
        assert [row["learning_rate"] for row in case["steps"]] == [0.2, 0.2, 0.05, 0, 0.025, 0.025]
        for row in case["steps"]:
            assert [value["name"] for value in row["parameters"]] == ["weight", "bias"]
            for value in row["parameters"]:
                assert value["parameter_max_abs_difference"] <= 1e-12
                assert value["buffer_presence_matches"]
                gap = value["buffer_max_abs_difference"]
                assert gap is None or gap <= 1e-12
    assert result == run_momentum_reference_trace()
