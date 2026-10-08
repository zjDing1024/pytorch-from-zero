import pytest

from pytorch_lab.tensors import inspect_accumulation, inspect_broadcast_gradient, inspect_layouts


def test_transpose_is_a_strided_view_but_flattening_needs_a_copy():
    result = inspect_layouts()
    assert result["base_stride"] == [4, 1]
    assert result["transposed_stride"] == [1, 4]
    assert result["transpose_is_contiguous"] is False
    assert result["transpose_shares_storage"] is True
    assert result["mutation_visible_in_base"] is True
    assert result["view_flatten_rejected"] is True
    assert result["reshape_copy_in_this_case"] is True


@pytest.mark.parametrize("batch_size", [1, 4, 19])
def test_broadcast_backward_reduces_over_broadcast_axis(batch_size):
    result = inspect_broadcast_gradient(batch_size)
    assert result["bias_gradient"] == batch_size
    assert result["output_shape"] == [batch_size, 1]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_rejected(batch_size):
    with pytest.raises(ValueError):
        inspect_broadcast_gradient(batch_size)


def test_leaf_grad_accumulation_is_explicit():
    assert inspect_accumulation() == {"first": 6.0, "without_reset": 12.0, "after_reset": 6.0}
