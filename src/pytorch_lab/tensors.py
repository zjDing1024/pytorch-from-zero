"""Observable storage, layout, and broadcasting behavior on concrete tensors."""

import torch


def inspect_layouts() -> dict:
    """Use mutations as well as pointers to distinguish a view from a copy."""
    base = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    transposed = base.transpose(0, 1)
    transposed[1, 2] = -7.0
    aliases = base[2, 1].item() == -7.0
    try:
        transposed.view(-1)
    except RuntimeError:
        flatten_view_rejected = True
    else:
        flatten_view_rejected = False
    reshaped = transposed.reshape(-1)
    base_before = base.clone()
    reshaped[0] = 999.0
    reshape_is_independent = torch.equal(base, base_before)
    return {
        "base_shape": list(base.shape),
        "base_stride": list(base.stride()),
        "transposed_shape": list(transposed.shape),
        "transposed_stride": list(transposed.stride()),
        "transpose_is_contiguous": transposed.is_contiguous(),
        "transpose_shares_storage": (
            base.untyped_storage().data_ptr() == transposed.untyped_storage().data_ptr()
        ),
        "mutation_visible_in_base": aliases,
        "view_flatten_rejected": flatten_view_rejected,
        "reshape_copy_in_this_case": reshape_is_independent,
    }


def inspect_broadcast_gradient(batch_size: int = 4) -> dict:
    """A broadcast scalar participates once per row, so its derivative is summed."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    x = torch.arange(batch_size, dtype=torch.float64).reshape(-1, 1)
    bias = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    output = x + bias
    output.sum().backward()
    return {
        "output_shape": list(output.shape),
        "bias_shape": list(bias.shape),
        "bias_gradient": bias.grad.item(),
        "expected_gradient": float(batch_size),
    }


def inspect_accumulation() -> dict:
    """Rebuild the graph each time; intentionally do not retain a freed graph."""
    weight = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    (weight.square()).backward()
    first = weight.grad.item()
    (weight.square()).backward()
    second = weight.grad.item()
    weight.grad = None
    (weight.square()).backward()
    return {"first": first, "without_reset": second, "after_reset": weight.grad.item()}
