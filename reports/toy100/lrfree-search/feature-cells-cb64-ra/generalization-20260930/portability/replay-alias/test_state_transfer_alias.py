"""Checkpoint control tensors retain shared identity across allocating copies."""
from copy import deepcopy
import io

import pytest
import torch

from particlegan.continuous import SettleTest
from particlegan.policy import _state_to_device


class AllocatingTensor(torch.Tensor):
    """CPU-only transfer fixture: every .to() allocates, as a device move does."""

    @staticmethod
    def __new__(cls, value):
        return torch.Tensor._make_subclass(cls, value.detach().clone(), False)

    def to(self, *args, **kwargs):
        kwargs["copy"] = True
        return super().to(*args, **kwargs)

    def __deepcopy__(self, memo):
        key = id(self)
        if key not in memo:
            memo[key] = AllocatingTensor(self.as_subclass(torch.Tensor))
        return memo[key]


def allocating_state(value, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key not in memo: memo[key] = AllocatingTensor(value)
        return memo[key]
    if isinstance(value, dict): return {k: allocating_state(v, memo) for k, v in value.items()}
    if isinstance(value, list): return [allocating_state(v, memo) for v in value]
    if isinstance(value, tuple): return tuple(allocating_state(v, memo) for v in value)
    return value


def prior_transfer(value, device):
    """The original allocating traversal; its repeated leaves lose identity."""
    if isinstance(value, torch.Tensor): return value.to(device=device)
    if isinstance(value, dict): return {k: prior_transfer(v, device) for k, v in value.items()}
    if isinstance(value, list): return [prior_transfer(v, device) for v in value]
    if isinstance(value, tuple): return tuple(prior_transfer(v, device) for v in value)
    return value


def bits(value):
    return value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.device == b.device and a.shape == b.shape
        assert bits(a) == bits(b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b


def window():
    parameter = torch.nn.Parameter(torch.zeros(3, 4, device="cpu", dtype=torch.float32))
    tester = SettleTest()
    tester.rows = 3
    tester.begin([parameter])
    with torch.no_grad(): parameter.add_(1.)
    tester.observe([parameter], 1., step=1)
    state = tester.state_dict()
    assert state["last_block"] is state["blocks"][-1]
    return parameter, state


def test_allocating_copy_preserves_nested_tensor_alias_and_uint8():
    tensor = AllocatingTensor(torch.arange(16, device="cpu", dtype=torch.uint8))
    saved = {"stream": tensor, "nested": [tensor, (tensor,)]}
    moved = _state_to_device(saved, "cpu")
    assert moved["stream"] is not tensor
    assert moved["stream"].data_ptr() != tensor.data_ptr()
    assert moved["stream"] is moved["nested"][0] is moved["nested"][1][0]
    assert moved["stream"].dtype == torch.uint8 and moved["stream"].device.type == "cpu"
    assert bits(moved["stream"]) == bits(tensor)


def test_equal_values_with_distinct_owners_are_not_merged():
    a = AllocatingTensor(torch.ones(4, device="cpu"))
    b = AllocatingTensor(torch.ones(4, device="cpu"))
    moved = _state_to_device([a, b, a], "cpu")
    assert moved[0] is moved[2] and moved[0] is not moved[1]
    moved[0].add_(2.)
    assert torch.equal(moved[1], torch.ones(4, device="cpu"))


def test_row_rebase_updates_last_block_and_nan_row_energy_after_copy():
    parameter, saved = window()
    buffer = io.BytesIO(); torch.save(saved, buffer); buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)
    assert loaded["last_block"] is loaded["blocks"][-1]
    native, old_mapped, fixed = SettleTest(), SettleTest(), SettleTest()
    native.load_state_dict(loaded, parameter.numel())
    allocating = allocating_state(loaded)
    old_mapped.load_state_dict(prior_transfer(allocating, "cpu"), parameter.numel())
    fixed.load_state_dict(_state_to_device(allocating, "cpu"), parameter.numel())
    assert native.last_block is native.blocks[-1]
    assert old_mapped.last_block is not old_mapped.blocks[-1]
    assert fixed.last_block is fixed.blocks[-1]
    with torch.no_grad(): parameter[1].add_(4.)
    for tester in (native, old_mapped, fixed):
        tester.rebase([parameter], torch.tensor([1], device="cpu"))
    assert int(torch.isnan(native.last_block).sum()) == 4
    assert int(torch.isnan(old_mapped.last_block).sum()) == 0
    assert int(torch.isnan(fixed.last_block).sum()) == 4
    assert native.diagnostics()["top1pct_row_energy_share"] == .5
    assert fixed.diagnostics()["top1pct_row_energy_share"] == .5
    assert old_mapped.diagnostics()["top1pct_row_energy_share"] == pytest.approx(1. / 3.)
    equal(native.state_dict(), fixed.state_dict())


def test_real_meta_allocation_retains_partial_window_identity():
    _, saved = window()
    copied = _state_to_device(deepcopy(saved), "meta")
    assert copied["last_block"].device.type == "meta"
    assert copied["last_block"] is copied["blocks"][-1]
    assert copied["last_block"].dtype == saved["last_block"].dtype
    assert copied["last_block"].shape == saved["last_block"].shape
