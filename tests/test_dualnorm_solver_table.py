"""Behaviour-preserving solver plumbing for NormalizedOptimizer (review workstream I3).

The batched per-offset kernel branches are CUDA-only in production. These CPU
tests pin (a) the historical branch-selection order and (b) bitwise parity of
the shared ``_apply_conv_batch`` helper against the pre-refactor inline code,
with the CUDA-graph caches replaced by their eager equivalents.
"""
from types import SimpleNamespace
import inspect
import math

import pytest
import torch
from torch import nn

import particlegan.optim.convolution_quintic as quintic
import particlegan.optim.dualnorm as dualnorm
from particlegan.optim.constraint_geometry import ConstraintGeometryOptimizer, constraint_geometry_backward
from particlegan.optim.direction_blend import DirectionBlendOptimizer
from particlegan.optim.dualnorm import NormalizedOptimizer
from particlegan.optim.strict_progress import StrictProgressOptimizer

CUDA, CPU = SimpleNamespace(type="cuda"), SimpleNamespace(type="cpu")


def _legacy_solver(svd_backend, rule, device, largest):
    """Pre-refactor branch order of NormalizedOptimizer.step, transcribed."""
    cuda = device.type == "cuda"
    if svd_backend == "batched" and cuda and largest <= 32:
        return "batched_svd"
    if rule != "polar" and cuda and largest > 32:
        if rule == "frobenius":
            return "frobenius"
        return "quintic_graph" if svd_backend == "batched" else "quintic"
    if svd_backend == "newton_schulz" and cuda and largest > 32:
        return "newton_schulz"
    return None


@pytest.mark.parametrize("svd_backend,rule", [
    ("native", "polar"), ("cpu", "polar"), ("newton_schulz", "polar"), ("batched", "polar"),
    ("native", "frobenius"), ("batched", "frobenius"), ("native", "quintic"), ("batched", "quintic"),
])
def test_solver_selection_matches_legacy_branch_order(svd_backend, rule):
    optimizer = SimpleNamespace(svd_backend=svd_backend, convolution_rule=rule)
    for device in (CUDA, CPU):
        for largest in (1, 32, 33, 512):
            assert (NormalizedOptimizer._conv_batch_solver(optimizer, device, largest)
                    == _legacy_solver(svd_backend, rule, device, largest))


class _EagerPolarCache:
    def solve(self, oriented, smoothing):
        scaled, regularizer, result, lower, identity = dualnorm._polar_initial(oriented, smoothing)
        for iteration in range(4, 33, 4):
            result, lower = dualnorm._polar_block(result, lower, identity)
            if iteration >= 8:
                failed = dualnorm._polar_failed(result, lower, scaled, regularizer, identity)
                if not bool(failed.any()):
                    break
        return result, failed


class _EagerQuinticCache:
    def direction(self, matrices, *, smoothing):
        return quintic._quintic_direction(matrices, smoothing=smoothing)


def _legacy_apply(solver, optimizer, parameter, gradient, direction, metadata, rate, factor, eps):
    """The three pre-refactor inline batched branches, verbatim modulo dispatch."""
    gradient_batch = dualnorm._convolution_batch(gradient, metadata)
    direction_batch = dualnorm._convolution_batch(direction, metadata)
    smoothing = optimizer.smoothing
    if solver == "batched_svd":
        update = dualnorm._batched_polar(direction_batch, smoothing=smoothing)
    elif solver == "frobenius":
        update = dualnorm.frobenius_direction(direction_batch, smoothing=smoothing)
    elif solver == "quintic":
        update = quintic.quintic_direction(direction_batch, smoothing=smoothing)
    elif solver == "quintic_graph":
        update = _EagerQuinticCache().direction(direction_batch, smoothing=smoothing)
    else:
        update = dualnorm._newton_schulz_polar(direction_batch, smoothing=smoothing, cache=_EagerPolarCache())
    active = ((gradient_batch.norm(dim=(-2, -1)) >= eps)
              & (direction_batch.norm(dim=(-2, -1)) >= eps))
    update = torch.where(active[..., None, None], update, 0.)
    if metadata["layout"] == "conv_transpose2d_in_out":
        update = update.permute(0, 4, 3, 1, 2)
    else:
        update = update.permute(0, 3, 4, 1, 2)
    parameter.add_(update.reshape_as(parameter), alpha=-rate * factor)


@pytest.mark.parametrize("solver", sorted(dualnorm._CONV_BATCH_SOLVERS))
@pytest.mark.parametrize("module", [
    lambda: nn.Conv2d(4, 6, 3), lambda: nn.ConvTranspose2d(6, 4, 2),
    lambda: nn.Conv2d(4, 6, 3, groups=2), lambda: nn.ConvTranspose2d(4, 6, 3, groups=2),
], ids=["conv", "conv_transposed", "groups", "groups_transposed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_apply_conv_batch_is_bitwise_legacy(monkeypatch, solver, module, dtype):
    monkeypatch.setattr(dualnorm, "_PolarGraphCache", _EagerPolarCache)
    monkeypatch.setattr(quintic, "QuinticGraphCache", _EagerQuinticCache)
    torch.manual_seed(0)
    layer = module().to(dtype)
    group, = dualnorm.convolution_parameter_groups([dict(params=[layer.weight])], [layer], family="dualnorm")
    metadata = group["dualnorm_convolution"]
    optimizer = NormalizedOptimizer([group], lr=.05, smoothing=.001, convolution="per_offset")
    gradient = torch.randn_like(layer.weight)
    gradient[0].zero_()  # exercise the inactive-slice skip
    direction = gradient * 2
    factor = math.sqrt(metadata["out_channels"] / metadata["in_channels"]) / math.prod(metadata["kernel_size"])
    expected = layer.weight.detach().clone()
    actual = layer.weight.detach().clone()
    _legacy_apply(solver, optimizer, expected, gradient, direction, metadata, .05, factor, 1e-8)
    optimizer._apply_conv_batch(solver, actual, gradient, direction, metadata, rate=.05, factor=factor, eps=1e-8)
    assert not torch.equal(actual, layer.weight.detach())
    assert torch.equal(actual, expected)
    assert "_quintic_graphs" not in optimizer.state_dict()


class _CountingBase(NormalizedOptimizer):
    """A cooperative mixin between ConstraintGeometryOptimizer and the base."""

    def step(self, closure=None):
        self.mixin_calls = getattr(self, "mixin_calls", 0) + 1
        return super().step(closure)


@pytest.mark.parametrize("wrapper", [ConstraintGeometryOptimizer, DirectionBlendOptimizer, StrictProgressOptimizer])
def test_wrapper_base_update_follows_the_mro(wrapper):
    # Previously DirectionBlend/StrictProgress called NormalizedOptimizer.step(self)
    # directly, silently skipping any cooperative class later in the MRO.
    # torch.optim.Optimizer.__init__ installs a profiling wrapper as ``cls.step``
    # on first construction, so compare the unwrapped functions.
    assert inspect.unwrap(wrapper.step) is inspect.unwrap(ConstraintGeometryOptimizer.step)
    combined = type("Combined", (wrapper, _CountingBase), {})
    parameter = nn.Parameter(torch.tensor([.1, -.2], dtype=torch.float64))
    optimizer = combined([parameter], lr=.1, smoothing=.001)
    constraint_geometry_backward((parameter * torch.tensor([1., 2.], dtype=torch.float64)).sum(), optimizer,
                                 (parameter.square().sum(),),
                                 protected_evaluator=lambda: (parameter.square().sum(),))
    optimizer.step()
    assert optimizer.mixin_calls == 1
    assert optimizer.constraint_geometry_stats["steps"] == 1
