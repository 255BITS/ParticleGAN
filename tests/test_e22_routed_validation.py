"""Complete-forward validation avoids accelerator readbacks at every site."""
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from particlegan import RoutedRows


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]


class Router(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.register_buffer("log_mass", torch.zeros(3, device=device, dtype=torch.float64))


def setup(device, forward):
    table = nn.Parameter(torch.tensor([[.2, -.1], [.4, .3], [-.2, .5]],
                                     device=device, dtype=torch.float64))
    context = torch.linspace(-.3, .4, 12, device=device, dtype=torch.float64).reshape(2, 3, 2)
    models = {"router": Router(device)}
    rows = RoutedRows(model_forward=forward, features=lambda models, context, samples, targets: samples.flatten(1),
                      sites=tuple(f"site{i}" for i in range(71)))
    return rows, models, table, context, rows.candidate_for(models, table)


def assert_closed(execution):
    assert not execution._active
    assert execution._candidate is None and execution._usage is None
    assert execution._perturb_fn is None and execution._validation_checks is None
    with pytest.raises(ValueError, match="only during"):
        execution.finish()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_71_site_mix_has_no_scalar_readbacks_and_finish_has_exactly_one(monkeypatch):
    reads, traces = [], []
    old_bool, old_int = torch.Tensor.__bool__, torch.Tensor.__int__

    def checked_bool(value):
        if value.device.type == "cuda":
            raise AssertionError("mix must not read accelerator predicates")
        return old_bool(value)

    def checked_int(value):
        if value.device.type == "cuda":
            assert traces and traces[-1]._next == 71
            reads.append(tuple(value.shape))
        return old_int(value)

    def forward(models, context, candidate, routing):
        traces.append(routing)
        hidden = context
        for index in range(71):
            hidden = routing.mix(f"site{index}", hidden @ candidate.table.T).tanh()
            assert len(routing._validation_checks) == 2 * (index + 1)
            assert routing._validation_checks[-1].grad_fn is None
            assert not routing._validation_checks[-1].requires_grad
            assert reads == []
        return hidden

    rows, models, table, context, candidate = setup("cuda:0", forward)
    monkeypatch.setattr(torch.Tensor, "__bool__", checked_bool)
    monkeypatch.setattr(torch.Tensor, "__int__", checked_int)
    output, usage = rows.forward_with_usage(models, context, candidate)
    assert reads == [()]
    assert_closed(traces[0])
    gradient = torch.autograd.grad(output.sum() + usage.square().sum(), table)[0]
    assert gradient.shape == table.shape
    # Tensor predicates are used only after restoring the guard against reads.
    monkeypatch.setattr(torch.Tensor, "__bool__", old_bool)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fault", ["nan", "inf", "all_inactive", "nan_mass", "overflow_cast"])
def test_invalid_values_fail_before_complete_forward_returns_and_cleanup(device, fault):
    traces, visited = [], []

    def forward(models, context, candidate, routing):
        traces.append(routing)
        output = None
        for index in range(71):
            visited.append(index)
            logits = context.new_zeros((2, 3, 3))
            if fault == "nan":
                logits[..., 0] = torch.nan
            elif fault == "inf":
                logits[..., 0] = torch.inf
            elif fault == "overflow_cast":
                logits = logits.double() + 1e100
            output = routing.mix(f"site{index}", logits)
        return output

    rows, models, table, context, candidate = setup(device, forward)
    if fault in ("all_inactive", "nan_mass"):
        models["router"].log_mass.fill_(-torch.inf if fault == "all_inactive" else torch.nan)
    if fault == "overflow_cast":
        table = nn.Parameter(table.float())
        models["router"].float()
        candidate = rows.candidate_for(models, table)
    message = "logits must be finite" if fault in ("nan", "inf") else "softmax must retain finite mass"
    with pytest.raises(ValueError, match=message):
        rows.forward(models, context, candidate)
    # CPU failures stay eager. Accelerators batch the predicates at finish.
    assert visited == ([0] if device == "cpu" else list(range(71)))
    assert_closed(traces[0])


@pytest.mark.parametrize("device", DEVICES)
def test_callback_exception_clears_pending_validation_and_activation_references(device):
    traces = []

    def failed(models, context, candidate, routing):
        traces.append(routing)
        routing.mix("site0", context @ candidate.table.T)
        raise RuntimeError("caller model failed")

    rows, models, _, context, candidate = setup(device, failed)
    with pytest.raises(RuntimeError, match="caller model failed"):
        rows.forward(models, context, candidate)
    assert_closed(traces[0])


@pytest.mark.parametrize("device", DEVICES)
def test_shape_validation_stays_immediate_and_closes_even_on_accelerator(device):
    traces, visited = [], []

    def malformed(models, context, candidate, routing):
        traces.append(routing)
        for index in range(71):
            visited.append(index)
            logits = context.new_zeros((1, 3, 3)) if index == 1 else context @ candidate.table.T
            routing.mix(f"site{index}", logits)
        return context

    rows, models, _, context, candidate = setup(device, malformed)
    with pytest.raises(ValueError, match="logits must be finite"):
        rows.forward(models, context, candidate)
    assert visited == [0, 1]
    assert_closed(traces[0])


@pytest.mark.parametrize("device", DEVICES)
def test_activation_recompute_keeps_exact_gradients_and_closes_each_execution(device):
    traces = []

    def forward(models, context, candidate, routing):
        traces.append(routing)
        hidden = context
        for index in range(71):
            hidden = routing.mix(f"site{index}", hidden @ candidate.table.T).tanh() + .001 * context
        return hidden

    rows, models, table, context, candidate = setup(device, forward)
    ordinary = rows.forward(models, context, candidate)
    expected_gradient = torch.autograd.grad(ordinary.square().sum(), table)[0]
    traces.clear()
    with set_checkpoint_early_stop(False):
        recomputed = checkpoint(lambda inputs: rows.forward(models, inputs, candidate), context, use_reentrant=False)
    actual_gradient = torch.autograd.grad(recomputed.square().sum(), table)[0]
    torch.testing.assert_close(recomputed, ordinary, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert len(traces) == 2
    for execution in traces:
        assert_closed(execution)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fault", [None, "nan", "negative", "normalization"])
def test_legacy_route_preserves_value_and_normalization_errors_with_one_readback(device, fault, monkeypatch):
    generator_called, reads = [], []

    def route(models, context, candidate):
        weights = (context[:, 0] @ candidate.table.T + candidate.log_mass).softmax(1)
        if fault == "nan":
            return weights + torch.nan
        if fault == "negative":
            return torch.cat((-weights[:, :1], weights[:, 1:]), dim=1)
        return weights * 2 if fault == "normalization" else weights

    def generate(models, context, candidate, weights):
        generator_called.append(True)
        return candidate.codes

    _, models, table, context, _ = setup(device, lambda *args: None)
    rows = RoutedRows(route=route, generate=generate, features=lambda *args: None)
    candidate = rows.candidate_for(models, table)
    old_int = torch.Tensor.__int__

    def checked_int(value):
        if value.device.type == "cuda":
            reads.append(tuple(value.shape))
        return old_int(value)

    monkeypatch.setattr(torch.Tensor, "__int__", checked_int)
    if fault is None:
        actual = rows.forward(models, context, candidate)
        manual = route(models, context, candidate) @ table
        torch.testing.assert_close(actual, manual, rtol=0, atol=0)
        assert generator_called == [True]
    else:
        message = "sum to one" if fault == "normalization" else "finite nonnegative"
        with pytest.raises(ValueError, match=message):
            rows.forward(models, context, candidate)
        assert generator_called == []
    assert reads == ([()] if device != "cpu" else [])
