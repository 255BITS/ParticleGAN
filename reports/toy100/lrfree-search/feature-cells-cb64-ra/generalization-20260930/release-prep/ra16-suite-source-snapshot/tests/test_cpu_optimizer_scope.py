"""CPU optimizer portability without changing accelerator graph checks."""
from types import SimpleNamespace

import pytest
import torch

from particlegan.k3p import (
    K3PCriticAdam, K3PGeneratorAdam, _scope_reference_adam_graph_checks,
)


def _optimizer(kind, parameter, options):
    if kind == "generator":
        return K3PGeneratorAdam([parameter], **options)
    if kind == "critic":
        critic = torch.nn.Module()
        critic.register_parameter("weight", parameter)
        return K3PCriticAdam([parameter], critic=critic, guard_ratio=0, **options)
    optimizer = getattr(torch.optim, kind)([parameter], **options)
    _scope_reference_adam_graph_checks(optimizer)
    return optimizer


@pytest.mark.parametrize("kind", ["generator", "critic", "Adam", "AdamW"])
def test_cpu_updates_skip_accelerator_queries_and_match_adam(kind, monkeypatch):
    def forbidden_query(*args, **kwargs):
        raise AssertionError("CPU optimizer queried accelerator graph capture")

    for name in ("_accelerator_graph_capture_health_check", "_cuda_graph_capture_health_check"):
        if hasattr(torch.optim.Adam, name):
            monkeypatch.setattr(torch.optim.Adam, name, forbidden_query)
    parameter = torch.nn.Parameter(torch.tensor([.25, -.75], device="cpu"))
    reference = torch.nn.Parameter(parameter.detach().clone())
    options = dict(lr=.007, betas=(.2, .9), eps=1e-8, weight_decay=.12,
                   amsgrad=True, foreach=False)
    optimizer = _optimizer(kind, parameter, options)
    reference_type = torch.optim.AdamW if kind == "AdamW" else torch.optim.Adam
    baseline = reference_type([reference], **options)
    # The numerical reference bypasses only the known upstream CPU graph bug.
    baseline._accelerator_graph_capture_health_check = lambda: None
    baseline._cuda_graph_capture_health_check = lambda: None
    cuda_initialized = torch.cuda.is_initialized()
    for gradient in ([.2, -.1], [-.05, .3], [.125, -.25]):
        parameter.grad = torch.tensor(gradient, device="cpu")
        reference.grad = parameter.grad.clone()
        optimizer.step()
        baseline.step()
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        assert optimizer.state[parameter].keys() == baseline.state[reference].keys()
        for key, value in optimizer.state[parameter].items():
            torch.testing.assert_close(value, baseline.state[reference][key], rtol=0, atol=0)
    assert torch.cuda.is_initialized() == cuda_initialized


@pytest.mark.parametrize("kind", ["generator", "critic", "Adam", "AdamW"])
@pytest.mark.parametrize("name", ["_accelerator_graph_capture_health_check",
                                  "_cuda_graph_capture_health_check"])
def test_accelerator_parameter_keeps_original_graph_check(kind, name, monkeypatch):
    if not hasattr(torch.optim.Adam, name):
        pytest.skip("This PyTorch version uses the other graph-check name")
    parameter = torch.nn.Parameter(torch.tensor([.25], device="cpu"))
    optimizer = _optimizer(kind, parameter, dict(lr=.007))
    calls = []
    result = object()

    def upstream(owner):
        calls.append(owner)
        return result

    monkeypatch.setattr(torch.optim.Adam, name, upstream)
    # This marker exercises dispatch without constructing an accelerator tensor.
    optimizer.param_groups[0]["params"].append(SimpleNamespace(device=torch.device("cuda:0")))
    assert getattr(optimizer, name)() is result
    assert calls == [optimizer]


def test_caller_instance_graph_check_is_preserved():
    parameter = torch.nn.Parameter(torch.tensor([.25], device="cpu"))
    optimizer = torch.optim.Adam([parameter], lr=.007)
    names = [name for name in ("_accelerator_graph_capture_health_check",
                              "_cuda_graph_capture_health_check")
             if hasattr(torch.optim.Adam, name)]
    callback = lambda: None
    for name in names:
        setattr(optimizer, name, callback)
    _scope_reference_adam_graph_checks(optimizer)
    assert all(getattr(optimizer, name) is callback for name in names)
