"""Wrapper optimizers recover from a failed step (review findings #10, #22)."""
import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.optim.constraint_geometry import ConstraintGeometryOptimizer, constraint_geometry_backward
from particlegan.optim.direction_blend import DirectionBlendOptimizer
from particlegan.optim.dualnorm import NormalizedOptimizer, make_normalized_optimizer
from particlegan.optim.strict_progress import StrictProgressOptimizer


def _fail_base_once(monkeypatch):
    original = NormalizedOptimizer.step
    calls = {"n": 0}

    def flaky(self, closure=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected base failure")
        return original(self, closure)

    monkeypatch.setattr(NormalizedOptimizer, "step", flaky)


@pytest.mark.parametrize("cls", [ConstraintGeometryOptimizer, DirectionBlendOptimizer, StrictProgressOptimizer])
def test_failed_base_step_aborts_pending_losses(monkeypatch, cls):
    _fail_base_once(monkeypatch)
    parameter = nn.Parameter(torch.tensor([.1], dtype=torch.float64))
    optimizer = cls([parameter], lr=.1, smoothing=.001)

    def bind():
        optimizer.zero_grad()
        protected = parameter.square().sum()
        constraint_geometry_backward(-parameter.sum(), optimizer, (protected,),
                                     protected_evaluator=lambda: (parameter.square().sum(),))

    bind()
    with pytest.raises(RuntimeError, match="injected"):
        optimizer.step()
    assert optimizer._protected is None
    bind()  # previously: "constraint_geometry losses already pending"
    optimizer.step()
    assert optimizer.constraint_geometry_stats["steps"] == 1
    assert optimizer.state_dict()["constraint_geometry"]["pending"] is None


def test_strict_progress_missing_evaluator_does_not_strand_pending_losses():
    parameter = nn.Parameter(torch.tensor([.1], dtype=torch.float64))
    optimizer = StrictProgressOptimizer([parameter], lr=1., smoothing=.001)
    constraint_geometry_backward(-parameter.sum(), optimizer, (parameter.square().sum(),),
                                 protected_evaluator=lambda: (parameter.square().sum(),))
    optimizer._protected_evaluator = None  # e.g. restored pending state without rebinding
    with pytest.raises(ValueError, match="evaluator"):
        optimizer.step()
    optimizer.zero_grad()
    constraint_geometry_backward(-parameter.sum(), optimizer, (parameter.square().sum(),),
                                 protected_evaluator=lambda: (parameter.square().sum(),))
    optimizer.step()
    assert optimizer.state_dict()["strict_progress"]["pending_values"] is None


def test_abort_pending_is_public_and_clears_rows_and_losses():
    table = nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    optimizer = DirectionBlendOptimizer([dict(params=[table], role="prior")], lr=.1, smoothing=.001)
    optimizer.set_sampled_rows(table, torch.tensor([0, 2]))
    constraint_geometry_backward(table[[0, 2]].sum(), optimizer, (table[[0, 2]].sum(),))
    optimizer.abort_pending()
    assert optimizer._protected is None and optimizer.sampled_rows_for(table) is None


def _critic_cap():
    recipe = get_recipe('bcap').replace(optimizer_family='dualnorm', critic_step_mode='finite_cap',
        lr=.2, d_lr_mult=1., reg_every=1, input_noise_std=0., output_noise_std=0.)
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(.9)
    optimizer = recipe.make_critic_optimizer(critic)
    return critic, optimizer, recipe.make_critic_penalty(optimizer)


def test_critic_cap_failed_base_step_remains_checkpointable(monkeypatch):
    critic, optimizer, penalty = _critic_cap()

    def bind():
        optimizer.zero_grad()
        x = torch.tensor([[-1.], [1.]])
        (-critic.weight.sum() + penalty(critic, x, x / 2)).backward()

    _fail_base_once(monkeypatch)
    bind()
    with pytest.raises(RuntimeError, match="injected"):
        optimizer.step()
    optimizer.state_dict()  # previously: "checkpoint finite cap at completed-step boundaries"
    bind()
    optimizer.step()
    assert optimizer.critic_cap_stats["steps"] == 1
    assert "critic_cap" in optimizer.state_dict()


def test_frozen_parameter_in_group_gets_zero_protected_normal():
    live = nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    frozen = nn.Parameter(torch.tensor([3.], dtype=torch.float64), requires_grad=False)
    optimizer = ConstraintGeometryOptimizer([live, frozen], lr=.1, smoothing=.001)
    loss = (live.square().sum() + frozen.sum())
    constraint_geometry_backward(loss, optimizer, (loss,))
    assert torch.equal(optimizer._protected, torch.tensor([[2., 4., 0.]], dtype=torch.float64))
    optimizer.step()
    assert frozen.item() == 3.


@pytest.mark.parametrize("as_generator", [False, True])
def test_factory_accepts_plain_tensor_lists_and_generators(as_generator):
    recipe = get_recipe('bcap').replace(optimizer_family='dualnorm', prior_proximal_strength=.1)
    tensors = [nn.Parameter(torch.ones(2, 2)), nn.Parameter(torch.ones(2))]
    params = (t for t in tensors) if as_generator else tensors
    optimizer = make_normalized_optimizer(recipe, params)
    assert sum(len(g["params"]) for g in optimizer.param_groups) == 2


def test_prior_proximal_precheck_failure_aborts_direction_blend_pending_losses():
    from particlegan.optim.prior_proximal import PriorProximalDirectionBlendOptimizer
    table = nn.Parameter(torch.zeros(4, 2, dtype=torch.float64))
    optimizer = get_recipe("bcap", prior_proximal_strength=.5, constraint_geometry_mode="direction_blend",
                           lr=.03).make_generator_optimizer([dict(params=[table], role="prior")])
    assert isinstance(optimizer, PriorProximalDirectionBlendOptimizer)
    gradient = torch.arange(8., dtype=torch.float64).view(4, 2)

    def bind():
        optimizer.zero_grad()
        optimizer.set_sampled_rows(table, torch.tensor([0, 2]))
        loss = (table * gradient).sum()
        constraint_geometry_backward(loss, optimizer, [loss])

    bind()
    lr = optimizer.param_groups[0]["lr"]
    optimizer.param_groups[0]["lr"] = float("inf")  # nonfinite lr*strength: the proximal precheck rejects
    with pytest.raises(ValueError, match="must be finite"):
        optimizer.step()
    assert optimizer._protected is None and optimizer.sampled_rows_for(table) is None
    assert torch.equal(table, torch.zeros_like(table))
    optimizer.param_groups[0]["lr"] = lr
    bind()  # previously: "constraint_geometry losses already pending"
    optimizer.step()
    assert optimizer.prior_proximal_stats["steps"] == 1
