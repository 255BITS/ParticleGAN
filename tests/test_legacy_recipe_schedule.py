"""The custom-host control must use the candidate's LR schedule fields."""

from copy import deepcopy

import pytest
import torch

from benchmarks.toy_suite import _check_actions
from benchmarks.smart_descent import evaluate
from benchmarks.toy100.schedule import policy_multipliers
from benchmarks.transfer_suite.compare_defaults import optimizer_defaults
from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
from benchmarks.transfer_suite.public_default_verification import declared_spec, load_declaration
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
from particlegan import get_recipe, learning_rate_scale


def test_legacy_control_uses_declared_anneal_start_and_floor():
    recipe = get_recipe().replace(lr_anneal_start=0.4, lr_floor=0.01)
    applied = []
    with optimizer_defaults(recipe, applied):
        control = evaluate.FixedControl({"schedule": "cosine"}, 100)
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(()))], lr=0.005)
        control.step(optimizer, 0, role="g")
        control.step(optimizer, 80, role="g")
    expected = learning_rate_scale(80, 100, 0.4, 0.01)
    assert applied[0]["lr"] == recipe.lr
    assert optimizer.param_groups[0]["lr"] == recipe.lr * expected
    assert control.trace == [dict(step=0, role="g", multiplier=1.0),
                             dict(step=80, role="g", multiplier=expected)]
    assert expected != learning_rate_scale(80, 100, 0.6, 0.05)


def _mixed_generator_update(cap, network_floor=None):
    torch.manual_seed(123)
    recipe = get_recipe().replace(num_particles=4, z_dim=2,
                                  lr_anneal_start=.6, lr_floor=.05)
    applied = []
    network = torch.nn.Parameter(torch.ones(()))
    with optimizer_defaults(recipe, applied, network_lr_horizon_cap=cap,
                            network_lr_floor=network_floor):
        prior = recipe.make_prior()
        optimizer = torch.optim.Adam([
            {"params": [network]}, {"params": list(prior.parameters())},
        ], lr=.005)
        control = evaluate.FixedControl({"schedule": "cosine"}, 100)
        for step in range(100):
            control.step(optimizer, step, role="g")
            optimizer.zero_grad()
            network.grad = torch.full_like(network, .1)
            prior.z.grad = torch.full_like(prior.z, .1)
            optimizer.step()
    return network.detach().clone(), prior.z.detach().clone(), applied, control.trace


def test_legacy_network_cap_at_or_above_budget_is_bitwise_old_path():
    old = _mixed_generator_update(None)
    capped = _mixed_generator_update(1600)
    assert torch.equal(old[0], capped[0])
    assert torch.equal(old[1], capped[1])
    assert old[2] == capped[2]
    assert [row["multiplier"] for row in old[3]] == [
        row["multiplier"] for row in capped[3] if row["step"] % 20 == 0
    ]


def test_legacy_network_floor_changes_only_generator_rate_on_short_budget():
    old = _mixed_generator_update(None)
    equal_floor = _mixed_generator_update(1600, .05)
    lower_floor = _mixed_generator_update(1600, .005)
    assert torch.equal(old[0], equal_floor[0])
    assert torch.equal(old[1], equal_floor[1])
    assert not torch.equal(old[0], lower_floor[0])
    assert torch.equal(old[1], lower_floor[1])
    assert "network_lr_floor" not in _mixed_generator_update(1600)[3][-1]
    action = lower_floor[3][-1]
    network, prior = policy_multipliers(
        99, 100, .6, .05, 1600, network_lr_floor=.005,
    )
    assert action["network_lr_floor"] == .005
    assert action["network_multiplier"] == network
    assert action["prior_multiplier"] == prior
    assert action["group_lrs"] == [
        dict(role="g", lr=get_recipe().lr * network),
        dict(role="prior", lr=get_recipe().lr * get_recipe().prior_lr_mult * prior),
    ]


def test_legacy_cap_changes_network_rate_but_preserves_prior_schedule():
    recipe = get_recipe().replace(num_particles=4, z_dim=2,
                                  lr_anneal_start=.6, lr_floor=.05)
    network = torch.nn.Parameter(torch.ones(()))
    applied = []
    with optimizer_defaults(recipe, applied, network_lr_horizon_cap=40):
        prior = recipe.make_prior()
        optimizer = torch.optim.Adam([
            {"params": [network]}, {"params": list(prior.parameters())},
        ], lr=.005)
        control = evaluate.FixedControl({"schedule": "cosine"}, 100)
        control.step(optimizer, 80, role="g")
    network_scale, prior_scale = policy_multipliers(80, 100, .6, .05, 40)
    assert network_scale == .05
    assert prior_scale > network_scale
    assert [group["lr"] for group in optimizer.param_groups] == [
        recipe.lr * network_scale,
        recipe.lr * recipe.prior_lr_mult * prior_scale,
    ]
    assert control.trace == [dict(
        step=80, role="g", multiplier=network_scale,
        network_lr_horizon_cap=40,
        network_multiplier=network_scale, prior_multiplier=prior_scale,
        group_lrs=[dict(role="g", lr=recipe.lr * network_scale),
                   dict(role="prior", lr=recipe.lr * recipe.prior_lr_mult * prior_scale)],
    )]


@pytest.mark.parametrize("network_floor", [None, .005])
def test_custom_host_cap_receipts_cover_every_update_and_reject_missing_middle(
    network_floor,
):
    declaration = {"network_lr_horizon_cap": 40}
    if network_floor is not None:
        declaration["network_lr_floor"] = network_floor
    recipe, noise, _ = declared_recipe(declaration)
    jobs, profile = load_declaration()
    job = next(row for row in jobs if row["spec"]["name"] == "two_pole")
    spec, _, _ = declared_spec(job, profile, recipe)
    policy = declared_model_policy(declaration)
    result, context = run_legacy(spec, recipe, noise, model_policy=policy)
    record = dict(spec=spec, result=result, applied=context["applied"])
    assert len(result["actions"]) == spec["steps"] * 2
    _check_actions(record, recipe, noise, policy)

    missing = deepcopy(record)
    missing["result"]["actions"].pop(3)
    with pytest.raises(ValueError, match="optimizer trace is incomplete"):
        _check_actions(missing, recipe, noise, policy)

    altered = deepcopy(record)
    altered["result"]["actions"][-1]["group_lrs"][0]["lr"] = .99
    with pytest.raises(ValueError, match="optimizer rate differs"):
        _check_actions(altered, recipe, noise, policy)
    if network_floor is not None:
        altered = deepcopy(record)
        altered["result"]["actions"][-1].pop("network_lr_floor")
        with pytest.raises(ValueError, match="network floor differs"):
            _check_actions(altered, recipe, noise, policy)
