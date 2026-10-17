"""Activation counters and tiny synthetic checks cannot substitute for quality."""
from copy import deepcopy

import pytest
import torch

from particlegan import Recipe
from experiments.forge.mechanisms import MechanismAudit, _ScalarCritic, mechanism_blockers


def fixture():
    recipe = Recipe()
    critic = _ScalarCritic()
    optimizer = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic))
    audit = MechanismAudit(recipe, optimizer, [])
    audit.observe_penalty({"applied": True, "phase": "a"})
    return audit


def test_component_probes_consume_no_rng_and_preserve_actual_host_counters():
    before = torch.get_rng_state().clone()
    audit = fixture().receipt()
    assert torch.equal(before, torch.get_rng_state())
    assert not mechanism_blockers(audit)
    for name, row in audit["mechanisms"].items():
        if name != "critic_penalty":
            assert row["calls"] == (1 if name == "critic_anchor" else 0)
            assert row["applied"] == 0
            assert row["probe"]["status"] == "PASS"
            assert row["probe"]["training_evidence"] is False


@pytest.mark.parametrize("bad_probe", [None, [], {"status": "PASS"}, {"status": "PASS", "measurements": []}])
def test_status_stamps_and_malformed_probes_do_not_prove_activation(bad_probe):
    audit = fixture().receipt()
    audit["mechanisms"]["critic_guard"]["probe"] = bad_probe
    assert any("critic_guard" in reason for reason in mechanism_blockers(audit))


def test_disabled_or_impossible_application_counts_are_rejected():
    audit = fixture().receipt()
    row = audit["mechanisms"]["critic_penalty"]
    row["requested"] = False
    assert mechanism_blockers(audit)
    row["requested"] = True
    row["applied"] = row["calls"] + 1
    assert any("inconsistent" in reason for reason in mechanism_blockers(audit))


def test_observers_preserve_public_optimizer_updates_exactly():
    recipe = Recipe(d_guard_min_steps=0)
    left, right = _ScalarCritic(), _ScalarCritic()
    left_d = recipe.make_critic_optimizer(left, ema_critic=deepcopy(left))
    right_d = recipe.make_critic_optimizer(right, ema_critic=deepcopy(right))
    left_z = torch.nn.Parameter(torch.zeros(4, 1, dtype=torch.float64))
    right_z = torch.nn.Parameter(torch.zeros(4, 1, dtype=torch.float64))
    left_g = recipe.make_generator_optimizer([left_z], latent_table=left_z)
    right_g = recipe.make_generator_optimizer([right_z], latent_table=right_z)
    audit = MechanismAudit(recipe, left_d, [left_g])
    for value in (1., -100.):
        for model, optimizer in ((left, left_d), (right, right_d)):
            model.weight.grad = torch.full_like(model.weight, value)
            optimizer.step()
        for table, optimizer in ((left_z, left_g), (right_z, right_g)):
            table.grad = torch.zeros_like(table)
            table.grad[0] = value
            optimizer.step()
    assert torch.equal(left.weight, right.weight)
    assert torch.equal(left_z, right_z)
    assert left_d.record.observed_steps == right_d.record.observed_steps == 2
    assert left_d.guard.clipped_tensors == right_d.guard.clipped_tensors > 0
    assert audit.rows["a2"]["applied"] == 1
    assert torch.equal(left_g.latent_history, right_g.latent_history)
