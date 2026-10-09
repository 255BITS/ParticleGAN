"""Bounded API software checks; task quality still needs its full Forge budget."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from benchmarks.toy_audit.api_images import WordFixture, word_bank, word_oracle_controls
from experiments.forge.api import FormulationContext
from experiments.forge.behavior_adapters import HOSTS, run_behavior
from experiments.forge.state import state_digest
from experiments.forge.views import load_tasks
from particlegan import get_recipe
from particlegan.conditional_transport import OutputMarginalTransport


ROOT = Path(__file__).resolve().parents[1]
WINNER = "bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36"


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def winner():
    return json.loads((ROOT / "configs/forge/configurations" / (WINNER + ".json")).read_text())


def integrated_task(host, *, declared=True):
    task = deepcopy(load_tasks(ROOT)[host])
    task["execution"]["steps"] = 2
    task["task_cohort"] = "component_output_transport_v1"
    if declared:
        task["execution"]["transport_consumer"] = "output_marginal_v1"
    else:
        task["execution"].pop("transport_consumer", None)
    return task


def inactive_candidate(family):
    if family == "bcap":
        return winner()
    if family == "r1r2":
        return json.loads((ROOT / "configs/forge/ideas/k3p-r1r2-matched-v1.json").read_text())
    if family == "released":
        return json.loads((ROOT / "configs/forge/ideas/release07-gan-v3-task-adapted-v1.json").read_text())
    return {"recipe_preset": family}


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("family", ["bcap", "k3p", "ka2", "r1r2", "released"])
def test_disabled_consumer_needs_no_hooks_and_preserves_actual_state(tmp_path, host, family):
    declaration = dict(protocol=dict(seed=0), candidate=inactive_candidate(family))
    before = torch.get_rng_state().clone()
    plain = run_behavior(declaration, integrated_task(host, declared=False), tmp_path / "plain")
    optional = run_behavior(declaration, integrated_task(host), tmp_path / "optional")
    assert torch.equal(before, torch.get_rng_state())
    assert plain["evidence"]["observations"] == optional["evidence"]["observations"]
    assert plain["applied"]["component_transport"] is optional["applied"]["component_transport"] is None
    assert plain["evidence"]["guards"] == optional["evidence"]["guards"]
    for name in ("models", "role_parameters", "optimizers", "streams"):
        original = torch.load(tmp_path / "plain/component-state.pt", weights_only=False)[name]
        compatible = torch.load(tmp_path / "optional/component-state.pt", weights_only=False)[name]
        assert state_digest(original) == state_digest(compatible), name


@pytest.mark.parametrize("host", HOSTS)
def test_every_component_host_consumes_active_recipe_and_advances_once(tmp_path, host):
    candidate = winner()
    candidate["recipe_overrides"].update(constraint_geometry_mode="direction_blend",
        kinetic_transport_weight=1., kinetic_transport_local_weight=1.)
    result = run_behavior(dict(protocol=dict(seed=0), candidate=candidate), integrated_task(host), tmp_path)
    guards = result["evidence"]["guards"]
    assert guards["all_finite"] and guards["hooks_exercised"]
    assert guards["unintended_rng_deviations"] == 0
    assert all(count == 2 for count in guards["optimizer_updates"].values())
    assert guards["component_transport_requested"] and guards["component_transport_active_calls"] == 2
    assert result["applied"]["component_transport"]["active_calls"] == 2
    saved = torch.load(tmp_path / "component-state.pt", weights_only=False)
    joint = saved["optimizers"]["generator"]
    assert len(joint) == 1
    assert joint[0]["constraint_geometry"]["mode"] == "direction_blend"
    assert joint[0]["constraint_geometry"]["stats"]["steps"] == 2


def test_unused_slot_is_excluded_from_transport_training(tmp_path, monkeypatch):
    panels = []
    original = OutputMarginalTransport.add
    def observe(self, total, fake, real, **kwargs):
        panels.append((fake.detach().clone(), real.detach().clone()))
        return original(self, total, fake, real, **kwargs)
    monkeypatch.setattr(OutputMarginalTransport, "add", observe)
    candidate = winner()
    candidate["recipe_overrides"].update(kinetic_transport_weight=1., kinetic_transport_local_weight=1.)
    run_behavior(dict(protocol=dict(seed=0), candidate=candidate), integrated_task("unused_token_hold"), tmp_path)
    assert len(panels) == 2
    for fake, real in panels:
        assert fake.shape == real.shape == (8, 2)
        assert torch.equal(real, torch.tensor([[0., 1.]]).expand(8, -1))
        assert torch.equal(fake, fake[:1].expand_as(fake))


def test_disabled_consumer_returns_the_exact_loss_without_validating_optional_panels():
    consumer = OutputMarginalTransport(get_recipe("ka2", kinetic_transport_weight=0., kinetic_transport_local_weight=0.))
    parameter = torch.tensor(2., requires_grad=True)
    loss = parameter.square()
    before = torch.get_rng_state().clone()
    assert consumer.add(loss, None, None, conditioning=None) is loss
    loss.backward()
    assert parameter.grad == 4.
    assert torch.equal(before, torch.get_rng_state())
    assert consumer.active_calls == 0


def test_transport_detaches_targets_and_excludes_conditioning_from_distance():
    recipe = get_recipe("bcap", kinetic_transport_weight=1., kinetic_transport_local_weight=1.)
    consumer = OutputMarginalTransport(recipe)
    real = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
    fake = (real.detach() + .3).requires_grad_()
    conditioning = torch.tensor([[0.], [1.], [2.], [3.]], requires_grad=True)
    before = torch.get_rng_state().clone()
    consumer.add(fake.sum() * 0, fake, real, conditioning=conditioning).backward()
    assert fake.grad is not None and bool((fake.grad != 0).any())
    assert real.grad is None and conditioning.grad is None
    assert torch.equal(before, torch.get_rng_state())


def test_perfect_word_marginal_cannot_replace_paired_reconstruction_gates():
    controls = word_oracle_controls()
    assert all(control["passed"] == control["expected_pass"] for control in controls.values())
    assert not controls["swapped_reconstruction"]["passed"]
    target = word_bank(device="cpu").flatten(1)
    consumer = OutputMarginalTransport(get_recipe("bcap", kinetic_transport_weight=1., kinetic_transport_local_weight=1.))
    swapped = target[[1, 0, 2, 3, 4]].clone().requires_grad_()
    assert consumer.add(swapped.sum() * 0, swapped, target).item() == pytest.approx(0., abs=1e-12)


def word_fixture(*, active):
    overrides = dict(winner()["recipe_overrides"], num_particles=5, z_dim=2, batch_size=256, total_steps=20000)
    if active:
        overrides.update(constraint_geometry_mode="direction_blend", kinetic_transport_weight=1., kinetic_transport_local_weight=1.)
    context = FormulationContext(recipe_preset="bcap", recipe_overrides=overrides,
        prior=dict(kind="particle_cloud", sigma=0., standardize=False, learnable=True,
                   exception_reason="software check of the original finite vocabulary"),
        seed=0, device="cpu", initializer="deterministic_orthogonal", execution_path="public_components",
        component_transport="output_marginal_v1" if active else None)
    return WordFixture(device="cpu", seed=0, recipe_name=None, max_steps=3, components=context), context


@pytest.mark.parametrize("active", [False, True])
def test_word_consumer_resume_preserves_actual_next_update_and_all_named_streams(active):
    fixture, context = word_fixture(active=active)
    fixture.step()
    saved = deepcopy(fixture.state_dict())
    streams = deepcopy(context.streams.state_dict())
    original_next = fixture.step()
    resumed, resumed_context = word_fixture(active=active)
    resumed_context.streams.load_state_dict(streams)
    resumed.policy.load_state_dict(saved["api_state"])
    resumed.data_generator.set_state(saved["data_generator"])
    resumed.restore_component_transport(saved.get("component_transport"))
    resumed_next = resumed.step()
    assert state_digest(original_next) == state_digest(resumed_next)
    assert state_digest(fixture.state_dict()) == state_digest(resumed.state_dict())
    assert state_digest(context.streams.state_dict()) == state_digest(resumed_context.streams.state_dict())
    if active:
        assert fixture.transport.active_calls == 2
        assert fixture.opt_g.constraint_geometry_stats["steps"] == 2
    else:
        assert "component_transport" not in saved
        assert fixture.transport is None


def test_transport_checkpoint_rejects_nonfinite_counters():
    consumer = OutputMarginalTransport(get_recipe("bcap", kinetic_transport_weight=1.))
    saved = consumer.state_dict()
    saved["loss_sum"] = float("nan")
    with pytest.raises(ValueError, match="checkpoint"):
        consumer.load_state_dict(saved)
