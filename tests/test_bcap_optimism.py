"""Public Recipe algebra/resume software controls, not trained qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.optim.dualnorm import polar_factor
from particlegan.optim.optimism import PREVIOUS, SEEN
from tests.test_bcap_svd_backend import equal

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_public_raw_math_changes_polar_orientation_and_resumes(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    recipe = get_recipe("bcap", optimizer_optimism="raw_gradient", lr=.03)
    model = nn.Linear(2, 2, bias=False, device=device, dtype=torch.float64)
    optimizer = recipe.make_generator_optimizer(model)
    g0 = torch.tensor([[1., 2.], [-3., 1.]], device=device, dtype=torch.float64)
    g1 = torch.tensor([[2., -1.], [1., 3.]], device=device, dtype=torch.float64)
    model.weight.grad = g0
    before = model.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    optimizer.step()
    torch.testing.assert_close(model.weight, before - .03 * polar_factor(g0, smoothing=.001))
    assert torch.equal(rng, torch.get_rng_state())
    assert model.weight.grad is g0
    saved = deepcopy(optimizer.state_dict())
    weights = deepcopy(model.state_dict())
    before = model.weight.detach().clone()
    model.weight.grad = g1
    optimizer.step()
    expected = before - .03 * polar_factor(2 * g1 - g0, smoothing=.001)
    torch.testing.assert_close(model.weight, expected)
    assert not torch.allclose(expected, before - .03 * polar_factor(g1, smoothing=.001))
    restored = nn.Linear(2, 2, bias=False, device=device, dtype=torch.float64)
    restored.load_state_dict(weights)
    resumed = recipe.make_generator_optimizer(restored)
    resumed.load_state_dict(saved)
    restored.weight.grad = g1.clone()
    resumed.step()
    equal(model.state_dict(), restored.state_dict())
    equal(optimizer.state_dict(), resumed.state_dict())
    assert optimizer.optimism_stats["extrapolated_applications"] == 1


def test_actual_sampled_rows_have_independent_history_and_no_unsampled_motion():
    table = nn.Parameter(torch.zeros(4, 2, dtype=torch.float64))
    recipe = get_recipe("bcap", optimizer_optimism="raw_gradient", lr=.1)
    opt = recipe.make_generator_optimizer([table], latent_table=table)
    table.grad = torch.ones_like(table)
    opt.set_sampled_rows(table, torch.tensor([0, 0, 2]))
    opt.step()
    assert torch.equal(table[1], torch.zeros(2, dtype=table.dtype))
    assert torch.equal(opt.state[table][SEEN], torch.tensor([True, False, True, False]))
    before = table.detach().clone()
    saved = deepcopy(opt.state_dict())
    table.grad = torch.full_like(table, 50.)  # unsampled dense gradients never enter history
    table.grad[0] = torch.tensor([0., 1.])
    table.grad[1] = torch.tensor([1., 0.])
    opt.set_sampled_rows(table, torch.tensor([0, 1]))
    opt.step()
    forecast = torch.tensor([[-1., 1.], [1., 0.]], dtype=table.dtype)
    expected = before[:2] - .1 * forecast / torch.hypot(forecast.norm(dim=1, keepdim=True),
                                                      torch.full((2, 1), .001, dtype=table.dtype))
    torch.testing.assert_close(table[:2], expected)
    equal(table[2:], before[2:])
    equal(opt.state[table][PREVIOUS][2], torch.ones(2, dtype=table.dtype))
    restored_table = nn.Parameter(before.clone())
    restored = recipe.make_generator_optimizer([restored_table], latent_table=restored_table)
    restored.load_state_dict(saved)
    restored_table.grad = table.grad.clone()
    restored.set_sampled_rows(restored_table, torch.tensor([0, 1]))
    restored.step()
    equal(table, restored_table)
    equal(opt.state_dict(), restored.state_dict())


def test_disabled_defaults_keep_old_packets_and_other_techniques():
    for name in ("bcap", "bcap_adam", "gan", "k3p", "r1r2"):
        try:
            recipe = get_recipe(name)
        except ValueError:
            continue
        assert "optimizer_optimism" not in recipe.to_dict()
        model = nn.Linear(2, 2)
        opt = recipe.make_generator_optimizer(model)
        model(torch.ones(2, 2)).sum().backward()
        opt.step()
        packet = deepcopy(opt.state_dict())
        opt.load_state_dict(packet)
        equal(packet, opt.state_dict())
        assert all(PREVIOUS not in state for state in packet["state"].values())
        assert "optimism" not in packet.get("dualnorm", {})


@pytest.mark.parametrize("delta", [dict(optimizer_family="adam", optimizer_smoothing=0., optimizer_convolution="none"),
    dict(optimizer_momentum=.5), dict(constraint_geometry_mode="direction_blend"), dict(critic_step_mode="finite_cap")])
def test_unsupported_active_compositions_fail_explicitly(delta):
    with pytest.raises(ValueError, match="optimism"):
        get_recipe("bcap", optimizer_optimism="raw_gradient", **delta)


def test_invalid_forecast_or_checkpoint_fails_before_mutation():
    recipe = get_recipe("bcap", optimizer_optimism="raw_gradient")
    model = nn.Linear(2, 2)
    opt = recipe.make_generator_optimizer(model)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    saved, weights = deepcopy(opt.state_dict()), deepcopy(model.state_dict())
    bad = deepcopy(saved)
    bad["state"][0][PREVIOUS][0, 0] = float("nan")
    with pytest.raises(ValueError, match="history"):
        opt.load_state_dict(bad)
    equal(saved, opt.state_dict())
    for p in model.parameters():
        p.grad = torch.full_like(p, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        opt.step()
    equal(weights, model.state_dict())
    equal(saved, opt.state_dict())
    inactive = get_recipe("bcap").make_generator_optimizer(model)
    with pytest.raises(ValueError, match="checkpoint"):
        inactive.load_state_dict(saved)
    with pytest.raises(ValueError, match="checkpoint"):
        opt.load_state_dict(inactive.state_dict())


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_public_word_all_roles_named_streams_and_exact_resume(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from benchmarks.toy_audit.api_images import WordFixture
    from experiments.forge.api import task_formulation_context
    candidate = json.loads((ROOT / "configs/forge/ideas/bcap-three-phase-incumbent-v1.json").read_text())
    candidate["recipe_overrides"]["optimizer_optimism"] = "raw_gradient"
    task = json.loads((ROOT / "configs/forge/tasks/five_word_joint_smoke.json").read_text())
    def build():
        ctx = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
        return ctx, WordFixture(device=device, seed=0, recipe_name=None, max_steps=20001, components=ctx)
    ctx, fixture = build()
    for _ in range(3):
        fixture.step()
    saved, streams = deepcopy(fixture.state_dict()), deepcopy(ctx.streams.state_dict())
    suffix = [fixture.step() for _ in range(2)]
    observation = fixture.observe()
    restored_ctx, restored = build()
    restored.policy.load_state_dict(saved["api_state"])
    restored.data_generator.set_state(saved["data_generator"])
    restored.restore_component_transport(saved.get("component_transport"))
    restored_ctx.streams.load_state_dict(streams)
    equal(suffix, [restored.step() for _ in range(2)])
    equal(observation, restored.observe())
    equal(fixture.state_dict(), restored.state_dict())
    equal(ctx.streams.state_dict(), restored_ctx.streams.state_dict())
    assert {g["role"] for g in fixture.opt_g.param_groups} >= {"generator", "prior"}
    # This original host intentionally owns G and E in one generator group.
    assert all(PREVIOUS in fixture.opt_g.state[p] for p in fixture.E.parameters())
    for opt in (fixture.opt_g, fixture.opt_d):
        assert opt.optimism_stats["steps"] == 5
        assert opt.optimism_stats["extrapolated_applications"] > 0
        assert all(PREVIOUS in state for state in opt.state.values())


def test_field_identity_and_scientific_manifest():
    from experiments.forge.boundaries import recipe_field_owner, validate_registry
    from experiments.forge.techniques import validate_same_technique
    from experiments.forge.sources import source_files
    validate_registry()
    assert recipe_field_owner("optimizer_optimism") == "technique"
    with pytest.raises(ValueError, match="optimism|anticipation"):
        validate_same_technique(get_recipe("bcap"), get_recipe("bcap", optimizer_optimism="raw_gradient"))
    assert ROOT / "particlegan/optim/optimism.py" in source_files(ROOT)
