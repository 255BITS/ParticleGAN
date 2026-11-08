"""Software controls for formula-faithful vector execution, without quality credit."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from benchmarks.toy_audit import api_vectors as api
from tests.test_toy_api_vectors import assert_nested_equal
from particlegan import get_recipe


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_case(identifier):
    case = deepcopy(api._registry()[identifier])
    case.update(particles=32, batch_size=16)
    return case


@pytest.mark.parametrize("import_first", [False, True])
def test_shared_generator_architecture_does_not_depend_on_vector_import(import_first):
    source = f'''
import json
import torch
if {import_first!r}:
    import benchmarks.toy_audit.api_vectors
import lib.toy_models as models
original = models.SimpleMLPGenerator
torch.manual_seed(0)
before = original(2, 32, 16, 2)
import benchmarks.toy_audit.api_vectors as api
torch.manual_seed(0)
after = models.SimpleMLPGenerator(2, 32, 16, 2)
print(json.dumps({{
    "same_class": original is models.SimpleMLPGenerator is api.SimpleMLPGenerator,
    "normalizations": [sum(isinstance(m, torch.nn.LayerNorm) for m in g.modules()) for g in (before, after)],
    "same_initial_state": all(torch.equal(v, after.state_dict()[k]) for k, v in before.state_dict().items()),
}}))
'''
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[1], check=True)
    assert json.loads(result.stdout) == {
        "same_class": True, "normalizations": [0, 0], "same_initial_state": True}


@pytest.mark.parametrize("identifier,expected_class,normalizations,residual_scale", [
    ("api-grid100-mlp16", api.SimpleMLPGenerator, 0, None),
    ("api-grid100-res16-tanh", api.ResidualMLPGenerator, 0, 1.),
    ("api-grid100-res16-scaled-tanh", api.ScaledResidualMLPGenerator, 1, 1 / 14 ** .5),
])
def test_deep_tasks_construct_distinct_declared_architectures(identifier, expected_class,
                                                            normalizations, residual_scale):
    fixture = api.VectorFixture(small_case(identifier), recipe_name="k3p", max_steps=1)
    generator = fixture.trainer.G
    assert type(generator) is expected_class
    assert sum(isinstance(m, torch.nn.LayerNorm) for m in generator.modules()) == normalizations
    if residual_scale is not None:
        assert all(block.scale == pytest.approx(residual_scale) for block in generator.blocks)
    assert fixture.metadata["architecture"]["generator"] == expected_class.__name__


def test_scaled_task_has_no_model_or_prior_writes_outside_the_trainer(monkeypatch):
    """Cross all former helper deadlines with the optimizer deliberately inactive.

    This is a writer-ownership check, not a learned-quality experiment. The
    former empirical helper reset the perturbed generator and rewrote rows.
    """
    fixture = api.VectorFixture(small_case("api-grid100-res16-scaled-tanh"),
                                recipe_name="k3p", max_steps=100)
    with torch.no_grad():
        fixture.trainer.G.head.bias.add_(.3)
        fixture.trainer.prior.z.add_(2.)
    before = deepcopy(fixture.trainer.state_dict())
    calls = []

    def inactive_step(real, *, generator_real):
        calls.append((real.clone(), generator_real.clone()))
        return {"step": len(calls)}

    monkeypatch.setattr(fixture.trainer, "step", inactive_step)
    for _ in range(100):
        fixture.step()
    assert len(calls) == fixture.completed_steps == 100
    assert_nested_equal(before, fixture.trainer.state_dict())
    assert not hasattr(fixture.trainer, "_scaled_res16_emp_state")


def test_scaled_fixture_fresh_object_continuation_matches_uninterrupted_state():
    case = small_case("api-grid100-res16-scaled-tanh")
    source = api.VectorFixture(case, recipe_name="k3p", max_steps=3)
    source.step()
    saved = source.state_dict()
    restored = api.VectorFixture(case, recipe_name="k3p", max_steps=3)
    restored.load_state_dict(saved)
    assert_nested_equal(saved, restored.state_dict())
    source.step()
    restored.step()
    assert_nested_equal(source.state_dict(), restored.state_dict())


def test_pre_repair_vector_checkpoint_requires_its_original_source():
    fixture = api.VectorFixture(small_case("api-grid100-res16-scaled-tanh"),
                                recipe_name="k3p", max_steps=1)
    before = fixture.state_dict()
    historical = deepcopy(before)
    historical["version"] = "toy-public-api-vectors-v2"
    with pytest.raises(ValueError, match="identity/recipe"):
        fixture.load_state_dict(historical)
    assert_nested_equal(before, fixture.state_dict())


def test_soft_bounded_tanh_is_the_derivative_of_its_forward():
    u = torch.tensor([-15., -1., 0., 1., 15.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(api.TanhRailGuard.apply, (u,))
    assert torch.autograd.gradgradcheck(api.TanhRailGuard.apply, (u,))
    zero = torch.autograd.grad(api.TanhRailGuard.apply(u), u, torch.zeros_like(u))[0]
    assert torch.equal(zero, torch.zeros_like(u))
    one = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
    observed = torch.autograd.grad(api.TanhRailGuard.apply(one), one)[0]
    expected = 1 - torch.tanh(one / 5.25).square()
    torch.testing.assert_close(observed, expected)


def test_task_profiles_cannot_replace_requested_trainer_fields():
    case = small_case("api-vector-two-broad")
    case["spec"].update(lr=9., d_lr_mult=8., prior_lr_mult=7., prior_reg=6.,
                        betas=[.1, .2], ema_decay=.3)
    case["recipe_overrides"] = {"lr": 100., "loss": "relativistic"}
    fixture = api.VectorFixture(case, recipe_name="bispf", max_steps=1)
    expected = get_recipe("bispf", num_particles=case["particles"],
                          z_dim=case["z_dim"], batch_size=case["batch_size"],
                          total_steps=case["default_steps"])
    assert fixture.recipe == expected


def test_scaled_metadata_declares_the_clean_law_actually_sampled(monkeypatch):
    case = small_case("api-grid100-res16-scaled-tanh")
    fixture = api.VectorFixture(case, recipe_name="bispf", max_steps=1)
    original = fixture.trainer.sample
    flags = []
    def sample(*args, **kwargs):
        flags.append(kwargs["output_noise"])
        return original(*args, **kwargs)
    monkeypatch.setattr(fixture.trainer, "sample", sample)
    fixture.observe(n=32)
    assert flags and not any(flags)
    assert "without additive output noise" in fixture.metadata["sampling"]


def test_intact_ka2_default_caller_is_blocked_before_optimizers_are_built(monkeypatch):
    from benchmarks.toy_audit.api_contract import UnsupportedRecipeOverrides
    def optimizer_forbidden(*args, **kwargs):
        raise AssertionError("unsupported candidate reached optimizer construction")
    monkeypatch.setattr(api.Recipe, "make_optimizers", optimizer_forbidden)
    case = api._registry()["api-reserved-alternating-critic-updates"]
    assert case["default_recipe"] == "ka2"
    with pytest.raises(UnsupportedRecipeOverrides, match="intact recipe"):
        api.VectorFixture(case, recipe_name="auto", max_steps=1)


def test_caller_checkpoint_restore_is_explicitly_unsupported_and_never_mutates():
    from benchmarks.toy_audit.api_contract import UnsupportedRecipeOverrides
    case = small_case("api-reserved-alternating-critic-updates")
    fixture = api.VectorFixture(case, recipe_name="bispf", max_steps=3)
    fixture.step()
    before = fixture.state_dict()
    assert before["checkpoint_scope"] == "export_only_no_resume"
    assert fixture.metadata["checkpoint_capabilities"] == dict(
        state_export="diagnostic_only", resume=False, exact_continuation=False)
    assert set(before["caller"]["penalty"]) == {"coeff", "last_stats", "cooler"}
    corrupt = deepcopy(before)
    corrupt["caller"]["generator"]["net.0.weight"].add_(2)
    corrupt["caller"]["latent_rng"] = torch.ones(1, dtype=torch.uint8)
    with pytest.raises(UnsupportedRecipeOverrides, match="complete checkpoint restore is unsupported"):
        fixture.load_state_dict(corrupt)
    assert_nested_equal(before, fixture.state_dict())


def test_required_checkpoint_contract_blocks_caller_before_construction(monkeypatch):
    from benchmarks.toy_audit import api_contract
    case = small_case("api-reserved-alternating-critic-updates")
    case.update(provider="api_vectors", checkpoint_required=True)
    def optimizer_forbidden(*args, **kwargs):
        raise AssertionError("missing checkpoint capability reached optimizer construction")
    monkeypatch.setattr(api.Recipe, "make_optimizers", optimizer_forbidden)
    with pytest.raises(api_contract.UnsupportedRecipeOverrides, match="checkpoint continuation"):
        api_contract.build(case, recipe_name="bispf", max_steps=1)
    with pytest.raises(api_contract.UnsupportedRecipeOverrides, match="checkpoint continuation"):
        api.VectorFixture(case, recipe_name="bispf", max_steps=1)


def test_original_caller_default_is_a_blocked_receipt_without_training(tmp_path):
    from benchmarks.toy_audit import api_run
    case = api._registry()["api-reserved-alternating-critic-updates"]
    case["provider"] = "api_vectors"
    result = api_run.run_case(case, tmp_path / "blocked-ka2", recipe_name="auto", steps=1)
    assert case["default_recipe"] == "ka2"
    assert result["status"] == "BLOCKED" and result["verdict"] == "FAIL"
    assert result["completed_updates"] == 0 and result["artifacts"] == {}
    assert result["checkpoint_capabilities"] == case["checkpoint_capabilities"]
    assert "intact recipe mechanisms" in result["failed_bounds"][0]


def test_fixed_penalty_question_requires_matching_explicit_recipe():
    from benchmarks.toy_audit.api_contract import UnsupportedRecipeOverrides
    case = small_case("api-stress-r1-r2")
    with pytest.raises(UnsupportedRecipeOverrides, match="explicitly declared matching recipe"):
        api.VectorFixture(case, recipe_name="auto", max_steps=1)
    declared = get_recipe("k3p", num_particles=case["particles"], z_dim=case["z_dim"],
                          batch_size=case["batch_size"], total_steps=case["default_steps"],
                          reg_arm=case["penalty_arm"], reg_coeff=case["spec"]["reg_coeff"])
    fixture = api.VectorFixture(case, recipe=declared, max_steps=1)
    assert fixture.recipe == declared
    fixture.step()


def test_supplied_complete_recipe_is_retained_without_task_tuning():
    case = small_case("api-reserved-alternating-critic-updates")
    declared = get_recipe("ka2", num_particles=case["particles"], z_dim=case["z_dim"],
                          batch_size=case["batch_size"], total_steps=case["default_steps"],
                          input_noise_std=0., output_noise_std=0., latent_damping_max_rate=0.)
    fixture = api.VectorFixture(case, recipe=declared, max_steps=1)
    assert fixture.recipe == declared
    assert fixture.step()["g"] == 1
