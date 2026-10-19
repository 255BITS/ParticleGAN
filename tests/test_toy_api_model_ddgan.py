"""Scientific scorer controls and public-component execution/replay contracts."""
from copy import deepcopy
import json

import pytest
import torch

from particlegan import DDGAN
from experiments.forge.rng import NamedStreams
from experiments.forge.state import state_digest
from benchmarks.toy_audit.api_contract import validate_case, validate_observation
from benchmarks.toy_audit.api_model_ddgan import (
    ALPHA_BAR, BASELINE, CASES, ROOT, MixtureTarget, build_case,
    distribution_failures, distribution_metrics, final_metrics, list_cases,
    sample_chain,
)


@pytest.fixture(autouse=True)
def single_torch_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def rng(component):
    return NamedStreams(0).generator("eval", component=component)


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_exact_target_and_conditional_oracles_pass_frozen_bounds(kind):
    target, schedule = MixtureTarget(kind), DDGAN(ALPHA_BAR)
    samples = target.sample(2048, rng(kind + "_final"))
    _, failures = final_metrics(samples, target)
    assert failures == []
    for step in range(1, 5):
        for index, xy in enumerate(((-.5, .25), (.5, -.25))):
            parameters = target.transition_parameters(xy, step, schedule)
            samples = target.draw(*parameters, 2048, rng(f"{kind}_{step}_{index}"))
            assert distribution_failures(distribution_metrics(samples, *parameters)) == []


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_scorer_rejects_mean_collapse_identity_and_nonfinite_output(kind):
    target = MixtureTarget(kind)
    collapsed = (target.weights[:, None] * target.means).sum(0).expand(2048, 2)
    assert "final_covariance_eigen_min" in final_metrics(collapsed, target)[1]
    identity = torch.randn(4096, 2, generator=rng("identity"))
    assert final_metrics(identity, target)[1]
    assert final_metrics(torch.full((2048, 2), float("nan")), target)[1]
    if kind == "bimodal":
        center_only = target.means[torch.arange(2048) % 2]
        assert "final_local_covariance_min" in final_metrics(center_only, target)[1]
        one_mode = target.draw(torch.ones(1), target.means[:1], target.variances[:1],
            4096, rng("one_mode"))
        assert "final_mode_mass_tv" in final_metrics(one_mode, target)[1]


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_transition_scorer_rejects_correct_marginal_without_condition(kind):
    target, schedule = MixtureTarget(kind), DDGAN(ALPHA_BAR)
    parameters = target.transition_parameters((-.5, .25), 1, schedule)
    marginal = target.sample(8192, rng("marginal"))
    assert distribution_failures(distribution_metrics(marginal, *parameters))
    wrong = target.transition_parameters((.5, -.25), 1, schedule)
    wrong_condition = target.draw(*wrong, 8192, rng("wrong_condition"))
    assert distribution_failures(distribution_metrics(wrong_condition, *parameters))
    posterior_mean = (parameters[0][:, None] * parameters[1]).sum(0).expand(8192, 2)
    assert "covariance_eigen_min" in distribution_failures(distribution_metrics(posterior_mean, *parameters))


def test_analytic_posterior_matches_independent_gaussian_precision_formula():
    target = MixtureTarget("gaussian")
    xt = torch.tensor([[-.5, .25], [.5, -.25]])
    _, means, variances = target.posterior_parameters(xt, .5)
    v = (target.variances.reciprocal() + 1).reciprocal()
    expected = v * (target.means / target.variances + 2 ** .5 * xt)
    torch.testing.assert_close(variances[:, 0], v.expand_as(xt))
    torch.testing.assert_close(means[:, 0], expected)


@pytest.mark.parametrize("case", list_cases(), ids=lambda case: case["id"])
def test_full_reverse_chain_oracle_matches_target_from_gaussian_start(case):
    target, schedule = MixtureTarget(case["target"]), DDGAN(ALPHA_BAR)
    xt = torch.randn(8192, 2, generator=rng("oracle_start"))
    for step in range(4, 0, -1):
        t = torch.full((len(xt),), step, dtype=torch.long)
        clean = target.oracle_clean(xt, schedule.ab[step], rng(f"oracle_clean_{step}"))
        eta = torch.randn(xt.shape, generator=rng(f"oracle_noise_{step}"))
        xt = schedule.reverse(clean, xt, t, eta)
    assert final_metrics(xt, target)[1] == []


def test_full_chain_redraws_latent_kernel_and_reverse_noise_at_each_step():
    fixture = build_case("api-model-ddgan-gaussian-chain", max_steps=1)
    draws, noises, steps = [], [], []
    original_sample, original_reverse = fixture.prior.sample, fixture.schedule.reverse
    def recorded_sample(*args, **kwargs):
        z, indices = original_sample(*args, **kwargs)
        draws.append((z.clone(), indices.clone()))
        return z, indices
    def recorded_reverse(clean, xt, t, eta):
        steps.append(int(t[0])); noises.append(eta.clone())
        return original_reverse(clean, xt, t, eta)
    fixture.prior.sample, fixture.schedule.reverse = recorded_sample, recorded_reverse
    with torch.no_grad():
        _, states = sample_chain(fixture.G, fixture.prior, fixture.schedule, 32,
            start_rng=rng("start"), latent_rng=rng("latent"), kernel_rng=rng("kernel"),
            reverse_rng=rng("reverse"), return_states=True)
    assert steps == [4, 3, 2, 1] and len(states) == 5
    for index in range(3):
        assert not torch.equal(draws[index][1], draws[index + 1][1])
        assert not torch.equal(draws[index][0], draws[index + 1][0])
        assert not torch.equal(noises[index], noises[index + 1])


@pytest.mark.parametrize("case", list_cases(), ids=lambda case: case["id"])
def test_declarations_match_forge_and_selected_global_recipe(case):
    validate_case(case)
    task = json.loads((ROOT / "configs/forge/tasks" / ("model_ddgan_" + case["target"] + "_chain.json")).read_text())
    assert task["evaluation"]["thresholds"] == case["forge_thresholds"]
    assert task["execution"]["steps"] == case["default_steps"]
    fixture = build_case(case["id"], max_steps=2)
    candidate = json.loads((ROOT / "configs/forge/ideas" / (BASELINE + ".json")).read_text())
    adapted = {"model", "conditioning", "encoder_mode", "num_classes", "alpha_bar"}
    for name, value in candidate["recipe_overrides"].items():
        if name not in adapted:
            assert json.loads(json.dumps(getattr(fixture.recipe, name))) == value, name
    assert float(fixture.prior.sigma) == .25
    assert fixture.recipe.prior_update == "learned" and fixture.recipe.prior_regularizer == "sbnp"
    assert build_case(case["id"], max_steps=2, recipe=fixture.recipe).recipe == fixture.recipe


def test_updates_checkpoint_every_consumed_stream_and_resume_exactly():
    fixture = build_case("api-model-ddgan-gaussian-chain", max_steps=3)
    global_before = torch.get_rng_state().clone()
    before = fixture.prior.z.detach().clone()
    fixture.step()
    assert not torch.equal(before, fixture.prior.z)
    assert torch.equal(global_before, torch.get_rng_state())
    state = fixture.state_dict()
    components = {value["component"] for value in state["streams"]["manifest"]["bindings"].values()}
    assert {"models", "critic", "generator", "critic_time", "generator_time",
        "critic_forward", "generator_forward", "critic_reverse", "generator_reverse",
        "critic_latent", "generator_latent", "critic_kernel", "generator_kernel", "critic_penalty"} <= components
    expected = fixture.step()
    restored = build_case("api-model-ddgan-gaussian-chain", max_steps=3)
    restored.load_state_dict(state)
    assert restored.step() == expected
    assert state_digest(restored.state_dict()) == state_digest(fixture.state_dict())
    assert torch.equal(global_before, torch.get_rng_state())


def test_evaluation_is_pure_and_short_observations_fail_default_protocol():
    fixture = build_case("api-model-ddgan-gaussian-chain", max_steps=1)
    fixture.step()
    before = state_digest(fixture.state_dict())
    observation = validate_observation(fixture.observe(n=128))
    assert state_digest(fixture.state_dict()) == before
    assert not observation["passed"] and "samples_per_panel" in observation["failed_bounds"]
    assert observation["metrics"]["panel_count"] == 8


def test_public_policy_lifecycle_and_critic_diagnostics_are_exercised():
    fixture = build_case("api-model-ddgan-gaussian-chain", max_steps=1)
    order = []
    for name in ("begin_step", "observe_support", "observe_critic_pair", "after_critic_step",
            "after_generator_backward", "after_generator_step", "finish_step"):
        original = getattr(fixture.policy, name)
        def recorded(*args, _name=name, _original=original, **kwargs):
            order.append(_name)
            return _original(*args, **kwargs)
        setattr(fixture.policy, name, recorded)
    fixture.step()
    assert order == ["begin_step", "observe_support", "observe_critic_pair", "after_critic_step",
        "after_generator_backward", "after_generator_step", "finish_step"]
    assert fixture.policy.completed_steps == fixture.completed_steps == 1
    assert fixture.penalty.last_stats


def test_task_binding_preserves_frozen_prior_policy_and_rejects_unbound_policies():
    baseline = build_case("api-model-ddgan-gaussian-chain", max_steps=1)
    frozen = build_case("api-model-ddgan-gaussian-chain", max_steps=1,
        recipe=baseline.recipe.replace(prior_update="frozen"))
    prior = frozen.prior.z.clone()
    frozen.step()
    torch.testing.assert_close(prior, frozen.prior.z, atol=0, rtol=0)
    assert frozen.last_losses["prior_regularization"] == 0.
    for changed in (dict(input_noise_std=.1), dict(output_noise_std=.1),
            dict(serve_average=.9), dict(particle_birth_death=True)):
        with pytest.raises(ValueError):
            build_case("api-model-ddgan-gaussian-chain", recipe=baseline.recipe.replace(**changed))
    bad = deepcopy(baseline.state_dict()); bad["case_id"] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        baseline.load_state_dict(bad)
