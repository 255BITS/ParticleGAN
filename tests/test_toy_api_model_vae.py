"""Meaningful guards for the VAE scientific contract and exact public execution."""
from copy import deepcopy
import math

import pytest
import torch

from benchmarks.toy_audit import api_contract
from benchmarks.toy_audit.api_model_vae import (
    BASELINE, OBSERVATION_SIGMA, PRIOR_SIGMA, build_case, list_cases, observed_negative_elbo,
    sample_target, score_outputs, task_recipe,
)


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _oracle(mode):
    n = 4096
    rng = torch.Generator().manual_seed(713)
    data = sample_target(n, rng, "cpu")
    reconstruction = data[:, None].repeat(1, 8, 1) + .005 * torch.randn(n, 8, 2, generator=rng)
    log_probs = torch.full((n, 16), -torch.inf)
    if mode == "hard":
        log_probs.scatter_(1, (data[:, 0] > 0).long()[:, None], 0.)
    else:
        for side in (0, 1):
            log_probs[(data[:, 0] > 0) == bool(side), side*8:(side+1)*8] = -math.log(8)
    q = log_probs.exp()
    kl = math.log(16) + (q * log_probs.nan_to_num(neginf=0)).sum(1)
    recon = (reconstruction - data[:, None]).square().mean()
    return dict(data=data, reconstruction=reconstruction, shuffled=reconstruction.roll(n // 3, 0),
                posterior_codes=PRIOR_SIGMA * torch.randn(n, 8, 2, generator=rng),
                posterior_centers=torch.zeros(n, 8, 2), log_probs=log_probs, kl=kl,
                negative_elbo=recon / OBSERVATION_SIGMA**2 + kl.mean() + math.log(2 * math.pi * OBSERVATION_SIGMA**2),
                decoder_means=sample_target(n, rng, "cpu"),
                likelihood=sample_target(n, rng, "cpu") + OBSERVATION_SIGMA * torch.randn(n, 2, generator=rng))


@pytest.mark.parametrize("mode", ["hard", "categorical"])
def test_oracle_and_destructive_controls(mode):
    oracle = _oracle(mode)
    metrics, failures = score_outputs(**oracle)
    assert not failures, failures
    assert metrics["kl_nats"] > 0  # The hard constant is retained, not reported as zero.
    for destructive, expected in (
        ({"reconstruction": oracle["shuffled"]}, "reconstruction_mse"),
        ({"posterior_codes": oracle["posterior_centers"]}, "posterior_noise_variance_ratio"),
        ({"log_probs": torch.full_like(oracle["log_probs"], -math.log(16))}, "posterior_mutual_information"),
        ({"decoder_means": torch.zeros_like(oracle["decoder_means"])}, "decoder_mean_hq"),
        ({"likelihood": torch.zeros_like(oracle["likelihood"])}, "likelihood_hq"),
        ({"kl": torch.zeros_like(oracle["kl"])}, "kl_identity_error"),
    ):
        _, failed = score_outputs(**{**oracle, **destructive})
        assert any(expected in item for item in failed), (expected, failed)


@pytest.mark.parametrize("case", list_cases(), ids=lambda case: case["id"])
def test_exact_resume_and_evaluation_isolation(case):
    api_contract.validate_case(case)
    torch.set_num_threads(1)
    fixture = build_case(case["id"], max_steps=4)
    fixture.step()
    checkpoint = fixture.state_dict()
    observation = api_contract.validate_observation(fixture.observe(n=128, seed=713))
    assert observation["metrics"]["kl_nats"] >= 0
    _equal(checkpoint, fixture.state_dict())
    fixture.step()
    expected = fixture.state_dict()
    restored = build_case(case["id"], max_steps=4)
    restored.load_state_dict(checkpoint)
    restored.step()
    _equal(expected, restored.state_dict())
    assert {group.get("role") for group in fixture.opt_g.param_groups} == {"generator", "encoder", "prior"}
    assert all(fixture.recipe.to_dict()[key] == restored.recipe.to_dict()[key] for key in fixture.recipe.to_dict())


def test_baseline_binding_and_protocol_refusals():
    import json
    from benchmarks.toy_audit.api_model_vae import ROOT
    from particlegan import Recipe
    candidate = json.loads((ROOT / f"configs/forge/ideas/{BASELINE}.json").read_text())
    baseline = Recipe(**candidate["recipe_overrides"])
    case = list_cases()[0]
    recipe = task_recipe(case)
    fixture = build_case("api-model-vae-hard", recipe=recipe, max_steps=1)
    task_fields = {"model", "conditioning", "encoder_mode", "observation_sigma", "prior_kind", "sigma_rel",
                   "standardize", "num_particles", "z_dim", "batch_size", "total_steps"}
    assert all(value == fixture.recipe.to_dict()[key] for key, value in baseline.to_dict().items() if key not in task_fields)
    assert fixture.recipe == recipe
    assert fixture.recipe.total_steps == 1200
    assert float(fixture.prior.sigma) == pytest.approx(PRIOR_SIGMA)
    for delta in ({"model": "ddgan"}, {"encoder_mode": "categorical"}, {"num_particles": 32},
                  {"z_dim": 4}, {"batch_size": 128}, {"observation_sigma": .1}, {"total_steps": 2400}):
        with pytest.raises(ValueError, match="task-owned"):
            build_case(case["id"], recipe=recipe.replace(**delta), max_steps=1)
    with pytest.raises(ValueError, match="seed 0"):
        build_case("api-model-vae-hard", seed=1)
    with pytest.raises(ValueError, match="unknown"):
        build_case("api-model-vae-nonexistent")
    with pytest.raises(ValueError, match="allowance"):
        build_case("api-model-vae-hard", max_steps=1201)


@pytest.mark.parametrize("mode", ["hard", "categorical"])
def test_elbo_identity_uses_same_observations_without_float32_cancellation(mode):
    from particlegan import ParticleEncoding
    values = _oracle(mode)
    # Before training, the reconstruction error yields thousands of nats.
    # The diagnostic must remain precise without another model forward/draw.
    data = values["data"]
    reconstruction = -data[:, None].repeat(1, 8, 1) + .137
    indices = values["log_probs"].argmax(1)[:, None].expand(-1, 8)
    encoded = ParticleEncoding(values["posterior_codes"], indices,
                               values["kl"], values["log_probs"], mode)
    elbo = observed_negative_elbo(encoded, reconstruction, data)
    expected = (-torch.distributions.Normal(reconstruction.double(), OBSERVATION_SIGMA)
                .log_prob(data[:, None].double()).sum(-1).mean() + values["kl"].double().mean())
    torch.testing.assert_close(elbo, expected, rtol=0, atol=1e-9)
    assert elbo.dtype == torch.float64 and float(elbo) > 1000
    metrics, failed = score_outputs(**{**values, "reconstruction": reconstruction, "negative_elbo": elbo})
    assert metrics["elbo_identity_error"] <= 1e-4
    assert not any("elbo_identity_error" in bound for bound in failed)
    _, corrupted = score_outputs(**{**values, "reconstruction": reconstruction, "negative_elbo": elbo + .01})
    assert any("elbo_identity_error" in bound for bound in corrupted)


def test_prior_freezing_preserves_shared_initial_distribution():
    case = list_cases()[0]
    recipe = task_recipe(case)
    learned = build_case(case["id"], recipe=recipe, max_steps=1)
    frozen = build_case(case["id"], recipe=recipe.replace(prior_update="frozen"), max_steps=1)
    torch.testing.assert_close(learned.prior.z, frozen.prior.z, rtol=0, atol=0)
    assert learned.initial_model_hashes == frozen.initial_model_hashes
    assert learned.prior.z.requires_grad and not frozen.prior.z.requires_grad
    before = frozen.prior.z.detach().clone()
    frozen.step()
    torch.testing.assert_close(before, frozen.prior.z, rtol=0, atol=0)
    assert all(group.get("role") != "prior" for group in frozen.opt_g.param_groups)
