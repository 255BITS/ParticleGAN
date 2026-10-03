"""Bounded CPU structural controls; no full-budget acquisition/qualification."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from particlegan import get_recipe, ParticlePrior
from particlegan.particle_prior import MoGParticlePrior
from experiments.forge import ae_routed_policy_contracts as contract
from experiments.forge import ae_routed_policy_adapters as adapter
from experiments.forge.policy_adapters import typed_state_digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(1)


@pytest.fixture
def task():
    return contract.make_ae_variant(ROOT)


def candidate():
    return {"id": contract.FAMILY, "recipe_preset": "atlas", "task_cohort": contract.COHORT,
            "recipe_overrides": deepcopy(contract.SHARED_OVERRIDES)}


def test_original_architecture_objective_prior_horizon_and_gates_are_retained(task):
    value = contract.validate_ae_task(task, root=ROOT)
    assert task["execution"]["prior"] == {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}
    assert task["execution"]["steps"] == 250 and task["evaluation"]["observations"] == 24
    assert task["evaluation"]["thresholds"] == [["recon_mse", "<=", .05], ["hold", "<=", .35]]
    assert value["objective"] == {"reconstruction_weight": 1., "adversarial_weight": 1.,
                                   "cover_weight": 1.5, "particle_l2": .02, "fm_weight": 0.}
    assert value["evidence_reuse"] is False and value["independent_atlas_qualification"] is False
    assert value["guard"]["heldout_generalization_claim"] is False
    assert contract.ae_recipe_overrides(task) == contract.ADAPTATION
    assert contract.ae_recipe_overrides(task)["sigma_rel"] == 0.
    assert value["prior_requirement"]["sigma"] == .025
    assert contract.validate_ae_observation(task)["latent_policy"] == "clean_complete_routed_function_without_DV12_evaluation_draw"
    assert value["independent_atom_bh_or_feature_cell_claim"] is False


@pytest.mark.parametrize("change", ["cloud", "width", "standardize", "horizon", "recon", "hold", "owners", "family"])
def test_ae_variant_rejects_silent_law_objective_or_owner_changes(task, change):
    if change == "cloud": task["execution"]["prior"]["kind"] = "particle_cloud"
    elif change == "width": task["execution"]["prior"]["sigma"] = 0.
    elif change == "standardize": task["execution"]["prior"]["standardize"] = True
    elif change == "horizon": task["execution"]["steps"] = 2
    elif change == "recon": task["evaluation"]["thresholds"][0][2] = .5
    elif change == "hold": task["evaluation"]["thresholds"][1][2] = 3.5
    elif change == "owners": task["execution"]["policy_contract"]["row_semantics"] = "independent"
    else: task["policy_family"] = "atlas"
    with pytest.raises(ValueError): contract.validate_ae_task(task)


def callback_models():
    # This original AE Recipe is solely a public algebra control, not the
    # production Atlas variant or an owner-disabled training replacement.
    recipe = get_recipe("ae_gan", num_particles=12, z_dim=2, standardize=False)
    rng = torch.Generator().manual_seed(0)
    prior = MoGParticlePrior(num_particles=12, z_dim=2, sigma=.025, standardize=False, generator=rng)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        encoder, generator, critic = adapter.host.MLP(2, 4), adapter.host.MLP(2, 2), adapter.host.MLP(2, 1)
    return {"encoder": encoder, "generator": generator, "critic": critic, "prior": prior,
            "router": adapter.AERouter(recipe)}, rng


def test_complete_clean_callback_matches_public_hard_encode_outputs_and_both_gradient_paths():
    models, rng = callback_models()
    data = torch.randn((8, 2), generator=rng)
    query, offset = models["encoder"](data).chunk(2, 1)
    encoded = models["router"].recipe.encode(query, models["prior"], offset=offset)
    expected = models["generator"](encoded.codes[:, 0])
    parameters = [models["prior"].z, *models["encoder"].parameters(), *models["generator"].parameters()]
    expected_grad = torch.autograd.grad(expected.square().sum(), parameters)
    rows = adapter.routing_spec()
    candidate = rows.candidate_for(models, models["prior"].z)
    actual = rows.forward(models, adapter.ae_context(data), candidate)
    actual_grad = torch.autograd.grad(actual.square().sum(), parameters)
    assert torch.equal(actual, expected)
    for observed, reference in zip(actual_grad, expected_grad):
        assert torch.equal(observed, reference)
    assert torch.count_nonzero(actual_grad[0]) > 0
    assert any(torch.count_nonzero(value) > 0 for value in actual_grad[1:5])


def test_candidate_table_and_mass_are_not_borrowed_from_live_owners():
    models, rng = callback_models()
    contexts = adapter.mog_context(torch.tensor([.1, .9]), torch.zeros(2, 2))
    rows = adapter.routing_spec(); original = rows.candidate_for(models, models["prior"].z)
    # Retire every row except row 3. Original model prior values remain intact;
    # counterfactual uses its own table and complete mass-aware function.
    table = original.table.detach().clone() + 5.
    mass = torch.full_like(original.log_mass, -torch.inf); mass[3] = 0.
    changed = type(original)(table=table, log_mass=mass, row_state=original.row_state)
    result = rows.forward(models, contexts, changed)
    assert torch.equal(result, models["generator"](table[3].expand(2, 2)))
    assert not torch.equal(result, rows.forward(models, contexts, original))


def test_complete_callback_uses_supplied_encoder_and_generator_with_fixed_prior_width():
    models, rng = callback_models()
    data = torch.randn((8, 2), generator=rng)
    rows = adapter.routing_spec(); candidate = rows.candidate_for(models, models["prior"].z)
    context = adapter.ae_context(data)
    original = rows.forward(models, context, candidate)
    changed = {**models, "encoder": deepcopy(models["encoder"]), "generator": deepcopy(models["generator"])}
    with torch.no_grad():
        changed["encoder"].net[-1].bias[2:].add_(2.)
        changed["generator"].net[-1].bias.add_(3.)
    query, offset = changed["encoder"](data).chunk(2, 1)
    codes = changed["router"].recipe.encode(query, adapter.FunctionalMoG(candidate.table, changed["prior"].sigma), offset=offset).codes[:, 0]
    actual = rows.forward(changed, context, candidate)
    assert torch.equal(actual, changed["generator"](codes))
    assert not torch.equal(actual, original)
    assert torch.equal(models["prior"].sigma, torch.tensor(.025))


def test_generation_callback_retains_positive_fixed_mog_noise_and_original_clean_sampler_scope():
    models, rng = callback_models()
    normal = torch.tensor([[2., -1.], [1., 3.]])
    uniform = torch.tensor([.1, .7])
    context = adapter.mog_context(uniform, normal)
    rows = adapter.routing_spec(); candidate = rows.candidate_for(models, models["prior"].z)
    selected = torch.floor(uniform * len(candidate.table)).long()
    expected = models["generator"](candidate.table[selected] + models["prior"].sigma * normal)
    assert torch.equal(rows.forward(models, context, candidate), expected)
    assert adapter.ae_sampling_receipt() == {"sampling_contract_version": 1,
        "sampling_law": contract.SAMPLING_LAW, "eval_output_noise": contract.EVAL_OUTPUT_NOISE}


def test_actual_mixed_code_perturbation_reaches_decoder_and_preserves_context_shape():
    models, rng = callback_models()
    context = adapter.mog_context(torch.tensor([.1, .7]), torch.zeros(2, 2))
    rows = adapter.routing_spec(); candidate = rows.candidate_for(models, models["prior"].z)
    seen = []
    def perturb(codes):
        seen.append(codes.detach().clone())
        return codes + .25
    clean = rows.forward(models, context, candidate)
    actual = rows.forward(models, context, candidate, perturb_fn=perturb)
    assert len(seen) == 1 and seen[0].shape == (2, 2)
    assert torch.equal(actual, models["generator"](seen[0] + .25))
    assert not torch.equal(actual, clean)


@pytest.mark.parametrize("change", ["cloud", "sigma", "standardize", "branch", "uniform", "nan"])
def test_callback_refuses_invalid_prior_or_context(change):
    models, rng = callback_models()
    context = adapter.mog_context(torch.tensor([.1, .7]), torch.zeros(2, 2))
    rows = adapter.routing_spec(); candidate = rows.candidate_for(models, models["prior"].z)
    if change == "cloud": models["prior"] = ParticlePrior(num_particles=12, z_dim=2, generator=rng)
    elif change == "sigma": models["prior"].set_sigma(.03)
    elif change == "standardize": models["prior"].standardize = True
    elif change == "branch": context[0, 0] = 2.
    elif change == "uniform": context[0, 3] = 1.
    else: context[0, 1] = torch.nan
    with pytest.raises(ValueError): rows.forward(models, context, candidate)


def test_unmodified_independent_contract_remains_incompatible_with_ae_mog():
    with pytest.raises(ValueError):
        get_recipe("atlas", encoder_mode="ae", prior_kind="mog", sigma_rel=.025)


def test_named_variant_never_hides_an_unavailable_routed_mog_owner(task):
    blockers = contract.ae_routed_preflight(task, candidate())
    if blockers:
        assert any("particle_birth_death" in message and "particles" in message for message in blockers)
        with pytest.raises(ValueError): contract.resolved_recipe(candidate(), task)
        assert contract.ae_policy_contract_blockers(task, {})
    else:
        recipe = contract.resolved_recipe(candidate(), task)
        assert recipe.particle_birth_death is True and recipe.row_evidence_gate is True
        assert recipe.prior_kind == "mog" and recipe.row_policy == "routed_paired"
        assert recipe.encoder_mode == "ae" and recipe.total_steps is None and recipe.sigma_rel == 0.
        assert contract.ae_policy_contract_blockers(task, recipe) == []
        assert contract.ae_policy_contract_blockers(task, recipe.to_dict()) == []
        changed = recipe.to_dict(); changed["prior_kind"] = "particles"
        assert contract.ae_policy_contract_blockers(task, changed)


def ready_fixture(task):
    blockers = contract.ae_routed_preflight(task, candidate())
    if blockers:
        pytest.skip("source-bound public routed-MoG support unavailable: " + "; ".join(blockers))
    return adapter.AERoutedFixture({"candidate": candidate(), "protocol": {"seed": 0}}, task, device="cpu")


def test_real_two_update_lifecycle_selected_observation_and_complete_checkpoint_parity(task):
    fixture = ready_fixture(task)
    fixture.step(); checkpoint = deepcopy(fixture.state_dict())
    before = typed_state_digest(checkpoint)
    metrics = fixture.observe()
    assert set(metrics) == {"recon_mse", "hold"} and fixture.purity[-1]["pure"] is True
    assert typed_state_digest(fixture.state_dict()) == before
    fixture.step(); expected = fixture.state_dict()
    resumed = ready_fixture(task); resumed.load_state_dict(checkpoint); resumed.step()
    assert typed_state_digest(resumed.state_dict()) == typed_state_digest(expected)
    assert typed_state_digest(checkpoint) == before
    assert fixture.audit.receipt(2)["complete"] is True
    assert all(v == 2 for v in fixture.guards()["optimizer_updates"].values())
    assert fixture.max_steps == 250 and fixture.recipe.total_steps is None


@pytest.mark.parametrize("change", ["cursor", "table", "sigma", "stream", "hook", "mode", "fixed_width", "extra_state"])
def test_complete_checkpoint_refuses_corrupt_bound_owners(task, change):
    fixture = ready_fixture(task); fixture.step()
    state = deepcopy(fixture.state_dict()); before = typed_state_digest(fixture.state_dict())
    if change == "cursor": state["caller_cursor"] = 2
    elif change == "table": state["policy"]["models"]["prior"]["z"][0, 0] = torch.nan
    elif change == "sigma": state["recipe"]["sigma_rel"] = .025
    elif change == "stream": state["streams"]["states"][next(iter(state["streams"]["states"]))] = torch.zeros(2, dtype=torch.uint8)
    elif change == "hook": state["lifecycle_audit"]["calls"][next(iter(state["lifecycle_audit"]["calls"]))] -= 1
    elif change == "mode": state["module_modes"]["encoder"][""] = False
    elif change == "fixed_width": state["policy"]["models"]["prior"]["sigma"] *= 2
    else: state["policy"]["models"]["prior"]["_extra_state"]["standardize"] = True
    with pytest.raises((ValueError, RuntimeError)): fixture.load_state_dict(state)
    assert typed_state_digest(fixture.state_dict()) == before


def test_no_cuda_initialization():
    assert not torch.cuda.is_initialized()
