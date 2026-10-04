"""Tiny CPU structural controls for the named routed unused-token family.

These execute at most three updates on each software fixture and grant no
200-update/CUDA/default scientific qualification.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from benchmarks.locked_shared.hosts import unused_token_hold as host
from particlegan import init
from experiments.forge import routed_policy_contracts as contract
from experiments.forge.routed_policy_adapters import UnusedTokenRoutedFixture, slot_contexts, routing_spec
from experiments.forge.policy_adapters import evaluation_state, typed_state_digest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_one_thread(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def definition():
    task = contract.make_unused_variant(ROOT)
    request = {"candidate": {"id": "synthetic_atlas_routed_software_control", "task_cohort": contract.COHORT,
                             "recipe_preset": "atlas", "recipe_overrides": deepcopy(contract.SHARED_OVERRIDES)},
               "protocol": {"seed": 0}}
    return request, task


def fixture():
    request, task = definition()
    return UnusedTokenRoutedFixture(request, task)


def test_original_scaffold_resources_initializer_and_public_owners_are_bound():
    original_declarations = init.declarations(host.SharedSlotStudent())
    value = fixture()
    assert isinstance(value.G, host.SharedSlotStudent) and isinstance(value.D, host.SlotCritic)
    assert set(dict(value.G.named_parameters())) == {"shared", "slot"}
    assert value.policy.table is value.G.slot and value.policy._table_location == ("generator", "slot")
    assert value.recipe.total_steps is None and value.max_steps == host.STEPS == 200
    assert value.recipe.prior_kind == "particles" and value.recipe.sigma_rel == 0. and not value.recipe.standardize
    assert value.recipe.row_policy == "routed_paired" and value.policy.row_semantics == "conditional"
    assert value.policy.row_evidence is value.policy.routed_control.evidence
    assert value.policy.birth_death is value.policy.routed_control
    assert value.policy.recipe.particle_birth_death and value.policy.recipe.row_evidence_gate
    assert value.policy.surprise is not None and value.policy.reopen_guard is not None
    assert value.policy.log_output_sigma is not None
    assert value.G.slot.shape == (host.N_SLOTS, host.DIM) and value.fit_context.shape[0] == host.N_ROWS
    assert torch.equal(value.G.neu, host.NEU) and value.hold_weight == host.HOLD_WEIGHT
    assert value.pairs == host.hold_pairs("matched")
    assert value.policy.roles == [["generator", "table", "noise"], ["critic"]]
    # Both slot-bank controls retain actual ownership; no 12-atom demo cloud.
    assert value.opt_g.param_groups[1]["lr"] == value.recipe.lr * value.recipe.prior_lr_mult
    assert value.opt_g.latent_damping is not None and value.opt_g.direct_response is None
    assert value.recipe.direct_particle_gain
    controls = value.receipt()["policy_lifecycle"]["controls"]
    assert controls["cohort"] == contract.COHORT and controls["family"] == contract.FAMILY
    assert controls["row_semantics"] == "conditional" and not controls["independent_atlas_qualification"]
    direct = value.guards()["mechanism_audit"]["mechanisms"]["direct_particle_gain"]
    assert direct["requested"] and not direct["enabled"] and not direct["applicable_to_routed_table"]
    assert not direct["host_activation_credit"]
    assert value.initialization["generator"]["initializer"] == "deterministic_orthogonal_named_parameters_v1"
    assert torch.count_nonzero(value.G.slot) == torch.count_nonzero(value.G.shared) == 0
    assert init.declarations(host.SharedSlotStudent()) == original_declarations
    assert init.declarations(value.G) == {"shared": init.KEEP, "slot": init.KEEP}
    assert typed_state_digest(value.G.state_dict()) == typed_state_digest(host.SharedSlotStudent().state_dict())
    if any(spec is None for spec in original_declarations.values()):
        with pytest.raises(ValueError, match="has no declaration"):
            init.deterministic_orthogonal_(host.SharedSlotStudent())


def test_clean_original_function_and_full_objective_gradients_match_at_uniform_mass():
    value = fixture()
    before = typed_state_digest(value.state_dict())
    for scale in (0., 1., -.5):
        context = slot_contexts([host.UNUSED, host.CONCEPT], scale=scale)
        routed = value.policy.routed_generate(context, sigma=0., perturb=False)
        torch.testing.assert_close(routed, value.G.embeds(scale), rtol=0, atol=0)
    original = value.G.embeds(1.)
    routed = value.policy.routed_generate(value.train_hold_context, sigma=0., perturb=False)
    real_logits = value.D(value.real).detach()
    def objective(embeds):
        fake = embeds[host.CONCEPT].unsqueeze(0).expand(host.N_ROWS, -1)
        return (value.loss.g_loss(value.D(fake), real_logits)
                + host.HOLD_WEIGHT * host.unused_hold_loss(embeds, value.G.neu, host.hold_pairs("matched")))
    original_loss, routed_loss = objective(original), objective(routed)
    torch.testing.assert_close(original_loss, routed_loss, rtol=0, atol=0)
    original_grad = torch.autograd.grad(original_loss, (value.G.shared, value.G.slot))
    routed_grad = torch.autograd.grad(routed_loss, (value.G.shared, value.G.slot))
    for a, b in zip(original_grad, routed_grad):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert typed_state_digest(value.state_dict()) == before


def test_three_real_ordered_updates_and_observer_noninterference():
    request, task = definition()
    plain = UnusedTokenRoutedFixture(request, task)
    observed = UnusedTokenRoutedFixture(request, task)
    for _ in range(3):
        a, b = plain.step(), observed.step()
        assert a == b
        before = typed_state_digest(evaluation_state(observed.state_dict()))
        metrics = observed.observe()
        assert metrics.keys() == host.score_student(observed.G).keys()
        assert before == typed_state_digest(evaluation_state(observed.state_dict()))
    assert typed_state_digest(plain.state_dict()) == typed_state_digest(observed.state_dict())
    assert observed.guards()["optimizer_updates"] == {"generator": 3, "table": 3, "discriminator": 3}
    assert observed.guards()["hooks_exercised"] and observed.guards()["all_finite"]
    diagnostics = observed.policy.routed_control.diagnostics()
    assert diagnostics["rows"]["law"] == "conditional_paired_diagnostic"
    assert diagnostics["rows"]["counters"]["updates"] == 3
    assert diagnostics["counters"]["evals"] == 3 and diagnostics["counters"]["probes"] == 6
    assert observed.policy._feature_selection.state["actual_backend"] == "routed"
    assert observed.policy._feature_selection.state["selection_reason"] == "routed_rows_owns_controls"
    assert observed.policy_observations[-1]["sampler"] == "served_snapshot"
    assert observed.policy_observations[-1]["backend"] == "routed"
    assert observed.policy_observations[-1]["latent_policy"] == "not_applied_to_parameter_measurement"
    assert all(row["pure"] for row in observed.purity)


def test_separate_fit_and_guard_reservoirs_preserve_protected_unused_context():
    value = fixture(); value.step(); value.step()
    control = value.policy.routed_control
    assert set(control.fit_context[:control.fit_fill, 0].tolist()) == {float(host.CONCEPT)}
    assert set(control.guard_context[:control.guard_fill, 0].tolist()) == {float(host.UNUSED)}
    assert torch.equal(control.guard_targets[:control.guard_fill], host.NEU[host.UNUSED].expand(control.guard_fill, -1))
    assert control.spec.output_error_guard and control.spec.max_context_harm == 0.
    assert control.spec.max_output_error_increase == control.spec.max_output_context_harm == 0.
    assert value.guard_context is not value.train_hold_context
    # Guards are a known anchor protection contract, not unseen data evidence.
    scope = value.task["execution"]["policy_contract"]["paired_controls"]
    assert scope["known_unused_anchor_is_also_original_training_loss"] and not scope["heldout_generalization_claim"]


def test_mass_aware_retirement_and_split_replay_rejects_harmful_unused_counterfactual():
    value = fixture()
    with torch.no_grad():
        value.G.shared.zero_(); value.G.slot.copy_(torch.tensor([[0., 0.], [0., 1.]]))
        value.policy.ema_G.load_state_dict(value.G.state_dict())
    control = value.policy.routed_control
    base = control.candidate(copy=True)
    before = typed_state_digest(value.state_dict())
    baseline = control.spec.forward(control.models, value.guard_context, base)
    # A real public candidate retires UNUSED and splits CONCEPT. Routing reads
    # the cloned slot IDs and additive log mass instead of captured live rows.
    proposed = control._split(base, host.UNUSED, host.CONCEPT, torch.zeros(host.DIM))
    assert torch.equal(proposed.row_state["slot_ids"], torch.ones(2, 1, dtype=torch.long))
    assert torch.allclose(proposed.log_mass, torch.full((2,), -math_log_two()))
    altered = control.spec.forward(control.models, value.guard_context, proposed)
    torch.testing.assert_close(baseline, value.guard_targets, rtol=0, atol=0)
    assert (altered - value.guard_targets).square().mean() == .5
    feature_before, output_before = control._measure(value.guard_context, value.guard_targets, base, with_output_error=True)
    feature_after, output_after = control._measure(value.guard_context, value.guard_targets, proposed, with_output_error=True)
    assert float((feature_after - feature_before).max().detach()) > control.spec.max_context_harm
    assert float((output_after - output_before).max().detach()) > control.spec.max_output_context_harm
    # Candidate evaluation/protection makes no bank, moment, RNG or clock write.
    assert typed_state_digest(value.state_dict()) == before


def math_log_two():
    import math
    return math.log(2.)


def test_actual_delete_law_reroutes_entire_context_and_preserves_other_slot():
    value = fixture()
    with torch.no_grad():
        value.G.slot.copy_(torch.tensor([[.1, -.2], [.3, .4]]))
    control = value.policy.routed_control
    base = control.candidate(copy=True)
    delete_unused = control._delete(base, host.UNUSED)
    assert torch.isneginf(delete_unused.log_mass[host.UNUSED])
    normal = control.spec.forward(control.models, value.train_hold_context, base)
    deleted = control.spec.forward(control.models, value.train_hold_context, delete_unused)
    assert torch.equal(normal[host.CONCEPT], deleted[host.CONCEPT])
    assert not torch.equal(normal[host.UNUSED], deleted[host.UNUSED])


def test_complete_restore_replays_exact_next_update_and_preserves_input():
    request, task = definition()
    original = UnusedTokenRoutedFixture(request, task); original.step(); original.step()
    saved = original.state_dict(); before = typed_state_digest(saved)
    resumed = UnusedTokenRoutedFixture(request, task); resumed.load_state_dict(saved)
    assert typed_state_digest(resumed.state_dict()) == before
    a, b = original.step(), resumed.step()
    assert a == b and typed_state_digest(original.state_dict()) == typed_state_digest(resumed.state_dict())
    assert original.guards() == resumed.guards()
    assert typed_state_digest(saved) == before
    assert original.observe() == resumed.observe()
    assert all(torch.equal(original.last_views[key], resumed.last_views[key]) for key in original.last_views)


@pytest.mark.parametrize("field", ["recipe", "family", "clock", "streams", "parameter", "audit", "router", "mechanisms", "last_update"])
def test_bad_checkpoint_cannot_mutate_the_public_owner(field):
    request, task = definition()
    source = UnusedTokenRoutedFixture(request, task); source.step()
    target = UnusedTokenRoutedFixture(request, task)
    before = typed_state_digest(target.state_dict())
    state = source.state_dict()
    if field == "recipe":
        state["recipe"]["row_policy"] = "independent"
    elif field == "family":
        state["family"] = "atlas"
    elif field == "clock":
        state["caller_cursor"] = 0
    elif field == "streams":
        state["policy"]["streams"]["noise_generator"] = state["policy"]["streams"]["eval_generator"].clone()
    elif field == "parameter":
        state["policy"]["models"]["generator"]["shared"][0] = float("nan")
    elif field == "audit":
        state["lifecycle_audit"]["calls"]["finish_step"] = 0
    elif field == "router":
        state["policy"]["models"]["router"]["slot_ids"] = torch.ones(3, 1, dtype=torch.long)
    elif field == "mechanisms":
        state["mechanism_audit_state"]["a2"]["calls"] = 5
    else:
        state["last_update"]["loss_g"] = float("inf")
    with pytest.raises(ValueError):
        target.load_state_dict(state)
    assert typed_state_digest(target.state_dict()) == before


@pytest.mark.parametrize("field", ["gate", "budget", "shared_owner", "row_policy", "family", "sources", "parent"])
def test_changed_scientific_contract_or_hidden_family_swap_is_rejected(field):
    _, task = definition()
    parent = json.loads((ROOT / "configs/forge/tasks/unused_token_hold.json").read_bytes())
    if field == "gate":
        task["evaluation"]["thresholds"][0][2] = 0.
    elif field == "budget":
        task["execution"]["steps"] = 3
    elif field == "shared_owner":
        task["execution"]["policy_contract"]["shared_owner"] = "independent_prior"
    elif field == "row_policy":
        task["execution"]["policy_recipe_overrides"]["row_policy"] = "independent"
    elif field == "family":
        task["policy_family"] = "atlas"
    elif field == "sources":
        task["execution"]["policy_contract"]["sources"].pop(contract.HOST_SOURCE)
    else:
        task["policy_parent"]["execution_fingerprint"] = "a" * 64
    with pytest.raises(ValueError):
        contract.validate_routed_task(task, parent=parent)


def test_original_parent_bytes_and_gates_stay_unchanged_and_new_law_never_aliases():
    path = ROOT / "configs/forge/tasks/unused_token_hold.json"
    before = path.read_bytes()
    request, task = definition()
    contract.validate_routed_task(task, root=ROOT)
    parent = json.loads(before)
    assert task["policy_parent"]["task_sha256"] == hashlib.sha256(before).hexdigest()
    assert task["id"] == contract.TASK_ID and task["task_cohort"] != "policy_selected_cloud_v1"
    assert task["evaluation"]["thresholds"] == parent["evaluation"]["thresholds"]
    assert task["execution"]["host_definition"] == parent["execution"]["host_definition"]
    assert not task["execution"]["policy_contract"]["independent_atom_bh_or_feature_cell_claim"]
    assert not contract.routed_policy_contract_blockers(task, contract.resolved_recipe(request["candidate"], task))
    assert path.read_bytes() == before


def test_old_independent_recipe_and_changed_shared_tuple_are_not_supported():
    from particlegan import get_recipe
    request, task = definition()
    assert contract.routed_policy_contract_blockers(task, get_recipe("atlas", **contract.SHARED_OVERRIDES,
                                                                 **contract.HOST_RESOURCES))
    request["candidate"]["recipe_overrides"]["lr"] *= 2
    with pytest.raises(ValueError, match="C6 shared"):
        contract.resolved_recipe(request["candidate"], task)


def test_public_routed_transport_rejects_independent_direct_particle_history():
    value = fixture()
    wrong_owner = value.recipe.make_generator_optimizer(
        [{"params": [value.G.shared]}, {"params": [value.G.slot]}],
        latent_table=value.G.slot, direct_particles=[value.G.slot])
    before = typed_state_digest(value.state_dict())
    with pytest.raises(ValueError, match="does not support direct-particle response"):
        value.policy.routed_control._adam_transport_state(wrong_owner, value.G.slot)
    assert typed_state_digest(value.state_dict()) == before


@pytest.mark.parametrize("field,value", [("output_noise", True), ("weight_selector", "fast"),
                                        ("latent_policy", "actual_selected_public_policy")])
def test_parameter_observation_cannot_claim_sampling_or_unselected_weights(field, value):
    _, task = definition()
    assert contract.validate_routed_observation(task)["sampler"] == "served_snapshot"
    task["evaluation"]["policy_observation"][field] = value
    with pytest.raises(ValueError, match="numerical bounds"):
        contract.validate_routed_observation(task)
