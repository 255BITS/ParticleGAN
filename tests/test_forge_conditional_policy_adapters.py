"""CPU structural controls; short prefixes provide no convergence credit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from particlegan import RoutedBatch, UpdatePolicy, init
from benchmarks.locked_shared import trajectory
from benchmarks.locked_shared.hosts import residual_student, unipolar, mid_scale_identity
from experiments.forge import conditional_policy_contracts as contract
from experiments.forge.conditional_policy_adapters import (
    ConditionalPolicyFixture, RoutedUnipolarResidual, RoutedMidScaleResidual, run_behavior)
from experiments.forge.policy_adapters import evaluation_state, typed_state_digest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_one_thread(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def definition(host="trajectory"):
    task = contract.make_conditional_variant(ROOT, host)
    request = {"candidate": {"id": "software_only_conditional_prefix", "task_cohort": contract.COHORT,
                              "recipe_preset": "atlas", "recipe_overrides": deepcopy(contract.SHARED_OVERRIDES)},
               "protocol": {"seed": 0}}
    return request, task


def fixture(host="trajectory"):
    request, task = definition(host)
    return ConditionalPolicyFixture(request, task)


@pytest.mark.parametrize("host", contract.HOSTS)
def test_original_parent_gates_budget_source_and_gpu_transform_are_exact(host):
    request, task = definition(host)
    raw = (ROOT / f"configs/forge/tasks/{host}.json").read_bytes(); parent = json.loads(raw)
    assert task["policy_parent"]["task_sha256"] == hashlib.sha256(raw).hexdigest()
    assert task["evaluation"]["thresholds"] == parent["evaluation"]["thresholds"]
    for field in ("kind", "evaluator", "observations", "minimum_stable_checks", "sampling_law", "eval_output_noise"):
        assert task["evaluation"][field] == parent["evaluation"][field]
    assert task["execution"]["steps"] == parent["execution"]["steps"]
    assert task["execution"]["initializer"] == parent["execution"]["initializer"]
    assert task["execution"]["host_definition"] == parent["execution"]["host_definition"]
    assert task["execution"]["prior"] == parent["execution"]["prior"]
    assert parent["execution"]["device"] == "cpu" and task["execution"]["device"] == "cuda"
    assert task["resources"]["device"] == "cuda" and not task["resources"]["allow_cpu"]
    assert contract.validate_conditional_task(task, root=ROOT)
    resolved = contract.resolved_recipe(request["candidate"], task)
    assert resolved.total_steps is None and resolved.row_policy == "routed_paired"
    assert not contract.conditional_policy_contract_blockers(task, resolved)
    assert task["execution"]["policy_contract"]["paired_controls"]["all_original_training_contexts_preserved"]
    assert not task["execution"]["policy_contract"]["paired_controls"]["heldout_generalization_claim"]


@pytest.mark.parametrize("field", ("gate", "budget", "source", "initializer", "parent", "cohort", "resources", "routing", "noise"))
def test_unknown_or_changed_conditional_contract_is_rejected(field):
    _, task = definition()
    if field == "gate": task["evaluation"]["thresholds"][0][2] = .5
    elif field == "budget": task["execution"]["steps"] = 2
    elif field == "source": task["execution"]["policy_contract"]["sources"]["particlegan/routing.py"] = "0" * 64
    elif field == "initializer": task["execution"]["initializer"] = "constructor_random"
    elif field == "parent": task["policy_parent"]["task_sha256"] = "0" * 64
    elif field == "cohort": task["task_cohort"] = "policy_selected_cloud_v1"
    elif field == "resources": task["execution"]["resources"]["num_particles"] = 256
    elif field == "routing": task["execution"]["policy_contract"]["routing"]["sites"] = []
    else: task["evaluation"]["policy_observation"]["output_noise"] = False
    with pytest.raises(ValueError): contract.validate_conditional_task(task, root=ROOT)


@pytest.mark.parametrize("host", ("trajectory", "residual_student"))
def test_original_arc_architecture_initializer_function_and_objective_gradients_match(host):
    value = fixture(host)
    assert isinstance(value.G, trajectory._Generator if host == "trajectory" else residual_student.ResidualHead)
    module = trajectory if host == "trajectory" else residual_student
    assert isinstance(value.D, module._Critic)
    assert value.table is value.prior.z and value.table.shape == (12, 4)
    assert value.slow.shape == value.real.shape == (12, 16)
    assert torch.equal(value.train_context[:, 0], torch.arange(12, dtype=torch.float32))
    assert set(value.fit_context[:, 0].tolist()) == set(range(0, 12, 2))
    assert set(value.guard_context[:, 0].tolist()) == set(range(1, 12, 2))
    original = value.G(value.slow, value.table)
    routed = value.policy.routed_generate(value.train_context, sigma=0., perturb=False)
    torch.testing.assert_close(routed, original, rtol=0, atol=0)
    original_loss = value.loss.g_loss(value.D(value.slow, original), value.D(value.slow, value.real).detach())
    original_loss = original_loss + 1.5 * module._cover(original, value.real) + .02 * value.table.square().mean() + value.spread(value.table)
    if host == "residual_student":
        assert torch.equal(value.mask, torch.ones(12, dtype=torch.bool))
        original_loss = original_loss + module.RESIDUAL_WEIGHT * (original[value.mask] - value.real[value.mask]).square().mean()
    routed_loss = value.loss.g_loss(value.D(value.slow, routed), value.D(value.slow, value.real).detach())
    routed_loss = routed_loss + sum(value.clean_original_auxiliary(routed).values())
    parameters = [*value.G.parameters(), value.table]
    a = torch.autograd.grad(original_loss, parameters); b = torch.autograd.grad(routed_loss, parameters)
    torch.testing.assert_close(original_loss, routed_loss, rtol=0, atol=0)
    for one, two in zip(a, b): torch.testing.assert_close(one, two, rtol=0, atol=0)


@pytest.mark.parametrize("host,original_cls,new_cls", (("unipolar", unipolar.FreeOriginResidual, RoutedUnipolarResidual),
    ("mid_scale_identity", mid_scale_identity.MidScaleResidual, RoutedMidScaleResidual)))
def test_basis_storage_keeps_original_formula_degrees_initializer_and_all_gradients(host, original_cls, new_cls):
    declarations_before = init.declarations(original_cls())
    value = fixture(host)
    assert isinstance(value.G, original_cls) and isinstance(value.G, new_cls)
    assert set(dict(value.G.named_parameters())) == {"bank"}
    assert sum(p.numel() for p in original_cls().parameters()) == value.table.numel()
    assert init.declarations(value.G)["bank"] is init.KEEP
    assert init.declarations(original_cls()) == declarations_before
    assert torch.count_nonzero(value.table) == 0
    original = original_cls()
    fixed = torch.arange(value.table.numel(), dtype=torch.float32).reshape_as(value.table) / 19
    with torch.no_grad():
        value.table.copy_(fixed)
        for index, name in enumerate(value.G.BASIS_ROLES): getattr(original, name).copy_(fixed[index])
    for scale in (-1., -.5, 0., .25, .5, 1.):
        context = torch.tensor([[scale], [scale]])
        routed = value.policy.routed_generate(context, sigma=0., perturb=False)
        expected = original.delta(scale) if host == "unipolar" else original.state(scale)
        torch.testing.assert_close(routed, expected.expand(2, -1), rtol=0, atol=0)
        a = torch.autograd.grad(expected.square().sum(), list(original.parameters()), retain_graph=True)
        b, = torch.autograd.grad(routed.square().mean(0).sum(), [value.table])
        torch.testing.assert_close(torch.stack(a), b, rtol=0, atol=0)
    assert value.scales == (unipolar.SCALES if host == "unipolar" else mid_scale_identity.EVAL_SCALES)
    assert len(value.train_context) == 8 * len(value.scales)
    if host == "mid_scale_identity":
        clean = value.policy.routed_generate(torch.tensor([[s] for s in value.scales]), sigma=0., perturb=False)
        original_clean = torch.stack([original.state(s) for s in value.scales])
        original_cover = mid_scale_identity.FORMULATION["cover_weight"] * sum(
            (original.state(s) - value.targets[s]).square().mean() for s in value.scales) / len(value.scales)
        routed_cover = value.clean_original_auxiliary(clean)["cover_loss"]
        torch.testing.assert_close(clean, original_clean, rtol=0, atol=0)
        torch.testing.assert_close(original_cover, routed_cover, rtol=1e-6, atol=1e-7)
    # Full original generator objective, with equally weighted conditioned
    # critics and every original scale. No policy noise is drawn for this
    # software identity witness; prospective training noise remains declared.
    original_loss = fixed.new_zeros(()); routed_loss = fixed.new_zeros(())
    clean = value.policy.routed_generate(torch.tensor([[s] for s in value.scales]), sigma=0., perturb=False)
    for index, scale in enumerate(value.scales):
        real = value.targets[scale].expand(8, -1)
        point = original.delta(scale) if host == "unipolar" else original.state(scale)
        original_loss = original_loss + value.loss.g_loss(value.D(point.expand(8, -1), scale), value.D(real, scale).detach()) / len(value.scales)
        routed_loss = routed_loss + value.loss.g_loss(value.D(clean[index].expand(8, -1), scale), value.D(real, scale).detach()) / len(value.scales)
    if host == "mid_scale_identity":
        original_loss = original_loss + original_cover
        routed_loss = routed_loss + value.clean_original_auxiliary(clean)["cover_loss"]
    original_grad = torch.autograd.grad(original_loss, list(original.parameters()))
    routed_grad, = torch.autograd.grad(routed_loss, [value.table])
    torch.testing.assert_close(original_loss, routed_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(torch.stack(original_grad), routed_grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("host", contract.HOSTS)
def test_two_public_updates_and_observer_preserve_exact_next_training_state(host):
    plain, observed = fixture(host), fixture(host)
    for _ in range(2):
        assert plain.step() == observed.step()
        before = typed_state_digest(evaluation_state(observed.state_dict()))
        metrics = observed.observe()
        assert set(row[0] for row in observed.task["evaluation"]["thresholds"]) <= metrics.keys()
        assert before == typed_state_digest(evaluation_state(observed.state_dict()))
    assert typed_state_digest(plain.state_dict()) == typed_state_digest(observed.state_dict())
    assert observed.audit.receipt(2)["complete"] and all(v == 2 for v in observed.audit.calls.values())
    assert observed.guards()["optimizer_updates"] == {"generator": 2, "table": 2, "discriminator": 2}
    assert observed.guards()["all_finite"] and all(row["pure"] for row in observed.purity)
    assert observed.policy._feature_selection.state["actual_backend"] == "routed"
    assert observed.policy.routed_control.evidence.counters["updates"] == 2
    assert observed.receipt()["family"] == "atlas_conditional"
    assert not observed.receipt()["independent_atlas_qualification"]
    assert not observed.receipt()["policy_lifecycle"]["controls"]["quality_qualification"]
    assert observed.receipt()["execution_phase"] == "cpu_structural_only"


@pytest.mark.parametrize("host", contract.HOSTS)
def test_complete_checkpoint_restores_exact_next_update_and_selected_observation(host):
    original = fixture(host); original.step(); original.step()
    state = original.state_dict(); before = typed_state_digest(state)
    resumed = fixture(host); resumed.load_state_dict(state)
    assert typed_state_digest(resumed.state_dict()) == before
    assert original.step() == resumed.step()
    assert typed_state_digest(original.state_dict()) == typed_state_digest(resumed.state_dict())
    assert original.observe() == resumed.observe()
    assert all(torch.equal(original.last_views[key], resumed.last_views[key]) for key in original.last_views)
    assert typed_state_digest(state) == before


@pytest.mark.parametrize("host", contract.HOSTS)
def test_guard_tensors_do_not_enter_original_losses_or_gradients(host):
    ordinary, changed = fixture(host), fixture(host)
    changed.guard_targets.add_(5.)
    assert ordinary.step() == changed.step()
    one, two = ordinary.policy.state_dict(), changed.policy.state_dict()
    for key in ("models", "optimizers", "averaged_table"):
        assert typed_state_digest(one[key]) == typed_state_digest(two[key])


def test_public_conditional_ownership_rejects_independent_atoms_and_missing_context_guards():
    value = fixture()
    with pytest.raises(ValueError, match="independent unconditional"):
        UpdatePolicy(value.recipe.replace(row_policy="independent"), value.G, value.D,
            prior=value.prior, generator_optimizer=value.opt_g, critic_optimizer=value.opt_d,
            row_semantics="conditional")
    with pytest.raises(ValueError, match="requires independent"):
        value.policy.served_model().sample(12)
    with pytest.raises(ValueError, match="require RoutedBatch"):
        value.policy.begin_step(value.real, execution_limit=400)
    with pytest.raises(ValueError, match="must not overlap"):
        value.policy.begin_step(value.real, routed=RoutedBatch(value.fit_context, value.fit_targets,
            value.fit_context.clone(), value.fit_targets.clone()), execution_limit=400)


@pytest.mark.parametrize("host", contract.HOSTS)
def test_counterfactual_replays_cloned_roles_and_mass_without_changing_live_state(host):
    value = fixture(host)
    with torch.no_grad(): value.table.copy_(torch.arange(value.table.numel()).reshape_as(value.table) / 11)
    control = value.policy.routed_control; base = control.candidate(copy=True)
    before = typed_state_digest(value.state_dict())
    retired, split = (1, 0) if value.arcs else (2, 0)
    candidate = control._split(base, retired, split, torch.zeros(value.table.shape[1]))
    assert candidate.row_state["row_roles"][retired].item() == split
    assert candidate.log_mass[retired] == candidate.log_mass[split] == -math_log_two()
    baseline = control.spec.forward(control.models, value.guard_context, base)
    proposed = control.spec.forward(control.models, value.guard_context, candidate)
    assert not torch.equal(baseline, proposed)
    # Choose a structural guard oracle equal to the baseline. Any changed
    # complete function has positive output error and is protected. This is
    # a counterfactual software test, not an observed accepted/retired row.
    _, e0 = control._measure(value.guard_context, baseline.detach(), base, with_output_error=True)
    _, e1 = control._measure(value.guard_context, baseline.detach(), candidate, with_output_error=True)
    assert float((e1 - e0).max().detach()) > control.spec.max_output_context_harm
    assert typed_state_digest(value.state_dict()) == before


def math_log_two():
    import math
    return math.log(2.)


@pytest.mark.parametrize("kind", ("clock", "recipe", "lifecycle", "rng", "nan", "source", "mechanism"))
def test_checkpoint_drift_is_rejected_before_owner_mutation(kind):
    original = fixture(); original.step()
    state = original.state_dict(); target = fixture(); before = typed_state_digest(target.state_dict())
    if kind == "clock": state["caller_cursor"] = 0
    elif kind == "recipe": state["recipe"]["row_policy"] = "independent"
    elif kind == "lifecycle": state["lifecycle_audit"]["calls"]["finish_step"] = 0
    elif kind == "rng": state["policy"]["streams"]["noise_generator"] = state["policy"]["streams"]["eval_generator"].clone()
    elif kind == "nan": state["policy"]["models"]["generator"]["net.0.weight"][0, 0] = float("nan")
    elif kind == "source": state["task_sha256"] = "0" * 64
    else: state["mechanism_audit_state"]["a2"]["calls"] = 25
    with pytest.raises(ValueError): target.load_state_dict(state)
    assert typed_state_digest(target.state_dict()) == before


def test_cpu_prefix_cannot_be_published_as_numerical_acquisition(tmp_path):
    request, task = definition()
    with pytest.raises(ValueError, match="CPU fixtures are structural only"):
        run_behavior(request, task, tmp_path / "forbidden", device="cpu")
    assert not (tmp_path / "forbidden").exists()


def test_all_four_variants_are_explicit_and_original_parent_definitions_untouched():
    original = {name: (ROOT / f"configs/forge/tasks/{name}.json").read_bytes() for name in contract.HOSTS}
    variants = contract.load_conditional_variants(ROOT)
    assert set(variants) == {name + contract.SUFFIX for name in contract.HOSTS}
    assert not set(variants) & set(original)
    for task in variants.values():
        assert contract.validate_conditional_observation(task) == task["evaluation"]["policy_observation"]
        assert contract.conditional_recipe_overrides(task) == {"row_policy": "routed_paired"}
    assert original == {name: (ROOT / f"configs/forge/tasks/{name}.json").read_bytes() for name in contract.HOSTS}


@pytest.mark.parametrize("host", ("unipolar", "mid_scale_identity"))
def test_selected_original_scorer_distinguishes_hold_and_midscale_counterexamples(host):
    from benchmarks.locked_shared.baseline import score_metrics
    value = fixture(host)
    with torch.no_grad():
        value.table.zero_()
        if host == "unipolar": value.table[0].copy_(unipolar.PLUS)
        else:
            value.table[0].copy_(value.teacher.concept)
            value.table[2].copy_(value.teacher.identity)
    positive = value.observe()
    assert all(row["status"] == "PASS" for row in score_metrics(positive, value.task["evaluation"]["thresholds"]))
    with torch.no_grad():
        if host == "unipolar":
            value.table.zero_(); value.table[2].copy_(unipolar.PLUS)
        else: value.table[3].copy_(value.teacher.stranger - value.teacher.identity)
    negative = value.observe()
    if host == "unipolar":
        assert negative["cover"] == positive["cover"] == 1.
        assert negative["neu_hold"] == 0. and positive["neu_hold"] == 1.
    else:
        assert negative["concept_cos_plus"] == positive["concept_cos_plus"]
        assert negative["concept_cos_minus"] == positive["concept_cos_minus"]
        assert negative["identity_at_0"] == positive["identity_at_0"]
        assert negative["identity_at_mid"] < .85 <= positive["identity_at_mid"]
    assert any(row["status"] == "FAIL" for row in score_metrics(negative, value.task["evaluation"]["thresholds"]))
    assert value.completed_steps == 0 and value.receipt()["execution_phase"] == "cpu_structural_only"
