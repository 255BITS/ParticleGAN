"""CPU structural context/ownership controls; no full numerical acquisition."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.forge import policy_cohorts as registry
from experiments.forge.api import CapabilityError, FormulationContext, task_formulation_context, task_policy_blockers
from experiments.forge.boundaries import ownership_receipt, prior_control_binding
from experiments.forge.policy_adapters import typed_state_digest


ROOT = Path(__file__).resolve().parents[1]
ROUTED_NAMES = ("trajectory", "residual_student", "unipolar", "mid_scale_identity", "unused", "cover", "ae")
NAMES = ROUTED_NAMES + ("word",)


@pytest.fixture(autouse=True)
def cpu_structural_scope(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    threads, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def definition(name):
    if name in NAMES[:4]:
        from experiments.forge import conditional_policy_contracts as module
        task = module.make_conditional_variant(ROOT, name)
    elif name == "unused":
        from experiments.forge import routed_policy_contracts as module
        task = module.make_unused_variant(ROOT)
    elif name == "cover":
        from experiments.forge import multibank_policy_contracts as module
        task = module.make_variant(ROOT)
    elif name == "ae":
        from experiments.forge import ae_routed_policy_contracts as module
        task = module.make_ae_variant(ROOT)
    elif name == "word":
        from experiments.forge import word_joint_policy_contracts as module
        task = module.make_variant(ROOT)
    else:
        from experiments.forge import policy_contracts as module
        raw = (ROOT / ("configs/forge/tasks/" + name + ".json")).read_bytes()
        parent = json.loads(raw)
        source_paths = module.REQUIRED_POLICY_SOURCES | set(parent["evaluation"].get("sources", {}))
        if name == "two_pole":
            source_paths |= {"benchmarks/locked_shared/two_pole.py", "benchmarks/legacy/locked_shared.py"}
        sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}
        task = module._prospective_variant(parent, module._parent_record(parent, hashlib.sha256(raw).hexdigest()), sources)
    request = {"candidate": {"id": "software_only_named_context", "recipe_preset": "atlas",
        "task_cohort": module.COHORT, "recipe_overrides": {"lr": .0053125, "prior_lr_mult": 1.5}}, "protocol": {"seed": 0}}
    return request, task


def fixture(name, request, task, context):
    if name in NAMES[:4]:
        from experiments.forge.conditional_policy_adapters import ConditionalPolicyFixture
        return ConditionalPolicyFixture(request, task, context=context)
    if name == "unused":
        from experiments.forge.routed_policy_adapters import UnusedTokenRoutedFixture
        return UnusedTokenRoutedFixture(request, task, context=context)
    if name == "cover":
        from experiments.forge.multibank_policy_adapters import CoverMultibankFixture
        return CoverMultibankFixture(request, task, context=context)
    if name == "word":
        from experiments.forge.word_joint_policy_adapters import WordJointPolicyFixture
        return WordJointPolicyFixture(request, task, context=context)
    from experiments.forge.ae_routed_policy_adapters import AERoutedFixture
    return AERoutedFixture(request, task, context=context)


def context_for(request, task):
    return task_formulation_context(request["candidate"], task, request["protocol"], device="cpu", root=ROOT)


@pytest.mark.parametrize("name", NAMES + ("two_pole", "img_intensity2"))
def test_exact_cohort_dispatch_and_original_bounds_without_model_construction(name):
    request, task = definition(name)
    before = torch.get_rng_state().clone()
    contract = registry.validate_policy_task(task, root=ROOT)
    assert contract["cohort"] in registry.KNOWN_COHORTS
    assert registry.module_for_task(task).COHORT == contract["cohort"]
    assert registry.validate_policy_observation(task) == task["evaluation"]["policy_observation"]
    context = context_for(request, task)
    assert not task_policy_blockers(task, request["candidate"])
    assert context.recipe.total_steps is None
    assert not registry.policy_contract_blockers(task, context.recipe)
    assert context.policy is None
    assert torch.equal(before, torch.get_rng_state())
    raw = (ROOT / ("configs/forge/tasks/" + task["policy_parent"]["id"] + ".json")).read_bytes()
    parent = json.loads(raw)
    assert task["execution"]["steps"] == parent["execution"]["steps"]
    assert task["evaluation"]["thresholds"] == parent["evaluation"]["thresholds"]
    assert task["evaluation"]["observations"] == parent["evaluation"]["observations"]
    assert task["resources"]["device"] == "cuda" and task["resources"].get("allow_cpu", False) is False
    caps = context.capabilities()
    assert caps["routed_rows"] is (name in ROUTED_NAMES)
    assert caps["mog_encoder"] is (name == "ae")
    assert caps["ae_encoder"] is (name == "ae")
    assert caps["uniform_masses"] is (name not in ROUTED_NAMES)
    if name == "word": assert caps["public_trainer"] is False
    assert registry.policy_recipe_overrides(task) == task["execution"]["policy_recipe_overrides"]


@pytest.mark.parametrize("name", NAMES)
def test_actual_two_update_context_injection_checkpoint_and_selected_ownership(name):
    request, task = definition(name)
    context = context_for(request, task)
    value = fixture(name, request, task, context)
    assert context.policy is value.policy
    assert context.policy._table_location == tuple(task["execution"]["policy_contract"]["table_owner"].split("."))
    value.step()
    checkpoint = deepcopy(value.state_dict())
    context_checkpoint = deepcopy(context.state_dict())
    assert context_checkpoint["external_max_steps"] == task["execution"]["steps"]
    lifecycle = context.receipt()["policy_lifecycle"]
    assert lifecycle["completed_steps"] == 1
    assert lifecycle["quality_qualification"] is False
    controls = lifecycle["controls"]
    assert controls["cohort"] == task["task_cohort"]
    assert controls["diagnostics"]["row_evidence"]["updates"] == 1
    keys = {("d" if index == 1 else "g") + str(group)
            for index, optimizer in enumerate(value.policy.optimizers)
            for group in range(len(optimizer.param_groups))}
    assert set(controls["diagnostics"]["stationarity_lr"]) == keys
    value.step()
    expected = typed_state_digest(value.state_dict())
    next_context = context_for(request, task)
    resumed = fixture(name, request, task, next_context)
    resumed.load_state_dict(checkpoint)
    next_context.load_state_dict(context_checkpoint)
    assert typed_state_digest(next_context.state_dict()) == typed_state_digest(context_checkpoint)
    resumed.step()
    assert typed_state_digest(resumed.state_dict()) == expected
    with pytest.raises(ValueError, match="one public update"):
        next_context.bind_policy(resumed.policy, external_max_steps=task["execution"]["steps"])


@pytest.mark.parametrize("name", NAMES)
def test_named_task_cannot_change_parent_gates_or_lose_explicit_candidate_opt_in(name):
    request, task = definition(name)
    forged = deepcopy(task)
    forged["evaluation"]["thresholds"][0][2] += 1
    with pytest.raises(ValueError): registry.validate_policy_task(forged, root=ROOT)
    request["candidate"].pop("task_cohort")
    assert task_policy_blockers(task, request["candidate"])
    with pytest.raises(CapabilityError): context_for(request, task)


@pytest.mark.parametrize("change", ["unknown", "word", "missing", "none", "family", "cross_cohort"])
def test_any_policy_claim_requires_known_exact_family_and_contract(change):
    request, task = definition("ae")
    if change == "unknown": task["task_cohort"] = task["execution"]["policy_contract"]["cohort"] = "unknown_policy_v1"
    elif change == "word": task["task_cohort"] = task["execution"]["policy_contract"]["cohort"] = "word_joint_policy_v1"
    elif change == "missing": task.pop("task_cohort")
    elif change == "none": task["execution"]["policy_contract"] = None
    elif change == "family": task["policy_family"] = "atlas"
    else: task["task_cohort"] = "routed_policy_selected_cloud_v1"
    assert registry.is_policy_task(task)
    with pytest.raises(ValueError): registry.validate_policy_task(task)
    # A nonadaptive Recipe cannot hide a malformed policy declaration.
    request["candidate"] = {"recipe_preset": "ka2", "recipe_overrides": {}}
    assert task_policy_blockers(task, request["candidate"])
    with pytest.raises(CapabilityError): context_for(request, task)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("change", ["adapter", "capabilities", "runtime", "evaluator", "parent"])
def test_frozen_grading_still_checks_full_parent_transform_without_live_source_repin(name, change):
    _, task = definition(name)
    if change == "adapter": task["adapter"] = "foreign_adapter"
    elif change == "capabilities": task["requires_capabilities"] = []
    elif change == "runtime": task["resources"]["timeout_seconds"] += 1
    elif change == "evaluator": task["evaluation"]["sources"] = {}
    else: task["policy_parent"]["execution_fingerprint"] = "0" * 64
    with pytest.raises(ValueError): registry.validate_policy_task(task)


@pytest.mark.parametrize("name", NAMES)
def test_frozen_grading_does_not_substitute_current_implementation_hashes(name):
    _, task = definition(name)
    # Synthetic older snapshot identity. It supplies no real scientific credit.
    task["execution"]["policy_contract"]["sources"]["particlegan/policy.py"] = "0" * 64
    assert registry.validate_policy_task(task)["sources"]["particlegan/policy.py"] == "0" * 64
    with pytest.raises(ValueError): registry.validate_policy_task(task, root=ROOT)


@pytest.mark.parametrize("name", ["unused", "unipolar", "mid_scale_identity", "cover", "ae", "word"])
def test_table_vs_prior_and_no_independent_direct_response_ownership_are_honest(name):
    request, task = definition(name)
    context = context_for(request, task)
    binding = prior_control_binding(task)
    assert binding["latent_table_controls"] is True
    assert binding["independent_direct_particle_response_owner"] is False
    assert binding["constructed_prior"] is (name in {"cover", "ae", "word"})
    receipt = ownership_receipt(request["candidate"], task, asdict(context.recipe))
    assert receipt["recipe_fields"]["prior_lr_mult"]["status"] == "effective"
    assert receipt["recipe_fields"]["row_policy"]["owner"] == "task"
    for key in ("num_particles", "z_dim", "batch_size"):
        assert receipt["recipe_fields"][key]["status"] == "effective"
    code_path = receipt["task_contract"]["prior"]["code_path"]
    assert code_path == ("MoGParticlePrior" if name == "ae" else "ParticlePrior" if name in {"cover", "word"} else None)


@pytest.mark.parametrize("change", ["callback", "sites", "owner", "disabled", "width", "budget", "recipe", "borrowed_owner"])
def test_actual_owner_binding_refuses_fake_or_incompatible_policy_before_acquiring_it(change):
    request, task = definition("ae")
    value = fixture("ae", request, task, context_for(request, task))
    context = context_for(request, task)
    p = value.policy
    budget = task["execution"]["steps"]
    if change == "callback": p.routed_control.spec.model_forward = lambda models, ctx, candidate, routing: models["generator"](ctx[:, :2])
    elif change == "sites": p.routed_control.spec.sites = ("foreign_site",)
    elif change == "owner": p._table_location = ("generator", "weight")
    elif change == "disabled": p.birth_death = None
    elif change == "width": p.prior.set_sigma(.03)
    elif change == "budget": budget += 1
    elif change == "borrowed_owner":
        other = fixture("ae", request, task, context_for(request, task)).policy
        p.routed_control = other.routed_control
        p.birth_death = other.routed_control
        p.row_evidence = other.routed_control.evidence
    else: p.recipe = p.recipe.replace(lr=p.recipe.lr * .5)
    with pytest.raises((CapabilityError, ValueError)):
        context.bind_policy(p, external_max_steps=budget)
    assert context.policy is None


def test_direct_declared_context_cannot_omit_the_task_owned_external_budget():
    request, task = definition("ae")
    bound = context_for(request, task)
    value = fixture("ae", request, task, bound)
    direct = FormulationContext(recipe_overrides=asdict(bound.recipe), prior=bound.prior_config,
        policy_task=task, policy_contract=task["execution"]["policy_contract"],
        policy_root=ROOT, execution_path="public_components")
    with pytest.raises(CapabilityError, match="frozen task budget"):
        direct.bind_policy(value.policy, external_max_steps=task["execution"]["steps"] + 1)
    assert direct.policy is None


def test_named_task_cannot_bind_an_undeclared_autocast_execution():
    request, task = definition("ae")
    value = fixture("ae", request, task, context_for(request, task))
    context = context_for(request, task)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(CapabilityError, match="FP32/no-autocast"):
        context.bind_policy(value.policy, external_max_steps=task["execution"]["steps"])
    assert context.policy is None


@pytest.mark.parametrize("change", ["generation", "split_code", "encoder", "foreign_encoder", "frozen_encoder", "disabled", "optimizer", "shape", "features"])
def test_word_joint_context_binds_complete_atoms_free_encoder_and_eleven_real_rows(change):
    request, task = definition("word")
    value = fixture("word", request, task, context_for(request, task))
    p = value.policy
    if change == "generation": p.generation = None
    elif change == "split_code": p.generation = lambda model, code: torch.cat((model(code).flatten(1), code), 1)
    elif change == "encoder": p.encoder = None
    elif change == "foreign_encoder":
        from benchmarks.toy_audit.api_images import WordEncoder
        p.encoder = WordEncoder()
    elif change == "frozen_encoder": p.encoder.requires_grad_(False)
    elif change == "disabled": p.birth_death = None
    elif change == "features": p.birth_death.rows.critic_features = lambda values: values
    elif change == "optimizer":
        group = next(group for group, role in zip(p.opt_g.param_groups, p.roles[0]) if role == "encoder")
        group["params"] = []
    else: p.table = p.table[:5]
    context = context_for(request, task)
    with pytest.raises(CapabilityError): context.bind_policy(p, external_max_steps=task["execution"]["steps"])
    assert context.policy is None


@pytest.mark.parametrize("change", ["cohort", "budget", "streams", "metadata", "missing_policy"])
def test_component_context_checkpoint_mismatch_is_atomic(change):
    request, task = definition("ae")
    context = context_for(request, task)
    value = fixture("ae", request, task, context)
    value.step()
    before = context.state_dict()
    bad = deepcopy(before)
    if change == "cohort": bad["policy_contract"]["cohort"] = "policy_selected_cloud_v1"
    elif change == "budget": bad["external_max_steps"] += 1
    elif change == "streams": bad["policy"]["streams"]["noise_generator"] = torch.Generator().manual_seed(987).get_state()
    elif change == "missing_policy": bad["policy"] = None
    else: bad["policy"]["models"]["prior"]["_extra_state"]["standardize"] = True
    with pytest.raises(ValueError): context.load_state_dict(bad)
    assert typed_state_digest(context.state_dict()) == typed_state_digest(before)


def test_original_independent_mog_and_undeclared_routed_contexts_remain_refused():
    cloud = {"kind": "particle_cloud", "sigma": 0., "standardize": False, "learnable": True,
             "exception_reason": "CPU structural software control"}
    with pytest.raises(CapabilityError, match="RoutedRows"):
        FormulationContext(recipe_preset="atlas", recipe_overrides={"row_policy": "routed_paired"}, prior=cloud)
    with pytest.raises(CapabilityError):
        FormulationContext(recipe_preset="atlas", prior={"kind": "mog", "sigma": .025, "standardize": False, "learnable": True})
    assert "word_joint_policy_v1" not in registry.KNOWN_COHORTS
    assert isinstance(registry.KNOWN_COHORTS, frozenset)
    assert not torch.cuda.is_initialized()
