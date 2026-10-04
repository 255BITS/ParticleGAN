"""Model-free prospective rate/owner rejection controls; no scientific credit.

Fake metrics and checkpoint bytes exercise metadata and grading only. No
training model, forward, sampler, evaluator or optimizer is constructed here.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.forge import api, boundaries, evaluate, named_policy_planning, planning, policy_cohorts
from experiments.forge.contracts import stable_hash
from experiments.forge import sampling, views
from experiments.forge import policy_snapshot_publication, technique_board
from experiments.forge import word_joint_policy_adapters as producer
from experiments.forge import word_joint_policy_contracts as original
from experiments.forge import word_joint_rate_policy_contracts as rates

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def model_free(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    before, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    torch.set_num_threads(1)
    def forbidden(*args, **kwargs):
        raise AssertionError("real training model construction is forbidden in these controls")
    for name in ("WordGenerator", "WordEncoder", "WordJointCritic"):
        monkeypatch.setattr(producer, name, forbidden)
    yield
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(threads)


@pytest.fixture
def task():
    return rates.make_variant(ROOT)


def request(profile):
    return {"candidate": rates.candidate(profile), "protocol": {"seed": 0}}


def original_candidate():
    return {"task_cohort": original.COHORT, "recipe_preset": "atlas",
            "recipe_overrides": deepcopy(original.SHARED_OVERRIDES)}


def synthetic(task, profile, directory):
    path = ROOT / "tests/test_forge_named_policy_grading.py"
    spec = importlib.util.spec_from_file_location("word_rate_synthetic_guard_inputs", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    raw = helper.synthetic_receipt(task, directory)
    recipe = rates.resolved_recipe(rates.candidate(profile), task)
    raw["evidence"]["policy_controls"]["word_rate_binding"] = rates.binding_receipt(recipe, profile)
    raw["applied"] = {"family": rates.FAMILY, "task_cohort": rates.COHORT,
        "execution_path": "public_components", "actual_resources": deepcopy(rates.HOST_RESOURCES),
        "recipe": recipe.to_dict(), "policy_lifecycle": {"owner": "particlegan.UpdatePolicy",
            "controls": deepcopy(raw["evidence"]["policy_controls"])}}
    assert raw["software_fixture_only"] is True
    return raw


@pytest.mark.parametrize("profile", tuple(rates.RATE_PROFILES))
def test_complete_public_recipe_changes_only_declared_rates(profile, task):
    old_task = original.make_variant(ROOT)
    old = original.resolved_recipe(original_candidate(), old_task).to_dict()
    resolved = rates.resolved_recipe(rates.candidate(profile), task).to_dict()
    assert resolved == {**old, **rates.rates(profile)}
    assert resolved["total_steps"] is None
    assert resolved["encoder_mode"] == "none" and resolved["row_policy"] == "independent"
    assert resolved["num_particles"] == 11 and resolved["serve_average"] == 4.
    assert resolved["output_noise_mode"] == "learnable" and resolved["prior_reg"] == 0.
    assert task["evaluation"] == old_task["evaluation"]
    for key in ("steps", "host_definition", "resources", "prior", "initializer"):
        assert task["execution"].get(key) == old_task["execution"].get(key)
    assert task["policy_parent"] == old_task["policy_parent"]
    assert task["id"] != old_task["id"]
    receipt = rates.binding_receipt(rates.resolved_recipe(rates.candidate(profile), task), profile)
    assert rates.validate_binding_receipt(receipt, task) == receipt


def test_original_c6_declaration_and_default_producer_selection_remain_separate(task):
    old_task = original.make_variant(ROOT)
    recipe = original.resolved_recipe(original_candidate(), old_task)
    assert (recipe.lr, recipe.prior_lr_mult, recipe.d_lr_mult) == (.0053125, 1.5, 1.)
    assert original.COHORT == "word_joint_policy_min11_v1"
    assert producer._word_declaration(old_task) is original
    assert producer._word_declaration(task) is rates
    with pytest.raises(ValueError):
        original.resolved_recipe(rates.candidate("half_base"), old_task)
    unknown = deepcopy(task); unknown["task_cohort"] = "word_joint_policy_unknown"
    with pytest.raises(ValueError):
        producer._word_declaration(unknown)


@pytest.mark.parametrize("profile", tuple(rates.RATE_PROFILES))
def test_registry_public_context_sampling_and_owner_layout_dispatch(profile, task):
    candidate = rates.candidate(profile)
    context = api.task_formulation_context(candidate, task, {"seed": 0}, root=ROOT)
    assert context.recipe.to_dict() == rates.resolved_recipe(candidate, task).to_dict()
    assert context.policy is None  # Metadata only, no policy/model allocation.
    assert context.capabilities()["public_components"] and not context.capabilities()["public_trainer"]
    assert not context.capabilities()["routed_rows"]
    assert policy_cohorts.module_for_task(task) is rates
    assert policy_cohorts.validate_policy_observation(task) == original.observation()
    assert boundaries.prior_control_binding(task)["auxiliary_encoder"] == "original_free_continuous_WordEncoder"
    assert views._named_policy_layout(task) == views._named_policy_layout(original.make_variant(ROOT))
    assert sampling.expected_policy(task) == sampling.expected_policy(original.make_variant(ROOT))
    assert not api.task_policy_blockers(task, candidate)


def corrupt_candidate(candidate, kind):
    if kind == "unknown_profile": candidate["word_rate_profile"] = "unregistered"
    elif kind == "wrong_tuple": candidate["recipe_overrides"]["lr"] = .0053125
    elif kind == "extra_disable": candidate["recipe_overrides"]["serve_average"] = 0.
    elif kind == "missing_d": del candidate["recipe_overrides"]["d_lr_mult"]
    elif kind == "wrong_family": candidate["trainer_family"] = original.FAMILY
    elif kind == "wrong_preset": candidate["recipe_preset"] = "e22"
    elif kind == "unknown_cohort": candidate["task_cohort"] = "word_joint_policy_unknown"
    elif kind == "bool_rate": candidate["recipe_overrides"]["d_lr_mult"] = True
    elif kind == "nan_rate": candidate["recipe_overrides"]["lr"] = float("nan")
    elif kind == "inf_rate": candidate["recipe_overrides"]["lr"] = float("inf")
    elif kind == "string_rate": candidate["recipe_overrides"]["lr"] = ".001328125"
    elif kind == "extension": candidate["extensions"] = {"hidden_override": {"serve_average": 0.}}
    elif kind == "host_adaptation": candidate["host_adaptation"] = {"num_particles": 5}
    elif kind == "execution_path": candidate["execution_path"] = "public_components"
    elif kind == "prior_mog": candidate["prior"].update(kind="mog", sigma=.025)
    elif kind == "missing_prior": del candidate["prior"]
    elif kind == "prior_init": candidate["prior"]["init_std"] = .01
    elif kind == "initializer": candidate["initializer"] = "keep"
    elif kind == "tuple_id": candidate["id"] = "word-min11-quarter_base-rates-v1"
    else: raise AssertionError(kind)


@pytest.mark.parametrize("kind", ("unknown_profile", "wrong_tuple", "extra_disable", "missing_d",
    "wrong_family", "wrong_preset", "unknown_cohort", "bool_rate", "nan_rate", "inf_rate", "string_rate",
    "extension", "host_adaptation", "execution_path", "prior_mog", "missing_prior", "prior_init", "initializer", "tuple_id"))
def test_bad_tuning_is_rejected_before_any_model(kind, task):
    req = request("half_base"); corrupt_candidate(req["candidate"], kind)
    with pytest.raises(ValueError):
        producer.WordJointPolicyFixture(req, task)
    with pytest.raises(api.CapabilityError):
        api.task_formulation_context(req["candidate"], task, req["protocol"], root=ROOT)


@pytest.mark.parametrize("seed", (1, True, -1, None, 0.))
def test_original_named_seed_binding_precedes_models(seed, task):
    req = request("half_base"); req["protocol"]["seed"] = seed
    with pytest.raises(ValueError):
        producer.WordJointPolicyFixture(req, task)
    with pytest.raises(api.CapabilityError):
        api.task_formulation_context(req["candidate"], task, req["protocol"], root=ROOT)


@pytest.mark.parametrize("profile", tuple(rates.RATE_PROFILES))
def test_valid_constructor_binds_recipe_before_first_fake_model_factory(profile, task, monkeypatch):
    class StoppedBeforeModel(Exception): pass
    calls = []
    def stop(*args, **kwargs):
        calls.append("generator")
        raise StoppedBeforeModel()
    monkeypatch.setattr(producer, "WordGenerator", stop)
    fixture = object.__new__(producer.WordJointPolicyFixture)
    with pytest.raises(StoppedBeforeModel):
        fixture.__init__(request(profile), task)
    assert calls == ["generator"]
    assert fixture.recipe.to_dict() == rates.resolved_recipe(rates.candidate(profile), task).to_dict()
    assert fixture.rate_profile == profile and fixture.KIND == rates.KIND
    assert fixture.declaration is rates


def test_old_constructor_keeps_its_default_recipe_and_checkpoint_kind(monkeypatch):
    class StoppedBeforeModel(Exception): pass
    def stop(*args, **kwargs): raise StoppedBeforeModel()
    monkeypatch.setattr(producer, "WordGenerator", stop)
    fixture = object.__new__(producer.WordJointPolicyFixture)
    old_task = original.make_variant(ROOT)
    with pytest.raises(StoppedBeforeModel):
        fixture.__init__({"candidate": original_candidate(), "protocol": {"seed": 0}}, old_task)
    assert fixture.recipe.to_dict() == original.resolved_recipe(original_candidate(), old_task).to_dict()
    assert fixture.rate_profile is None and fixture.KIND == "forge_word_joint_policy_min11_v1"
    assert fixture.declaration is original


def test_only_typed_inert_compiler_annotations_are_normalized(task, monkeypatch):
    class StoppedBeforeModel(Exception): pass
    def stop(*args, **kwargs): raise StoppedBeforeModel()
    monkeypatch.setattr(producer, "WordGenerator", stop)
    compiled = deepcopy(task); compiled["preflight_blockers"] = []
    fixture = object.__new__(producer.WordJointPolicyFixture)
    with pytest.raises(StoppedBeforeModel):
        fixture.__init__(request("half_base"), compiled)
    assert fixture.task == task
    compiled["preflight_blockers"] = {"ignored": True}
    with pytest.raises(ValueError):
        fixture.__init__(request("half_base"), compiled)


def corrupt_task(task, kind):
    if kind == "case": task["id"] += "_forged"
    elif kind == "source": task["execution"]["policy_contract"]["sources"][rates.SOURCES[0]] = "a" * 64
    elif kind == "parent": task["policy_parent"]["task_sha256"] = "b" * 64
    elif kind == "gate": task["evaluation"]["thresholds"][0][2] = .1
    elif kind == "cadence": task["evaluation"]["observations"] = 1
    elif kind == "horizon": task["execution"]["steps"] = 3
    elif kind == "n5": task["execution"]["resources"]["num_particles"] = 5
    elif kind == "objective": task["execution"]["policy_contract"]["objective"] = "extra_inverse_loss"
    elif kind == "sampler": task["evaluation"]["policy_observation"]["sampler"] = "plain_live"
    else: raise AssertionError(kind)


@pytest.mark.parametrize("kind", ("case", "source", "parent", "gate", "cadence", "horizon", "n5", "objective", "sampler"))
def test_canonical_parent_source_case_and_law_are_fail_closed(kind, task):
    corrupt_task(task, kind)
    with pytest.raises((ValueError, api.CapabilityError)):
        policy_cohorts.validate_policy_task(task, root=ROOT)
    with pytest.raises(ValueError):
        producer.WordJointPolicyFixture(request("half_base"), task)


@pytest.mark.parametrize("profile", tuple(rates.RATE_PROFILES))
def test_synthetic_full_receipt_reaches_same_strict_sampling_and_grade(profile, task, tmp_path):
    raw = synthetic(task, profile, tmp_path)
    before = deepcopy(raw)
    assert sampling.grade_sampling(task, raw["evidence"]) is None
    assert views.grade_result(task, raw)["status"] == "PASS"  # Invented software inputs only.
    assert raw == before
    name, comparison, bound = task["evaluation"]["thresholds"][0]
    raw["evidence"]["observations"][-2][name] = float(bound) + (-1 if comparison == ">=" else 1)
    assert views.grade_result(task, raw)["status"] == "FAIL"


def test_original_synthetic_grade_requires_no_new_rate_receipt(tmp_path):
    old_task = original.make_variant(ROOT)
    path = ROOT / "tests/test_forge_named_policy_grading.py"
    spec = importlib.util.spec_from_file_location("old_word_guard_inputs", path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    raw = helper.synthetic_receipt(old_task, tmp_path)
    assert "word_rate_binding" not in raw["evidence"]["policy_controls"]
    assert views.grade_result(old_task, raw)["status"] == "PASS"  # Synthetic receipt only.


@pytest.mark.parametrize("role", ("generator", "encoder", "prior", "discriminator"))
@pytest.mark.parametrize("count", (None, 0, 20000))
def test_every_actual_optimizer_owner_must_finish_full_budget(role, count, task, tmp_path):
    raw = synthetic(task, "half_base", tmp_path)
    if count is None: del raw["evidence"]["guards"]["optimizer_updates"][role]
    else: raw["evidence"]["guards"]["optimizer_updates"][role] = count
    assert views.grade_result(task, raw)["status"] != "PASS"


@pytest.mark.parametrize("kind", ("missing_binding", "wrong_profile", "changed_recipe", "wrong_recipe_hash",
    "old_family", "missing_encoder", "split_code", "all_coordinates_noise"))
def test_no_rate_or_owner_stamp_can_replace_actual_word_contract(kind, task, tmp_path):
    raw = synthetic(task, "half_base", tmp_path)
    controls = raw["evidence"]["policy_controls"]
    if kind == "missing_binding": del controls["word_rate_binding"]
    elif kind == "wrong_profile": controls["word_rate_binding"]["profile"] = "quarter_base"
    elif kind == "changed_recipe": controls["word_rate_binding"]["resolved_recipe"]["serve_average"] = 0.
    elif kind == "wrong_recipe_hash": controls["word_rate_binding"]["resolved_recipe_sha256"] = "c" * 64
    elif kind == "old_family": controls["family"] = original.FAMILY
    elif kind == "missing_encoder": controls["roles"][0].remove("encoder")
    elif kind == "split_code": controls["joint_atom_code"] = "raw_code"
    elif kind == "all_coordinates_noise": controls["output_noise_coordinates"] = "all170"
    assert views.grade_result(task, raw)["status"] != "PASS"


@pytest.mark.parametrize("kind", ("old_family", "no_controller", "output_noise", "forced_ema"))
def test_sampling_cannot_borrow_a_foreign_or_nonpolicy_observation(kind, task, tmp_path):
    raw = synthetic(task, "half_base", tmp_path)
    observation = raw["evidence"]["policy_observation"]
    if kind == "old_family": observation["family"] = original.FAMILY
    elif kind == "no_controller": observation["controller"] = None
    elif kind == "output_noise": observation["output_noise"] = True
    else: observation["weight_selector"] = "forced_ema"
    assert sampling.grade_sampling(task, raw["evidence"]) is not None


@pytest.mark.parametrize("profile", tuple(rates.RATE_PROFILES))
def test_json_roundtrip_keeps_requested_profile_tuple_full_recipe_and_both_receipts(profile, task, tmp_path):
    req, raw = request(profile), synthetic(task, profile, tmp_path)
    req, raw = json.loads(json.dumps(req)), json.loads(json.dumps(raw))
    binding = rates.validate_result_binding(req, task, raw)
    assert binding["tuple_id"] == req["candidate"]["id"]
    assert stable_hash(binding) == stable_hash(raw["evidence"]["policy_controls"]["word_rate_binding"])
    assert stable_hash(binding) == stable_hash(raw["applied"]["policy_lifecycle"]["controls"]["word_rate_binding"])
    with pytest.raises(ValueError, match="tuple ID"):
        rates.candidate(profile, identifier="a-different-tuple")


@pytest.mark.parametrize("kind", ("cross_profile", "applied_recipe", "applied_defaults", "applied_resource",
    "applied_missing", "missing_first_receipt", "missing_second_receipt", "request_tuple_id", "request_seed"))
def test_request_bound_independent_evaluator_rejects_coherent_foreign_receipts(kind, task, tmp_path, monkeypatch):
    req, raw = request("half_base"), synthetic(task, "half_base", tmp_path)
    if kind == "cross_profile":
        for binding in (raw["evidence"]["policy_controls"]["word_rate_binding"],
                        raw["applied"]["policy_lifecycle"]["controls"]["word_rate_binding"]):
            binding.update(profile="quarter_base", tuple_id="word-min11-quarter_base-rates-v1")
            binding["resolved_recipe"]["lr"] = .001328125
    elif kind == "applied_recipe": raw["applied"]["recipe"]["lr"] = .0053125
    elif kind == "applied_defaults": raw["applied"]["recipe"]["serve_average"] = 0.
    elif kind == "applied_resource": raw["applied"]["actual_resources"]["num_particles"] = 5
    elif kind == "applied_missing": del raw["applied"]
    elif kind == "missing_first_receipt": del raw["evidence"]["policy_controls"]["word_rate_binding"]
    elif kind == "missing_second_receipt": del raw["applied"]["policy_lifecycle"]["controls"]["word_rate_binding"]
    elif kind == "request_tuple_id": req["candidate"]["id"] = "word-min11-quarter_base-rates-v1"
    elif kind == "request_seed": req["protocol"]["seed"] = 1
    root = tmp_path / "fake-independent-evaluator"; root.mkdir()
    resolved = root / "resolved.json"
    req.update(tasks={task["id"]: task}, source={"digest": "a" * 64})
    resolved.write_text(json.dumps({"request": req, "job": {"task_id": task["id"]}}))
    (root / "raw-result.json").write_text(json.dumps(raw))
    monkeypatch.setattr(evaluate, "MemoryProbe", lambda: SimpleNamespace(snapshot=lambda: {"synthetic": True}))
    result = evaluate.evaluate(resolved)
    assert result["grades"][task["id"]]["status"] == "INVALID"


@pytest.mark.parametrize("original_path", (False, True))
@pytest.mark.parametrize("numeric_fail", (False, True))
def test_independent_evaluator_preserves_valid_profiles_old_default_and_numeric_failure(
        original_path, numeric_fail, task, tmp_path, monkeypatch):
    if original_path:
        task = original.make_variant(ROOT)
        helper_path = ROOT / "tests/test_forge_named_policy_grading.py"
        spec = importlib.util.spec_from_file_location("old_evaluator_guard_inputs", helper_path)
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        raw, req = helper.synthetic_receipt(task, tmp_path), {"candidate": original_candidate(), "protocol": {"seed": 0}}
    else:
        raw, req = synthetic(task, "half_base", tmp_path), request("half_base")
    if numeric_fail: raw["evidence"]["observations"][-2]["reconstruction_exact"] = 0
    req.update(tasks={task["id"]: task}, source={"digest": "a" * 64})
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps({"request": req, "job": {"task_id": task["id"]}}))
    (tmp_path / "raw-result.json").write_text(json.dumps(raw))
    monkeypatch.setattr(evaluate, "MemoryProbe", lambda: SimpleNamespace(snapshot=lambda: {"synthetic": True}))
    before = deepcopy(raw)
    result = evaluate.evaluate(resolved)
    assert result["grades"][task["id"]]["status"] == ("FAIL" if numeric_fail else "PASS")
    assert json.loads((tmp_path / "raw-result.json").read_bytes()) == json.loads(json.dumps(before))


def test_named_projection_preserves_all_twenty_six_original_slots(task):
    view = json.loads((ROOT / "configs/forge/views/discriminator_stability.json").read_bytes())
    selected = {assignment["task"]: {} for assignment in view["assignments"]}
    selected.pop(rates.PARENT_ID); selected[task["id"]] = task
    projected = named_policy_planning.project_view(view, selected, rates.COHORT, rates.FAMILY)
    assert len(projected["assignments"]) == 26
    assert [sum(row["qualification_tier"] == tier for row in projected["assignments"]) for tier in (1,2,3)] == [5,19,2]
    for parent, actual in zip(view["assignments"], projected["assignments"]):
        assert {k:v for k,v in actual.items() if k != "task"} == {k:v for k,v in parent.items() if k != "task"}
        assert actual["task"] == (task["id"] if parent["task"] == rates.PARENT_ID else parent["task"])


def planned_request(monkeypatch):
    declaration = {**rates.candidate("half_base"), "schema_version": 1,
        "goal": "discriminator_stability", "hypothesis": "Synthetic metadata for one half-base word step contrast.",
        "changed_factors": ["Half the shared base LR while preserving prior/G ratio and all family owners."],
        "mechanism_class": "floor_constant",
        "claim_contract": {"schedule": "schedule_free", "scoring_weights": "state_selected", "sampling_law": "task_declared"}}
    # No hardware query, scientific snapshot, queue or source mutation in this software control.
    monkeypatch.setattr(planning, "compute_profile", lambda backend, *args: {
        "backend": backend, "model": "synthetic_metadata_gpu", "threads": 1})
    return planning.resolve_idea(ROOT, declaration["id"], declaration=declaration, freeze_source=False)


def test_actual_model_free_planner_keeps_reference_and_active_execution_distinct(task, monkeypatch):
    request = planned_request(monkeypatch)
    assert request["candidate"]["execution_path"] == "public_trainer"
    assert len(request["tasks"]) == 26
    assert [sum(a["qualification_tier"] == tier for a in request["view"]["assignments"]) for tier in (1,2,3)] == [5,19,2]
    actual = request["tasks"][task["id"]]
    assert actual["execution"]["execution_path"] == "public_components"
    assert actual["preflight_blockers"] == []
    assert request["candidate"]["capabilities"]  # Actual pure reference-context resolution succeeded.
    assert stable_hash(policy_cohorts.policy_task_declaration(actual)) == stable_hash(task)
    context = api.task_formulation_context(request["candidate"], actual, request["protocol"], root=ROOT)
    assert context.execution_path == "public_components" and context.policy is None
    assert context.recipe.to_dict() == rates.resolved_recipe(request["candidate"], task).to_dict()


def test_source_bound_publication_accepts_only_full_named_projection_and_current_tuple(task, monkeypatch):
    request = planned_request(monkeypatch)
    actual = request["view"]
    common = json.loads((ROOT / "configs/forge/views/discriminator_stability.json").read_bytes())
    original_row = {"candidate_id": request["candidate"]["id"], "status": "FAIL", "qualified_tier": 0,
        "runtime_cohort": {"execution_backend": "cuda"}, "cost": {},
        "qualification": {"view_revision": actual["revision"], "policy_fingerprint": stable_hash(actual),
            "tasks": [{"task_id": row["task"], "status": "FAIL" if row["task"] == task["id"] else "NOT_RUN"}
                      for row in actual["assignments"]]},
        "scientific_bindings": technique_board.request_bindings(request), **technique_board.policy_row_metadata(request)}
    board = {"view": common["id"], "view_revision": common["revision"], "policy_fingerprint": stable_hash(common),
        "current_rows": [original_row], "rows": [], "conflicts": []}
    report = technique_board.reduce_board(board, common)
    report["publication_scope"] = "live_current"
    report["fixture_scope"] = "synthetic_metadata_no_scientific_claim"
    row = report["rows"][0]
    assert policy_snapshot_publication.validate_policy_publication(ROOT, report, row) is None
    before = deepcopy((report, row))
    row["task_slot_map"][task["id"]] = "two_pole"
    with pytest.raises(ValueError):
        policy_snapshot_publication.validate_policy_publication(ROOT, report, row)
    report, row = deepcopy(before)
    row["bindings"]["trainer_family"] = original.FAMILY
    with pytest.raises(ValueError):
        policy_snapshot_publication.validate_policy_publication(ROOT, report, row)


@pytest.mark.parametrize("retired", ("quarter_base", "slower_prior"))
def test_retired_unexecuted_proposals_are_not_executable_profiles(retired, task):
    assert tuple(rates.RATE_PROFILES) == ("half_base",)
    with pytest.raises(ValueError): rates.candidate(retired)
    req = request("half_base"); req["candidate"]["word_rate_profile"] = retired
    with pytest.raises(ValueError): producer.WordJointPolicyFixture(req, task)


def fake_checkpoint_fixture(task, profile):
    fixture = object.__new__(producer.WordJointPolicyFixture)
    fixture.declaration, fixture.rate_profile, fixture.KIND = rates, profile, rates.KIND
    fixture.recipe = rates.resolved_recipe(rates.candidate(profile), task)
    fixture.task, fixture.max_steps = task, 20001
    fixture.initialization, fixture.last_update = {}, {}
    fixture.module_modes = lambda: {}
    fixture.words = torch.zeros(5, 28, 6)
    fixture.prior = SimpleNamespace(z=torch.zeros(11, 2))
    state = {"streams": {}, "table": fixture.prior.z, "averaged_table": fixture.prior.z.clone(),
             "completed_steps": 0, "recipe": fixture.recipe.to_dict()}
    def forbidden_restore(*args, **kwargs):
        raise AssertionError("the metadata rejection must precede public state restore")
    fixture.policy = SimpleNamespace(completed_steps=0, state_dict=lambda: deepcopy(state), load_state_dict=forbidden_restore,
        _training_modules=lambda: {}, streams={})
    fixture.streams = SimpleNamespace(state_dict=lambda: {}, validate_state_dict=lambda value: None, load_state_dict=lambda value: None)
    fixture.mechanisms = SimpleNamespace(rows={})
    calls = dict.fromkeys(("begin_step", "after_critic_step", "after_generator_backward", "after_generator_step", "finish_step"), 0)
    fixture.audit = SimpleNamespace(calls=calls, receipt=lambda *args: {"complete": True, "calls": deepcopy(calls),
        "start_completed_steps": 0, "end_completed_steps": 0, "last_order": [], "pending": [], "order_errors": 0})
    fixture.modules = lambda: {}
    return fixture


def test_complete_checkpoint_identity_prevents_foreign_c6_recipe_or_retired_profile(task):
    fixture = fake_checkpoint_fixture(task, "half_base")
    state = fixture.state_dict()
    assert state["kind"] == rates.KIND and state["family"] == rates.FAMILY
    assert state["word_rate_binding"] == rates.binding_receipt(fixture.recipe, "half_base")
    foreign = deepcopy(state)
    foreign["recipe"]["lr"] = foreign["policy"]["recipe"]["lr"] = .0053125
    with pytest.raises(ValueError, match="recipe"):
        fixture.load_state_dict(foreign)
    forged = deepcopy(state); forged["word_rate_binding"]["profile"] = "quarter_base"
    with pytest.raises(ValueError, match="word_rate_binding"):
        fixture.load_state_dict(forged)


def test_synthetic_checkpoint_metadata_roundtrip_preserves_profile_and_original_c6_is_rejected(task):
    fixture = fake_checkpoint_fixture(task, "half_base")
    before = fixture.state_dict()
    loaded = []
    fixture.policy.load_state_dict = lambda value: loaded.append(deepcopy(value))
    fixture.load_state_dict(deepcopy(before))  # Fake public owner, no modules or optimizer state.
    assert len(loaded) == 1
    assert fixture.state_dict()["word_rate_binding"] == before["word_rate_binding"]
    old_kind = deepcopy(before); old_kind["kind"] = "forge_word_joint_policy_min11_v1"
    with pytest.raises(ValueError, match="kind"):
        fixture.load_state_dict(old_kind)
    assert len(loaded) == 1
