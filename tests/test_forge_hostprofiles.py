"""Frozen host-profile boundaries; no model construction or training."""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.hostprofiles import MODULE, profile_source_paths, validate_request_host_profiles
from experiments.forge.queue import Queue
from experiments.forge.sources import inspect_source, snapshot_source
from experiments.forge.views import task_execution_fingerprint, task_evaluation_fingerprint
from test_forge_queue import request, campaign, grade

ROOT = Path(__file__).resolve().parents[1]
PRIOR = {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}


def task(name):
    return read_json(ROOT / "configs/forge/tasks" / f"{name}.json")


def native_tasks(continuation=False):
    from experiments.forge.nativeprofiles import task_from_profile
    parent = task_from_profile(task("grid100"), "grid100_profiled")
    if not continuation:
        return [parent]
    child = task_from_profile(task("grid100_14k"), "grid100_profiled_14k", parent_task_id=parent["id"])
    return [parent, child]



def bind_candidate(req):
    from dataclasses import asdict
    from experiments.forge.api import FormulationContext
    from experiments.forge.planning import candidate_revision_for
    candidate = req["candidate"]
    candidate["resolved_recipe"] = asdict(FormulationContext(
        recipe_overrides=candidate.get("recipe_overrides", {}), prior=candidate["prior"],
        initializer=candidate.get("initializer", "deterministic_orthogonal")).recipe)
    req["candidate_revision"] = candidate_revision_for(req["source"]["digest"], candidate)


def rebind(req):
    for job in req["jobs"]:
        members = job.get("task_ids", [job["task_id"]])
        job["science"].update(candidate_revision=req["candidate_revision"], initializer="deterministic_orthogonal",
            task_initializers={name: req["tasks"][name]["execution"]["initializer"] for name in members},
            execution={name: task_execution_fingerprint(req["tasks"][name]) for name in members},
            evaluation={name: task_evaluation_fingerprint(req["tasks"][name]) for name in members})
        parent = req["tasks"][job["task_id"]]["execution"].get("continuation_of")
        if parent:
            job["science"]["prerequisites"] = {parent: next(j["compatibility_key"] for j in req["jobs"] if j["task_id"] == parent)}
        job["compatibility_key"] = stable_hash(job["science"])


def prospective(tmp_path, tasks=None, *, grouped=False):
    req = request(tmp_path, cap=1)
    templates = tasks or [task("img_intensity2_residual16"), task("vector_unequal_mass_published")]
    req["candidate"].update(prior=deepcopy(PRIOR), recipe_overrides={}, requires_capabilities=[],
                            claim_contract={"sampling_law": "task_declared"})
    req["tasks"] = {t["id"]: deepcopy(t) for t in templates}
    req["view"]["assignments"] = [{"task": t["id"], "qualification_tier": 1, "importance": "required", "order": i}
                                  for i, t in enumerate(templates)]
    jobs = [deepcopy(req["jobs"][0]) for _ in templates]
    for job, t in zip(jobs, templates):
        job["task_id"] = t["id"]
        resources = t["resources"]
        job["resources"] = {"memory_mb": resources["gpu_memory_mb"], "gpus": resources["gpus"],
                            "cpu_threads": resources["cpu_threads"], "allow_cpu": True, "backend": "cpu"}
        job["budget_seconds"] = resources["timeout_seconds"]
        job["science"]["compute"] = {"backend": "cpu", "threads": resources["cpu_threads"]}
        req["tasks"][t["id"]]["preflight_blockers"] = []
    req["jobs"] = jobs
    if grouped:
        jobs[0]["task_ids"] = [t["id"] for t in templates]
        jobs[0]["budget_seconds"] = max(t["resources"]["timeout_seconds"] for t in templates)
        del jobs[1:]
    req["preflight_blockers"] = []
    checkout = tmp_path / "worktree"
    files = {MODULE} | {name for t in templates for name in profile_source_paths(t)}
    for name in files:
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    manifest = inspect_source(checkout, files)
    manifest["snapshot_path"] = str(snapshot_source(checkout, tmp_path / "queue", manifest))
    req["source"] = manifest
    bind_candidate(req)
    rebind(req)
    return req


@pytest.mark.parametrize("family", ["image", "vector", "native"])
def test_valid_profile_is_checked_then_queued_without_model_construction(tmp_path, monkeypatch, family):
    selected = {"image": [task("img_intensity2_residual16")],
                "vector": [task("vector_unequal_mass_published")], "native": native_tasks()}[family]
    req = prospective(tmp_path, selected)
    def forbidden(*args, **kwargs):
        raise AssertionError("request validation must not construct models")
    monkeypatch.setattr("experiments.forge.api.FormulationContext.construct", forbidden)
    queue = Queue(tmp_path / "queue", grader=grade)
    assert queue.submit(req, campaign())["status"] == "queued"
    assert all(not row["attempts"] for row in queue.inspect()["jobs"].values())


def test_frozen_request_preserves_task_mog_width_without_candidate_override(tmp_path):
    req = prospective(tmp_path)
    vector = req["tasks"]["vector_unequal_mass_published"]
    vector["execution"]["prior"]["sigma"] = .1
    rebind(req)
    assert req["candidate"]["prior"]["sigma"] == .025
    validate_request_host_profiles(req)
    queue = Queue(tmp_path / "queue", grader=grade)
    assert queue.submit(req, campaign())["status"] == "queued"
    assert req["tasks"][vector["id"]]["execution"]["prior"]["sigma"] == .1


@pytest.mark.parametrize("mutation", ["image_width", "vector_card", "profile_version", "missing_profile",
    "missing_prior", "invalid_prior", "resource_conflict", "unknown_recipe", "wrong_adapter", "missing_task"])
def test_semantic_forgery_blocks_even_with_rehashed_jobs_and_empty_cached_blockers(tmp_path, mutation):
    req = prospective(tmp_path)
    image, vector = req["tasks"].values()
    if mutation == "image_width": image["execution"]["host_definition"]["width"] = 12
    elif mutation == "vector_card": vector["execution"]["host_definition"]["research_discriminator"]["softplus_beta"] = 1.
    elif mutation == "profile_version": vector["execution"]["vector_profile"]["revision"] = 2
    elif mutation == "missing_profile": vector["execution"].pop("vector_profile")
    elif mutation == "missing_prior": vector["execution"].pop("prior")
    elif mutation == "invalid_prior":
        req["candidate"]["prior"]["sigma"] = 0
        vector["execution"]["prior"]["sigma"] = 0
    elif mutation == "resource_conflict": req["candidate"]["recipe_overrides"]["batch_size"] = 17
    elif mutation == "unknown_recipe": req["candidate"]["recipe_overrides"]["made_up_variable"] = 1
    elif mutation == "wrong_adapter": vector["adapter"] = "transfer_behavior"
    elif mutation == "missing_task": req["tasks"].pop(vector["id"])
    if mutation != "missing_task": rebind(req)
    with pytest.raises(ValueError, match="host profile blocked"):
        validate_request_host_profiles(req)
    queue = Queue(tmp_path / "queue", grader=grade)
    # Existing seed/formulation checks may independently reject first.
    with pytest.raises(ValueError):
        queue.submit(req, campaign())
    assert not (queue.root / "queue/state.json").exists()
    assert not queue.inspect()["jobs"]


@pytest.mark.parametrize("mutation", ["supplied", "prior_init_std", "raw_parent", "wrong_parent_key"])
def test_native_initialization_and_own_parent_contracts_are_revalidated(tmp_path, mutation):
    req = prospective(tmp_path, native_tasks(continuation=True))
    if mutation == "supplied": req["candidate"]["initializer"] = "supplied"
    elif mutation == "prior_init_std":
        req["candidate"]["prior"]["init_std"] = .5
        for t in req["tasks"].values(): t["execution"]["prior"]["init_std"] = .5
    elif mutation == "raw_parent":
        parent = req["tasks"]["grid100_profiled"]
        parent["execution"].pop("native_profile")
        parent["execution"].pop("host_definition")
    rebind(req)
    if mutation == "wrong_parent_key":
        req["jobs"][1]["science"]["prerequisites"]["grid100_profiled"] = "wrong"
        req["jobs"][1]["compatibility_key"] = stable_hash(req["jobs"][1]["science"])
    with pytest.raises(ValueError, match="host profile blocked"):
        validate_request_host_profiles(req)


def test_matching_native_parent_and_continuation_validate_without_updates(tmp_path):
    validate_request_host_profiles(prospective(tmp_path, native_tasks(continuation=True)))


def test_stale_job_identity_cannot_label_a_changed_host(tmp_path):
    req = prospective(tmp_path)
    req["tasks"]["vector_unequal_mass_published"]["execution"]["prior"]["sigma"] = .05
    req["candidate"]["prior"]["sigma"] = .05
    with pytest.raises(ValueError, match="scientific identity"):
        validate_request_host_profiles(req)


def test_grouped_child_cannot_hide_invalid_profile(tmp_path):
    req = prospective(tmp_path, grouped=True)
    req["view"]["assignments"][1]["qualification_tier"] = 2
    req["tasks"]["vector_unequal_mass_published"]["execution"]["vector_profile"]["revision"] = 7
    rebind(req)
    with pytest.raises(ValueError, match="vector_unequal_mass_published"):
        Queue(tmp_path / "queue", grader=grade).submit(req, campaign())


def test_pinned_source_semantics_are_checked_after_valid_manifest_rehash(tmp_path):
    from experiments.forge.vectorprofiles import LEADING_SOURCE
    req = prospective(tmp_path)
    root = Path(req["source"]["snapshot_path"])
    path = root / LEADING_SOURCE
    path.write_bytes(path.read_bytes() + b"\n")
    source = inspect_source(root, req["source"]["files"])
    source["snapshot_path"] = str(root)
    req["source"] = source
    with pytest.raises(ValueError, match="published vector profile source changed"):
        validate_request_host_profiles(req)


def test_manifest_omission_cannot_downgrade_source_floor(tmp_path):
    req = prospective(tmp_path)
    req["source"]["files"].pop(MODULE)
    req["source"]["digest"] = stable_hash(req["source"]["files"])
    with pytest.raises(ValueError, match="undeclared files"):
        validate_request_host_profiles(req)


def test_legacy_snapshot_with_profile_keeps_its_own_execution_contract(tmp_path):
    req = request(tmp_path, cap=1)
    req["tasks"]["t1"]["execution"] = {"native_profile": {"revision": "legacy"}}
    before = deepcopy(req)
    validate_request_host_profiles(req)
    assert req == before
    assert Queue(tmp_path / "queue", grader=grade).submit(req, campaign())["status"] == "queued"


def test_diagnostic_namespace_is_retained_without_granting_ordinary_keys(tmp_path):
    req = prospective(tmp_path)
    for job in req["jobs"]:
        job["qualification_compatibility_key"] = job["compatibility_key"]
        job["science"]["evidence_use"] = "calibration_diagnostic"
        job["compatibility_key"] = stable_hash(job["science"])
    before = deepcopy(req)
    validate_request_host_profiles(req)
    assert req == before
    assert all(j["compatibility_key"] != j["qualification_compatibility_key"] for j in req["jobs"])


def declared_configuration_request(card):
    """Bind only a declaration; construction and candidate checks draw no RNG."""
    from dataclasses import asdict
    from experiments.forge.api import FormulationContext
    from experiments.forge.planning import candidate_revision_for
    candidate = deepcopy(card)
    # Runtime requests carry a resolved prior even when reusable v3 cards do
    # not author one. Match the planner's default binding in this fixture.
    candidate["prior"] = {**read_json(ROOT / "configs/forge/defaults.json")["prior"],
                          **candidate.get("prior", {})}
    candidate["resolved_recipe"] = asdict(FormulationContext(
        recipe_preset=candidate.get("recipe_preset"),
        recipe_overrides=candidate.get("recipe_overrides", {}), prior=candidate.get("prior"),
        requires_capabilities=candidate.get("requires_capabilities", ()),
        initializer=candidate.get("initializer", "deterministic_orthogonal")).recipe)
    source = {"digest": "a" * 64}
    return {"candidate": candidate, "source": source,
            "candidate_revision": candidate_revision_for(source["digest"], candidate)}


def test_historical_configuration_default_loss_is_implicit_without_weakening_identity():
    from experiments.forge.hostprofiles import _validate_candidate_identity
    paths = sorted((ROOT / "configs/forge/configurations").glob("r1r2--*.json"))
    card = next(read_json(path) for path in paths
                if "loss" not in read_json(path)["resolved_configuration_recipe"])
    request = declared_configuration_request(card)
    assert request["candidate"]["resolved_recipe"]["loss"] == "relativistic"
    before = deepcopy(request)
    _validate_candidate_identity(request)
    assert request == before
    # The compatibility normalization must preserve nondefault objectives and
    # cannot accept an alternative loss under the old configuration identity.
    request["candidate"]["resolved_configuration_recipe"]["loss"] = "hinge"
    with pytest.raises(ValueError, match="configuration declaration hash"):
        _validate_candidate_identity(request)
    request = deepcopy(before)
    request["candidate"]["resolved_recipe"]["loss"] = "hinge"
    with pytest.raises(ValueError, match="candidate resolved_recipe differs"):
        _validate_candidate_identity(request)


@pytest.mark.parametrize("loss", ["relativistic", "non_saturating", "hinge", "wasserstein", "least_squares"])
def test_pure_configuration_checks_keep_exact_declared_loss_and_request_identity(loss):
    from experiments.forge.hostprofiles import _validate_candidate_identity
    from experiments.forge.configuration_search import recipe_identity_fields
    cards = [read_json(path) for path in sorted((ROOT / "configs/forge/configurations").glob("bcap-pure--*.json"))]
    card = next(card for card in cards if card["resolved_configuration_recipe"]["loss"] == loss)
    request = declared_configuration_request(card)
    assert stable_hash(recipe_identity_fields(request["candidate"]["resolved_recipe"])) == stable_hash(recipe_identity_fields(card["resolved_configuration_recipe"]))
    assert request["candidate"]["resolved_recipe"]["loss"] == loss
    before = deepcopy(request)
    _validate_candidate_identity(request)
    assert request == before
    request["candidate"]["resolved_recipe"]["loss"] = "hinge" if loss != "hinge" else "relativistic"
    with pytest.raises(ValueError, match="candidate resolved_recipe differs"):
        _validate_candidate_identity(request)


@pytest.mark.parametrize("fields", [("optimizer_momentum",), ("optimizer_adam_lr",),
                                   ("optimizer_momentum", "optimizer_adam_lr")])
def test_implicit_optimizer_defaults_preserve_exact_historical_request_identity(fields):
    from experiments.forge.hostprofiles import _validate_candidate_identity
    from experiments.forge.planning import candidate_revision_for
    card = next(read_json(path) for path in sorted((ROOT / "configs/forge/configurations").glob("bcap-pure--*.json")))
    request = declared_configuration_request(card)
    for field in fields:
        request["candidate"]["resolved_recipe"].pop(field)
    request["candidate_revision"] = candidate_revision_for(request["source"]["digest"], request["candidate"])
    before = deepcopy(request)
    _validate_candidate_identity(request)
    assert request == before


@pytest.mark.parametrize("field,value", [("optimizer_momentum", .5), ("optimizer_momentum", False),
                                        ("optimizer_adam_lr", .003)])
def test_rehashed_optimizer_recipe_tampering_cannot_use_implicit_default_compatibility(field, value):
    from experiments.forge.hostprofiles import _validate_candidate_identity
    from experiments.forge.planning import candidate_revision_for
    card = next(read_json(path) for path in sorted((ROOT / "configs/forge/configurations").glob("bcap-pure--*.json")))
    request = declared_configuration_request(card)
    request["candidate"]["resolved_recipe"][field] = value
    request["candidate_revision"] = candidate_revision_for(request["source"]["digest"], request["candidate"])
    with pytest.raises(ValueError, match="candidate resolved_recipe differs"):
        _validate_candidate_identity(request)


@pytest.mark.parametrize("overrides,field", [
    ({"optimizer_family": "dualnorm", "optimizer_momentum": .5}, "optimizer_momentum"),
    ({"optimizer_family": "dualnorm_D_only", "optimizer_adam_lr": .00425}, "optimizer_adam_lr"),
])
def test_active_optimizer_configuration_fields_remain_explicit_and_required(overrides, field):
    from dataclasses import asdict
    from experiments.forge.api import FormulationContext
    from experiments.forge.hostprofiles import _validate_candidate_identity
    from experiments.forge.planning import candidate_revision_for
    candidate = {"recipe_preset": "bcap", "recipe_overrides": overrides, "prior": deepcopy(PRIOR)}
    candidate["resolved_recipe"] = asdict(FormulationContext(
        recipe_preset="bcap", recipe_overrides=overrides, prior=candidate["prior"]).recipe)
    request = {"candidate": candidate, "source": {"digest": "a" * 64}}
    request["candidate_revision"] = candidate_revision_for(request["source"]["digest"], candidate)
    _validate_candidate_identity(request)
    candidate["resolved_recipe"].pop(field)
    request["candidate_revision"] = candidate_revision_for(request["source"]["digest"], candidate)
    with pytest.raises(ValueError, match="candidate resolved_recipe differs"):
        _validate_candidate_identity(request)


def test_runtime_rechecks_grouped_member_before_adapter(tmp_path, monkeypatch):
    from experiments.forge import runtime
    req = prospective(tmp_path, grouped=True)
    req["tasks"]["vector_unequal_mass_published"]["execution"]["vector_profile"]["revision"] = 99
    rebind(req)
    req["campaign_id"] = "boundary-fixture"
    calls = []
    monkeypatch.setattr("experiments.forge.adapters.run_task", lambda *args: calls.append(args))
    import sys
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda _: None, use_deterministic_algorithms=lambda _: None,
        backends=SimpleNamespace(cudnn=SimpleNamespace(), cuda=SimpleNamespace(matmul=SimpleNamespace()))))
    monkeypatch.setattr(runtime, "MemoryProbe", lambda **kwargs: SimpleNamespace(snapshot=lambda: {}))
    path = tmp_path / "attempt/request.json"
    atomic_json(path, {"request": req, "job": req["jobs"][0], "worker": {"device": "cpu", "attempt": "fixture"}})
    assert runtime.execute(path) == 1
    assert calls == []
    raw = read_json(path.parent / "raw-result.json")
    assert raw["applicability"]["status"] == "unsupported"
    assert "host profile blocked" in raw["error"]["message"]


def test_unselected_malformed_optional_profile_does_not_block_original_work(tmp_path):
    req = prospective(tmp_path)
    req["view"]["assignments"][1]["qualification_tier"] = 2
    req["tasks"]["vector_unequal_mass_published"]["execution"]["vector_profile"]["revision"] = 99
    rebind(req)
    validate_request_host_profiles(req)
    assert Queue(tmp_path / "queue", grader=grade).submit(req, campaign())["status"] == "queued"


def test_new_profile_source_requires_snapshot_before_queue_mutation(tmp_path):
    req = prospective(tmp_path)
    req["source"].pop("snapshot_path")
    with pytest.raises(ValueError, match="requires a frozen source snapshot"):
        Queue(tmp_path / "queue", grader=grade).submit(req, campaign())


@pytest.mark.parametrize("field", ["budget_seconds", "cpu_threads", "memory_mb"])
def test_job_resource_changes_cannot_bypass_frozen_host_budget(tmp_path, field):
    req = prospective(tmp_path)
    if field == "budget_seconds": req["jobs"][0][field] += 1
    else: req["jobs"][0]["resources"][field] += 1
    with pytest.raises(ValueError, match="host job resources"):
        validate_request_host_profiles(req)


@pytest.mark.parametrize("mutation", ["valid_recipe_override", "resolved_recipe_only", "both_stale_revision", "forged_metadata_revision"])
def test_actual_formulation_cannot_execute_under_an_unchanged_or_forged_revision(tmp_path, mutation):
    from experiments.forge.planning import candidate_revision_for
    req = prospective(tmp_path)
    original_revision = req["candidate_revision"]
    original_keys = [job["compatibility_key"] for job in req["jobs"]]
    changed_lr = req["candidate"]["resolved_recipe"]["lr"] * 2
    if mutation in {"valid_recipe_override", "both_stale_revision"}:
        req["candidate"]["recipe_overrides"]["lr"] = changed_lr
    if mutation in {"resolved_recipe_only", "both_stale_revision", "forged_metadata_revision"}:
        req["candidate"]["resolved_recipe"]["lr"] = changed_lr
    if mutation == "forged_metadata_revision":
        # Rehashing a fabricated resolved_recipe does not make it executable.
        req["candidate_revision"] = candidate_revision_for(req["source"]["digest"], req["candidate"])
        rebind(req)
    else:
        assert req["candidate_revision"] == original_revision
        assert [job["compatibility_key"] for job in req["jobs"]] == original_keys
    queue = Queue(tmp_path / "queue", grader=grade)
    with pytest.raises(ValueError, match="candidate (resolved_recipe|scientific identity)"):
        queue.submit(req, campaign())
    assert not (queue.root / "queue/state.json").exists()


def test_legitimate_formulation_change_requires_new_recipe_revision_and_job_keys(tmp_path):
    req = prospective(tmp_path)
    old_revision = req["candidate_revision"]
    old_keys = [job["compatibility_key"] for job in req["jobs"]]
    req["candidate"]["recipe_overrides"]["lr"] = .002
    bind_candidate(req)
    rebind(req)
    assert req["candidate_revision"] != old_revision
    assert all(job["compatibility_key"] != old for job, old in zip(req["jobs"], old_keys))
    # JSON round trips turn tuple Recipe fields into lists; identity is canonical.
    import json
    validate_request_host_profiles(json.loads(json.dumps(req)))


def test_central_revision_helper_preserves_the_existing_planning_formula(tmp_path):
    from experiments.forge.planning import FORMULATION_FIELDS, candidate_revision_for
    req = prospective(tmp_path)
    candidate = req["candidate"]
    formulation = {name: candidate.get(name) for name in FORMULATION_FIELDS}
    formulation.update(resolved_recipe=candidate["resolved_recipe"], prior=candidate["prior"],
                       api_version=candidate.get("api_version", "forge-api-v1"))
    old_formula = stable_hash({"source": req["source"]["digest"], "formulation": formulation})
    assert candidate_revision_for(req["source"]["digest"], candidate) == old_formula
