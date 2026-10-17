"""Initialization/resource ownership contracts without scientific qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.adapters import _context, adapter_preflight
from experiments.forge.api import CapabilityError
from experiments.forge.behavior_adapters import BehaviorComponents, behavior_preflight
from experiments.forge.initialization import MODULE, task_initializer
from experiments.forge.nativeprofiles import native_host_initialization, native_profile_blockers, resolve_native_spec
from experiments.forge.views import task_execution_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def task(name="vector_two_broad"):
    return json.loads((ROOT / "configs/forge/tasks" / f"{name}.json").read_text())


def current_request(tmp_path, tasks):
    from experiments.forge.sources import inspect_source, snapshot_source
    from test_forge_hostprofiles import bind_candidate, prospective, rebind
    request = prospective(tmp_path, tasks=tasks)
    checkout = tmp_path / "worktree"
    (checkout / MODULE).write_bytes((ROOT / MODULE).read_bytes())
    manifest = inspect_source(checkout, set(request["source"]["files"]) | {MODULE})
    manifest["snapshot_path"] = str(snapshot_source(checkout, tmp_path / "new-source-queue", manifest))
    request["source"] = manifest
    bind_candidate(request)
    rebind(request)
    return request


@pytest.mark.parametrize("value", [None, "unsupported", [], 0])
def test_invalid_task_initializer_blocks_preflight_and_runtime_before_rng_draws(value):
    declaration = task()
    if value is None:
        del declaration["execution"]["initializer"]
    else:
        declaration["execution"]["initializer"] = value
    candidate = {"initializer": "deterministic_orthogonal"}
    before = torch.get_rng_state().clone()
    assert "execution.initializer" in adapter_preflight(declaration, candidate)[0]
    with pytest.raises(ValueError, match="execution.initializer"):
        _context({"candidate": candidate, "protocol": {"seed": 0}}, declaration, "cpu",
                 {"num_particles": 256, "z_dim": 4, "batch_size": 128})
    assert torch.equal(before, torch.get_rng_state())


def test_task_policy_binds_even_without_a_candidate_initializer_and_changes_identity():
    declaration = task()
    original = deepcopy(declaration)
    declaration["execution"]["initializer"] = "supplied"
    candidate = {"recipe_overrides": {}}
    assert not adapter_preflight(declaration, candidate)
    context = _context({"candidate": candidate, "protocol": {"seed": 0}}, declaration, "cpu",
                       {"num_particles": 256, "z_dim": 4, "batch_size": 128})
    assert context.initializer == "supplied"
    assert context.receipt()["initializer"] == "supplied"
    assert "initializer" not in candidate
    assert task_execution_fingerprint(declaration) != task_execution_fingerprint(original)


@pytest.mark.parametrize("name", ["vector_two_broad", "two_pole", "grid100_affine_square_named_v1"])
def test_explicit_candidate_initializer_conflict_blocks_all_host_paths(name):
    declaration = task(name)
    candidate = {"initializer": "supplied"}
    assert any("candidate initializer conflicts" in reason for reason in adapter_preflight(declaration, candidate))
    with pytest.raises(ValueError, match="candidate initializer conflicts"):
        from experiments.forge.api import task_recipe_resources
        _context({"candidate": candidate, "protocol": {"seed": 0}}, declaration, "cpu",
                 task_recipe_resources(declaration))
    if name == "two_pole":
        assert "candidate initializer conflicts" in behavior_preflight(declaration, candidate)[0]
        with pytest.raises(CapabilityError, match="candidate initializer conflicts"):
            BehaviorComponents({"candidate": candidate, "protocol": {"seed": 0}}, declaration)
    if "affine" in name:
        assert "candidate initializer conflicts" in native_profile_blockers(declaration, candidate)[0]


def test_catalog_freezes_initializers_and_retains_specific_host_policies():
    declarations = [json.loads(path.read_text()) for path in (ROOT / "configs/forge/tasks").glob("*.json")]
    assert all(task_initializer(value) == "deterministic_orthogonal" for value in declarations)
    assert task("two_pole")["execution"]["fixed_initialization"] == {
        "critic": "stored_host_weights", "particles": "zeros"}
    pinned = native_host_initialization(task("grid100_affine_square_named_v1"))
    assert pinned["components"]["generator"] == {"method": "identity_linear_v1"}
    assert pinned["components"]["prior"]["parameters"]["z"] == {"kind": "uniform", "low": -5.0, "high": 5.0}


def test_plain_native_resources_belong_to_task_and_model_variants_need_a_profile():
    declaration = task("grid100")
    assert resolve_native_spec(declaration) is None
    assert declaration["execution"]["resources"] == {"num_particles": 20000, "z_dim": 2, "batch_size": 2048}
    assert any("num_particles" in reason for reason in adapter_preflight(
        declaration, {"recipe_overrides": {"num_particles": 12}}))
    declaration["execution"]["resources"] = {"num_particles": 12, "z_dim": 2, "batch_size": 4}
    assert not adapter_preflight(declaration, {})
    context = _context({"candidate": {}, "protocol": {"seed": 0}}, declaration, "cpu",
                       declaration["execution"]["resources"])
    assert (context.recipe.num_particles, context.recipe.z_dim, context.recipe.batch_size) == (12, 2, 4)
    declaration["execution"]["model"]["hidden"] = 64
    assert any("explicit profile" in reason for reason in adapter_preflight(declaration, {}))


@pytest.mark.parametrize("field", ["initializer", "resources"])
def test_plain_native_continuation_requires_matching_task_owned_fields(field):
    from experiments.forge.nativeprofiles import validate_native_continuation
    parent, child = task("grid100"), task("grid100_14k")
    validate_native_continuation(parent, child)
    if field == "initializer":
        child["execution"][field] = "supplied"
    else:
        child["execution"][field]["num_particles"] = 12
    with pytest.raises(ValueError, match=f"changes task-owned {field}"):
        validate_native_continuation(parent, child)


def test_legacy_fallback_requires_explicit_call_and_preserves_saved_task():
    declaration = task()
    del declaration["execution"]["initializer"]
    saved = deepcopy(declaration)
    with pytest.raises(ValueError, match="must be explicit"):
        task_initializer(declaration, {"initializer": "supplied"})
    assert task_initializer(declaration, {"initializer": "supplied"}, explicit=False) == "supplied"
    assert task_initializer(declaration, explicit=False) == "deterministic_orthogonal"
    assert declaration == saved


def test_frozen_host_boundary_requires_task_initialization_map_only_for_new_source(tmp_path):
    from experiments.forge.contracts import stable_hash
    from experiments.forge.hostprofiles import validate_request_host_profiles
    from experiments.forge.sources import inspect_source, snapshot_source
    from test_forge_hostprofiles import bind_candidate, prospective, rebind

    legacy = prospective(tmp_path)
    checkout = tmp_path / "worktree"
    (checkout / MODULE).unlink(missing_ok=True)
    manifest = inspect_source(checkout, set(legacy["source"]["files"]) - {MODULE})
    manifest["snapshot_path"] = str(snapshot_source(checkout, tmp_path / "legacy-queue", manifest))
    legacy["source"] = manifest
    bind_candidate(legacy)
    for declaration in legacy["tasks"].values():
        del declaration["execution"]["initializer"]
    from experiments.forge.views import task_evaluation_fingerprint
    for job in legacy["jobs"]:
        members = job.get("task_ids", [job["task_id"]])
        job["science"].update(candidate_revision=legacy["candidate_revision"],
                             execution={name: task_execution_fingerprint(legacy["tasks"][name]) for name in members},
                             evaluation={name: task_evaluation_fingerprint(legacy["tasks"][name]) for name in members})
        job["science"].pop("task_initializers", None)
        job["compatibility_key"] = stable_hash(job["science"])
    validate_request_host_profiles(legacy)
    assert all("initializer" not in declaration["execution"] for declaration in legacy["tasks"].values())

    (tmp_path / "current").mkdir()
    current = prospective(tmp_path / "current")
    checkout = tmp_path / "current" / "worktree"
    path = checkout / MODULE
    path.write_bytes((ROOT / MODULE).read_bytes())
    manifest = inspect_source(checkout, set(current["source"]["files"]) | {MODULE})
    manifest["snapshot_path"] = str(snapshot_source(checkout, tmp_path / "new-queue", manifest))
    current["source"] = manifest
    bind_candidate(current)
    rebind(current)
    for job in current["jobs"]:
        job["science"].pop("task_initializers", None)
        job["compatibility_key"] = stable_hash(job["science"])
    with pytest.raises(ValueError, match="experiment-owned task policies"):
        validate_request_host_profiles(current)
    for job in current["jobs"]:
        job["science"]["task_initializers"] = {name: task_initializer(current["tasks"][name], current["candidate"])
                                               for name in job.get("task_ids", [job["task_id"]])}
        job["compatibility_key"] = stable_hash(job["science"])
    validate_request_host_profiles(current)


def test_frozen_plain_native_continuation_checks_both_jobs_and_cached_task_changes(tmp_path):
    from experiments.forge.hostprofiles import validate_request_host_profiles
    from test_forge_hostprofiles import rebind

    current = current_request(tmp_path, [task("grid100"), task("grid100_14k")])
    validate_request_host_profiles(current)
    current["tasks"]["grid100_14k"]["execution"]["resources"]["num_particles"] = 12
    rebind(current)
    with pytest.raises(ValueError, match="changes task-owned resources"):
        validate_request_host_profiles(current)


def test_frozen_behavior_boundary_rejects_cached_initializer_or_host_objective_override(tmp_path):
    from experiments.forge.hostprofiles import validate_request_host_profiles
    from test_forge_hostprofiles import bind_candidate, rebind

    current = current_request(tmp_path, [task("two_pole")])
    validate_request_host_profiles(current)
    assert not current["tasks"]["two_pole"]["preflight_blockers"]
    current["candidate"]["initializer"] = "supplied"
    with pytest.raises(ValueError, match="candidate initializer conflicts"):
        validate_request_host_profiles(current)
    del current["candidate"]["initializer"]
    current["candidate"]["recipe_overrides"]["prior_reg"] = .123
    bind_candidate(current)
    rebind(current)
    validate_request_host_profiles(current)
    # The recipe owns prior penalties; the task still owns host objectives.
    current["candidate"]["recipe_overrides"]["reconstruction_weight"] = .123
    bind_candidate(current)
    rebind(current)
    with pytest.raises(ValueError, match="owned by the frozen host"):
        validate_request_host_profiles(current)
