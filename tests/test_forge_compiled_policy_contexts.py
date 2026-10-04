"""Compiled-request metadata controls; no scientific update or GPU admission."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import torch

from experiments.forge import contracts, planning, policy_cohorts as registry, views
from experiments.forge.api import CapabilityError, task_formulation_context
from experiments.forge import ae_routed_policy_contracts as ae
from experiments.forge import conditional_policy_contracts as conditional
from experiments.forge import multibank_policy_contracts as multibank
from experiments.forge import policy_contracts as independent
from experiments.forge import routed_policy_contracts as routed
from experiments.forge import word_joint_policy_contracts as word


ROOT = Path(__file__).resolve().parents[1]
NAMES = (*conditional.HOSTS, "unused", "cover", "ae", "word", "two_pole", "img_intensity2")
IDEA = "atlas-c6-observed-policy-current-v1"


@pytest.fixture(autouse=True)
def structural_cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    threads, state = torch.get_num_threads(), torch.get_rng_state().clone()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(state)


def definition(name):
    if name in conditional.HOSTS:
        module, task = conditional, conditional.make_conditional_variant(ROOT, name)
    elif name == "unused":
        module, task = routed, routed.make_unused_variant(ROOT)
    elif name == "cover":
        module, task = multibank, multibank.make_variant(ROOT)
    elif name == "ae":
        module, task = ae, ae.make_ae_variant(ROOT)
    elif name == "word":
        module, task = word, word.make_variant(ROOT)
    else:
        module = independent
        raw = (ROOT / f"configs/forge/tasks/{name}.json").read_bytes()
        parent = json.loads(raw)
        paths = module.REQUIRED_POLICY_SOURCES | set(parent["evaluation"].get("sources", {}))
        if name == "two_pole":
            paths |= {"benchmarks/locked_shared/two_pole.py", "benchmarks/legacy/locked_shared.py"}
        sources = {path: contracts.file_hash(ROOT / path) for path in paths}
        task = module._prospective_variant(parent, module._parent_record(parent, hashlib.sha256(raw).hexdigest()), sources)
    request = {"candidate": {"id": "software-only-compiled-context", "recipe_preset": "atlas",
                            "task_cohort": module.COHORT, "recipe_overrides": {"lr": .0053125, "prior_lr_mult": 1.5}},
               "protocol": {"seed": 0}}
    return request, task


def context(request, task, root=ROOT):
    return task_formulation_context(request["candidate"], task, request["protocol"], device="cpu", root=root)


def compiled(request, task):
    clean = context(request, task)
    annotated = deepcopy(task)
    annotated.update(preflight_blockers=["software-only inert planner annotation"],
                     field_ownership=deepcopy(clean.receipt()["field_ownership"]))
    return annotated, clean


@pytest.mark.parametrize("name", NAMES)
def test_only_actual_typed_compiler_annotations_are_inert_for_context_and_registry(name):
    request, task = definition(name)
    annotated, original = compiled(request, task)
    before, rng = deepcopy(annotated), torch.get_rng_state().clone()
    assert registry.policy_task_declaration(annotated) == task
    assert registry.validate_policy_task(annotated, root=ROOT) == registry.validate_policy_task(task, root=ROOT)
    assert registry.validate_policy_observation(annotated) == task["evaluation"]["policy_observation"]
    assert registry.policy_recipe_overrides(annotated) == task["execution"]["policy_recipe_overrides"]
    assert registry.policy_contract_blockers(annotated, original.recipe) == []
    restored = context(request, annotated)
    assert restored.policy_task == task
    assert restored.ownership_contract["task"] == task
    assert asdict(restored.recipe) == asdict(original.recipe)
    assert views.task_execution_fingerprint(annotated) == views.task_execution_fingerprint(task)
    assert views.task_evaluation_fingerprint(annotated) == views.task_evaluation_fingerprint(task)
    assert torch.equal(rng, torch.get_rng_state())
    assert annotated == before
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("name", NAMES)
def test_raw_json_validation_stays_exact_while_compiled_context_is_supported(name):
    request, task = definition(name)
    annotated, _ = compiled(request, task)
    assert registry.validate_policy_task(task, root=ROOT, allow_compiler_annotations=False)
    with pytest.raises(ValueError, match="raw policy declaration"):
        registry.validate_policy_task(annotated, root=ROOT, allow_compiler_annotations=False)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("change", ["unknown_top_level", "steps", "threshold", "sampler", "table_owner", "source"])
def test_compiler_annotations_never_hide_unknown_fields_or_scientific_drift(name, change):
    request, task = definition(name)
    annotated, _ = compiled(request, task)
    if change == "unknown_top_level": annotated["compiler_runtime_override"] = {"total_steps": 2}
    elif change == "steps": annotated["execution"]["steps"] -= 1
    elif change == "threshold": annotated["evaluation"]["thresholds"][0][2] += 1
    elif change == "sampler": annotated["evaluation"]["policy_observation"]["sampler"] = "forced_ema"
    elif change == "table_owner": annotated["execution"]["policy_contract"]["table_owner"] = "forged_owner"
    else: annotated["execution"]["policy_contract"]["sources"]["experiments/forge/policy_cohorts.py"] = "a" * 64
    with pytest.raises(ValueError):
        registry.validate_policy_task(annotated, root=ROOT)
    with pytest.raises(CapabilityError):
        context(request, annotated)


@pytest.mark.parametrize("change", ["blockers_dict", "blockers_non_string", "ownership_list", "ownership_empty",
                                   "wrong_task", "boolean_version", "foreign_key", "nonfinite"])
def test_annotation_types_are_not_an_arbitrary_metadata_escape(change):
    request, task = definition("word")
    annotated, _ = compiled(request, task)
    if change == "blockers_dict": annotated["preflight_blockers"] = {"override": "execute"}
    elif change == "blockers_non_string": annotated["preflight_blockers"] = [True]
    elif change == "ownership_list": annotated["field_ownership"] = []
    elif change == "ownership_empty": annotated["field_ownership"] = {}
    elif change == "wrong_task": annotated["field_ownership"]["task_id"] = "two_pole"
    elif change == "boolean_version": annotated["field_ownership"]["schema_version"] = True
    elif change == "foreign_key": annotated["field_ownership"]["runtime_override"] = {"lr": 9.}
    else: annotated["field_ownership"]["recipe_fields"]["lr"]["value"] = float("nan")
    with pytest.raises(ValueError, match="compiled"):
        registry.validate_policy_task(annotated, root=ROOT)


@pytest.fixture
def declared_source(tmp_path):
    """Actual parents/makers plus their actual source bytes, not a validator stub."""
    shutil.copytree(ROOT / "configs/forge", tmp_path / "configs/forge")
    parents = views.load_tasks(ROOT)
    for name in independent.PARENT_TASK_IDS:
        request, task = definition(name) if name in {"two_pole", "img_intensity2"} else (None, None)
        if task is None:
            parent = parents[name]
            raw = (ROOT / f"configs/forge/tasks/{name}.json").read_bytes()
            paths = independent.REQUIRED_POLICY_SOURCES | set(parent["evaluation"].get("sources", {}))
            sources = {path: contracts.file_hash(ROOT / path) for path in paths}
            task = independent._prospective_variant(parent, independent._parent_record(parent, hashlib.sha256(raw).hexdigest()), sources)
        path = tmp_path / f"configs/forge/task-variants/{independent.COHORT}/{task['id']}.json"
        path.write_text(json.dumps(task, indent=2) + "\n")
    paths = {independent.SELECTION_SOURCE}
    for parent in parents.values():
        paths.update(parent["evaluation"].get("sources", {}))
    for path in (tmp_path / "configs/forge/task-variants" / independent.COHORT).glob("*.json"):
        paths.update(json.loads(path.read_bytes())["execution"]["policy_contract"]["sources"])
    # Existing architecture/profile declarations can be extra snapshot inputs.
    from experiments.forge.hostprofiles import profile_source_paths
    for task in parents.values():
        paths.update(profile_source_paths(task))
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    return tmp_path


def test_real_resolve_idea_decision_bindings_accept_compiled_task_without_changing_source(declared_source, monkeypatch):
    """Exercise the reported planning failure with no Git, queue or model calls."""
    from experiments.forge.sources import source_files

    def inspect(root, extra):
        files = {path.relative_to(root).as_posix(): contracts.file_hash(path) for path in source_files(root, extra)}
        return {"schema_version": 1, "files": files, "digest": contracts.stable_hash(files), "origin_commit": None}

    monkeypatch.setattr(planning, "inspect_source", inspect)
    monkeypatch.setattr(planning, "compute_profile", lambda backend, model=None: {
        "backend": backend, "threads": 1, "model": "CPU structural software metadata", "deterministic": True})
    before = {path.relative_to(declared_source).as_posix(): path.read_bytes()
              for path in declared_source.rglob("*") if path.is_file()}
    result = planning.resolve_idea(declared_source, IDEA, execution_backend="cpu", freeze_source=False)
    assert len(result["tasks"]) == 26
    assert [sum(a["qualification_tier"] == tier for a in result["view"]["assignments"]) for tier in (1, 2, 3)] == [5, 19, 2]
    assert result["decision_review"]["status"] == "BLOCKED"  # The actual declaration is deliberately still draft.
    assert result["decision_review"]["qualification_input"] is False
    assert result["preflight_blockers"]
    for name, task in result["tasks"].items():
        assert "preflight_blockers" in task
        if "field_ownership" in task:
            assert task["field_ownership"]["task_id"] == name
        raw = json.loads((declared_source / f"configs/forge/task-variants/{independent.COHORT}/{name}.json").read_bytes())
        assert registry.policy_task_declaration(task) == raw
        assert views.task_execution_fingerprint(task) == views.task_execution_fingerprint(raw)
        assert views.task_evaluation_fingerprint(task) == views.task_evaluation_fingerprint(raw)
        source = f"configs/forge/tasks/{task['policy_parent']['id']}.json"
        assert result["source"]["files"][source] == task["policy_parent"]["task_sha256"]
    assert before == {path.relative_to(declared_source).as_posix(): path.read_bytes()
                      for path in declared_source.rglob("*") if path.is_file()}
    assert not torch.cuda.is_initialized()
