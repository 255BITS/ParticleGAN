"""Software controls for the scoped Atlas/E22 policy-cloud coverage cohorts.

Two scoring laws: clean (learned output noise removed from measured samples)
and as served (kept exactly as GANTrainer serves it). No qualification
experiment runs here. Variants are materialized in memory from the current
parent files and sources, so later source edits cannot make these controls
depend on stale committed pins.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from experiments.forge import attempt_coverage as runner
from experiments.forge import grouped_evidence as reuse
from experiments.forge.adapters import adapter_preflight
from experiments.forge.api import CapabilityError
from experiments.forge.artifacts import manifest_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.policy_coverage import (BLOCKED_HOSTS, COHORT, COVERAGE, DIAGNOSTIC_CONTINUATIONS, LABEL,
                                               LAW_INDEPENDENT, PARENTS, SERVED, SERVED_COHORT, SERVED_LABEL,
                                               SERVED_PARENTS, SERVED_VIEW, SOURCES, TIERS, VIEW, served_row_task,
                                               served_view_declaration, variant_id, view_declaration)
from experiments.forge.tier1_policy import (TIER1, is_policy_task, materialize, schedule_free_policy, validate,
                                            validate_evidence)
from experiments.forge.views import grade_result, load_tasks, task_evaluation_fingerprint, task_execution_fingerprint

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("atlas", "e22")
COMPATIBLE = tuple(name for name in PARENTS if name not in BLOCKED_HOSTS)


def parent(name):
    return json.loads((ROOT / "configs/forge/tasks" / (name + ".json")).read_text())


def candidate(name):
    return json.loads((ROOT / "configs/forge/ideas" / (name + ".json")).read_text())


@pytest.fixture(scope="module")
def variants():
    return {name: materialize(parent(name), ROOT, COVERAGE) for name in PARENTS}


@pytest.fixture(scope="module")
def served():
    return {name: materialize(parent(name), ROOT, SERVED) for name in SERVED_PARENTS}


def _undo(variant, original):
    """Reverse exactly the declared transformation; anything left over changed."""
    task = deepcopy(variant)
    for key in ("task_cohort", "policy_parent"):
        task.pop(key)
    if "task_cohort" in original:
        task["task_cohort"] = original["task_cohort"]
    task["id"] = original["id"]
    task["requires_capabilities"] = deepcopy(original["requires_capabilities"])
    for dependency in task["dependencies"]:
        assert dependency["task"].endswith(COVERAGE.suffix)
        dependency["task"] = dependency["task"].removesuffix(COVERAGE.suffix)
    execution = task["execution"]
    for key in ("policy_parent_definition", "policy_contract", "policy_clock_horizon"):
        execution.pop(key, None)
    if "continuation_of" in execution:
        assert execution["continuation_of"].endswith(COVERAGE.suffix)
        execution["continuation_of"] = execution["continuation_of"].removesuffix(COVERAGE.suffix)
    for key in ("prior", "device", "resources"):
        if key in original["execution"]:
            execution[key] = deepcopy(original["execution"][key])
        else:
            execution.pop(key, None)
    task["resources"] = deepcopy(original["resources"])
    evaluation = task["evaluation"]
    added = set(evaluation["sources"]) - set(original["evaluation"].get("sources", {}))
    assert added <= set(SOURCES) | {"configs/forge/tasks/" + original["id"] + ".json"}
    for key, digest in original["evaluation"].get("sources", {}).items():
        if key not in SOURCES:
            assert evaluation["sources"][key] == digest  # Parent evaluator pins never change.
    evaluation["sources"] = deepcopy(original["evaluation"].get("sources", {}))
    if "sources" not in original["evaluation"]:
        evaluation.pop("sources")
    evaluation.pop("policy_observation")
    evaluation["scoring_weights"] = original["evaluation"]["scoring_weights"]
    evaluation["sampling_law"] = original["evaluation"]["sampling_law"]
    return task


@pytest.mark.parametrize("name", PARENTS)
def test_transformation_changes_only_declared_fields(name, variants):
    original, variant = parent(name), variants[name]
    assert _undo(variant, original) == original
    execution, evaluation = variant["execution"], variant["evaluation"]
    assert variant["id"] == variant_id(name) and variant["task_cohort"] == COHORT
    assert execution["policy_parent_definition"] == original
    prior = execution["prior"]
    assert (prior["kind"], prior["sigma"], prior["standardize"]) == ("particle_cloud", 0., False)
    # A recipe-owned prior leaves learnability to Recipe.prior_update; a legacy
    # task-owned prior keeps its explicit learned table.
    recipe_owned = original["execution"].get("prior_contract") == "recipe_owned_v1"
    assert ("learnable" not in prior) if recipe_owned else (prior["learnable"] is True)
    assert prior["exception_reason"].strip()
    contract = execution["policy_contract"]
    assert contract["recipe_overrides"] == {} and contract["original_parent_credit"] is False
    assert contract["schedule"] == "preserve_resolved_recipe_none" and contract["cohort"] == COHORT
    assert set(contract["sources"]) == set(SOURCES)
    assert evaluation["scoring_weights"] == "state_selected"
    assert evaluation["sampling_law"] == "policy_cloud_" + original["evaluation"]["sampling_law"]
    observation = evaluation["policy_observation"]
    assert observation["output_noise"] is False and observation["eval_streams"] == "forge-rng-v1_isolated"
    assert observation["sampler"] == ("served_snapshot" if name in COVERAGE.parameter_parents else "GANTrainer.sample")
    assert ("policy_clock_horizon" in execution) == (original["adapter"] == "clockfree_audit")
    # Budget, timeout, cadence and gates are inherited unchanged.
    assert execution["steps"] == original["execution"]["steps"]
    assert variant["resources"]["timeout_seconds"] == original["resources"]["timeout_seconds"]
    assert variant["resources"]["gpu_memory_mb"] == max(2048, original["resources"]["gpu_memory_mb"])
    for key in ("thresholds", "observations", "eval_interval", "conditions", "start_step"):
        assert evaluation.get(key) == original["evaluation"].get(key)
    caps = variant["requires_capabilities"]
    assert "live_sampling" not in caps and "mog_prior" not in caps
    assert {"policy_controls", "policy_serving"} <= set(caps)
    assert validate(variant, root=ROOT) == original


def test_original_tier1_cohort_keeps_its_exact_transformation():
    for name in ("two_pole", "clockfree_audit", "gaussian1d_acquisition"):
        variant = materialize(parent(name), ROOT)
        assert variant["task_cohort"] == TIER1.id and variant["dependencies"] == parent(name)["dependencies"]
        assert variant["evaluation"]["sampling_law"].startswith("tier1_selected_")
    committed = sorted((ROOT / TIER1.directory).glob("*.json"))
    assert len(committed) == 7 and all(validate(read_json(path)) for path in committed)


def test_committed_declarations_view_and_tiers_cover_all_selected_tests():
    tasks = load_tasks(ROOT)
    committed = {task["policy_parent"]["id"]: task for task in tasks.values() if task.get("task_cohort") == COHORT}
    assert set(committed) == set(PARENTS) and len(PARENTS) == 35
    assert {tier: len(names) for tier, names in TIERS.items()} == {1: 7, 2: 21, 3: 7}
    assert len(BLOCKED_HOSTS) == 9 and len(COMPATIBLE) == 26
    view = read_json(ROOT / "configs/forge/views" / (VIEW + ".json"))
    assert view == view_declaration() and view["reporting"] == {"family_totals": False, "families": ["atlas", "e22"]}
    assignments = {row["task"]: row for row in view["assignments"]}
    assert all(row["importance"] == "required" for row in assignments.values())
    for tier, names in TIERS.items():
        assert {assignments[variant_id(name)]["qualification_tier"] for name in names} == {tier}
    # The as-served cohort: 23 own variants; its view shares the 12 law-independent clean declarations.
    as_served = {task["policy_parent"]["id"]: task for task in tasks.values() if task.get("task_cohort") == SERVED_COHORT}
    assert set(as_served) == set(SERVED_PARENTS) and len(SERVED_PARENTS) == 23 and len(LAW_INDEPENDENT) == 12
    assert set(SERVED_PARENTS) | set(LAW_INDEPENDENT) == set(PARENTS)
    served_view = read_json(ROOT / "configs/forge/views" / (SERVED_VIEW + ".json"))
    assert served_view == served_view_declaration() and served_view["reporting"] == view["reporting"]
    rows = {row["task"]: row for row in served_view["assignments"]}
    for tier, names in TIERS.items():
        assert {rows[served_row_task(name)]["qualification_tier"] for name in names} == {tier}
    assert {name for name in rows if tasks[name]["task_cohort"] == COHORT} == {variant_id(n) for n in LAW_INDEPENDENT}
    # Every selected-view requirement keeps its recorded tier.
    publication = read_json(ROOT / "reports/forge/technique-inventory.json")
    views = {v["id"]: v for v in publication["family_progress"]["views"]}
    family = next(f for f in publication["family_progress"]["families"] if f["id"] == "atlas")
    cohort = next(c for c in family["cohorts"] if c["anchor"] == family["current_cohort_anchor"])
    selected = {(a["task"], a["qualification_tier"]) for v in cohort["views"] for a in views[v["id"]]["assignments"]
                if a["importance"] == "required"}
    assert selected == {(name, tier) for tier, names in TIERS.items() for name in names}


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("name", PARENTS)
def test_preflight_runs_compatible_variants_and_blocks_only_declared_hosts(family, name, variants):
    reasons = adapter_preflight(variants[name], candidate(family), root=ROOT)
    if name in BLOCKED_HOSTS:
        assert reasons == [variant_id(name) + ": " + BLOCKED_HOSTS[name]]
    else:
        assert reasons == []


@pytest.mark.parametrize("name", sorted(BLOCKED_HOSTS))
def test_blocked_hosts_refuse_before_any_host_setup(name, variants, tmp_path):
    from experiments.forge.adapters import _dispatch_task
    request = {"candidate": candidate("atlas"), "protocol": {"seed": 0}, "tasks": {variant_id(name): variants[name]}}
    with pytest.raises(CapabilityError) as refused:
        _dispatch_task(request, {"task_id": variant_id(name)}, tmp_path, "cpu")
    assert BLOCKED_HOSTS[name] in str(refused.value)
    assert not any(tmp_path.iterdir())


def test_routing_reaches_each_cohort_adapter(variants, served, tmp_path, monkeypatch):
    from experiments.forge import adapters, policy_behavior_adapters, policy_gaussian
    calls = []
    def recorder(label):
        def call(request, task, output, device, **kwargs):
            calls.append((label, task["policy_parent"]["id"], kwargs))
            return {"cost": {}}
        return call
    monkeypatch.setattr(adapters, "_ring", recorder("ring"))
    monkeypatch.setattr(adapters, "_native", recorder("native"))
    monkeypatch.setattr(adapters, "_vector", recorder("vector"))
    monkeypatch.setattr(adapters, "_image", recorder("image"))
    monkeypatch.setattr(policy_behavior_adapters, "run_behavior", recorder("policy_behavior"))
    monkeypatch.setattr(policy_gaussian, "run_gaussian", recorder("policy_gaussian"))
    expected = {"mode_hold": "ring", "ring_hold": "ring", "two_pole": "policy_behavior",
                "gaussian1d_smoke": "policy_gaussian", "gaussian1d_stability": "policy_gaussian",
                "grid100": "native", "grid100_14k": "native", "vector_spiral": "vector",
                "ring16_acquisition": "vector", "img_bars4": "image"}
    for name, label in expected.items():
        for cards in (variants, served):
            if name not in cards:
                assert name in LAW_INDEPENDENT  # Shared tests run once, in the clean cohort.
                continue
            task = cards[name]
            job = {"task_id": task["id"], "diagnostic_continuation": name in DIAGNOSTIC_CONTINUATIONS}
            adapters._dispatch_task({"candidate": candidate("e22"), "tasks": {task["id"]: task}}, job, tmp_path, "cpu")
            assert calls[-1][:2] == (label, name)
            if name in {"gaussian1d_stability", "grid100_14k"}:
                assert calls[-1][2]["diagnostic"] is True


def test_policy_gaussian_runs_unchanged_evaluator_code_with_parent_constants(variants, served):
    from experiments.forge import gaussian_tasks, policy_gaussian
    for name in ("gaussian1d_smoke", "gaussian1d_stability"):
        for task in (variants[name], served[name]):
            policy_gaussian.validate_task(task)
            with pytest.raises(ValueError, match="fixed by its evaluator"):
                gaussian_tasks.validate_task(task)
    # The evaluator's own code objects execute; only the validator and the
    # constructor's measurement-law binding are substituted in their scope.
    assert policy_gaussian._ORIGINAL_RUN.__code__ is gaussian_tasks.run_gaussian.__code__
    assert policy_gaussian._ORIGINAL_BUILD.__code__ is gaussian_tasks.build.__code__
    assert policy_gaussian.grade.__code__ is gaussian_tasks.grade.__code__
    assert policy_gaussian._SCOPE["build"] is policy_gaussian.build
    assert policy_gaussian._SCOPE["validate_task"] is policy_gaussian.validate_task
    assert policy_gaussian.build.__kwdefaults__ == gaussian_tasks.build.__kwdefaults__
    with pytest.raises(ValueError, match="registered scoped variant"):
        policy_gaussian.validate_task(parent("gaussian1d_smoke"))


def _policy_evidence(task, final):
    declaration = task["evaluation"]["policy_observation"]
    noise = declaration["output_noise"]
    draw = {"role": "measured", "model": "live", "n": 4096, "completed_steps": final,
            "output_noise": noise, "output_sigma": .029}
    evidence = {"policy_controls": {"cohort": task["task_cohort"], "implementation_observed": True,
                                    "requested_owners_bound": True, "lifecycle": {"complete": True},
                                    "completed_steps": final},
                "policy_observations": [{**declaration, "observed": True, "policy_owner": "particlegan.UpdatePolicy",
                                         "measured_draws": [draw]}],
                "policy_purity": [{"pure": True}]}
    if noise:
        evidence["served_output_noise"] = {"law": task["evaluation"]["eval_output_noise"], "output_noise": True,
                                           "draws": 1, "draws_without_output_noise": 0}
    return evidence


def test_uninterrupted_endurance_attests_its_own_early_stop(variants):
    task = variants["ring_hold"]
    evidence = {**_policy_evidence(task, 2900), "dense": [{"step": s} for s in range(1201, 2901)]}
    assert validate_evidence(task, evidence) is None
    evidence["policy_controls"]["completed_steps"] = 7500
    assert validate_evidence(task, evidence)["status"] == "INCOMPLETE"
    ordinary = variants["vector_spiral"]
    assert validate_evidence(ordinary, _policy_evidence(ordinary, 1600)) is None
    assert validate_evidence(ordinary, _policy_evidence(ordinary, 800))["status"] == "INCOMPLETE"
    assert validate_evidence(ordinary, {**_policy_evidence(ordinary, 1600),
                                        "policy_controls": {"cohort": TIER1.id}})["status"] == "INVALID"


def test_schedule_free_helpers_bind_only_registered_policy_variants(variants):
    assert schedule_free_policy(variants["target_shift_recovery"]) and schedule_free_policy(variants["grid100_14k"])
    assert not schedule_free_policy(parent("target_shift_recovery"))
    assert is_policy_task(variants["two_pole"]) and not is_policy_task(parent("two_pole"))


def test_policy_two_pole_receipt_attests_recipe_owned_prior(variants, tmp_path):
    from experiments.forge.policy_behavior_adapters import run_behavior
    task = variants["two_pole"]
    request = {"candidate": candidate("e22"), "protocol": {"seed": 0}}
    result = run_behavior(request, task, tmp_path, "cpu")
    proof = result["evidence"]["prior_policy"]
    assert proof["update"] == "learned" and proof["trainable_locations"] is True and proof["l2"] == 0
    assert result["evidence"]["host"]["particle_l2"] == .02
    grade = grade_result(task, result)
    assert grade["status"] in {"PASS", "FAIL"}, grade


def _group(variants, tmp_path, cohort=COVERAGE):
    cards = {name: deepcopy(variants[name]) for name in ("ring_hold", "ring_extension")}
    source = tmp_path / "frozen"
    files = {}
    for card in cards.values():
        for relative in list(card["evaluation"]["sources"]) + ["experiments/forge/views.py"]:
            if relative in files:
                continue
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            files[relative] = file_hash(target)
    manifest = dict(files=files, digest=stable_hash(files))
    names = [variant_id("ring_hold", cohort), variant_id("ring_extension", cohort)]
    cards = {card["id"]: card for card in cards.values()}
    execution = {name: task_execution_fingerprint(cards[name]) for name in names}
    evaluation = {name: task_evaluation_fingerprint(cards[name]) for name in names}
    science = dict(execution=execution, evaluation=evaluation)
    request = dict(source=manifest, candidate_revision="recipe", tasks=cards, jobs=[
        dict(task_ids=names, compatibility_key=stable_hash(science), science=science)])
    request_path = tmp_path / "request.json"
    atomic_json(request_path, request)
    directory = tmp_path / "producer"
    artifact = directory / "provenance/state.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"retained fixture state, never loaded or trained")
    checkpoint = dict(path="state.pt", sha256=file_hash(artifact), completed_steps=2900,
                      artifact_root=str(artifact.parent), artifact_manifest=manifest_artifacts(artifact.parent))
    producer = cards[names[0]]
    raw = dict(cost=dict(completed_steps=2900), evidence={
        "sampling_contract_version": 1, "sampling_law": producer["evaluation"]["sampling_law"],
        "eval_output_noise": producer["evaluation"]["eval_output_noise"], "scoring_weights": "state_selected",
        "prior_policy": {"contract": "recipe_owned_v1", "update": "learned", "regularizer": "vicreg", "weight": 0.,
                         "target_std": 1., "eps": 1e-4, "l2": 0., "applicability": "latent_prior",
                         "trainable_locations": True},
        **_policy_evidence(producer, 2900),
        "dense": [dict(step=s, modes=8, hq=.95) for s in range(1201, 2901)],
        "continuity": dict(mode="uninterrupted", run_id="one-run", resume_count=0,
                           original_schedule_horizon=None, max_total_steps=7500),
        "provenance_checkpoint": checkpoint})
    atomic_json(directory / "raw.json", raw)
    atomic_json(directory / "result.json", dict(status="PASS", runtime={"fixture": True},
                raw_sha256=file_hash(directory / "raw.json")))
    jobs = []
    for name in names:
        job = dict(family="policy-cloud-fixture", candidate_id="candidate", source_digest=manifest["digest"],
                   candidate_revision="recipe", task=name, execution_sha256=execution[name],
                   evaluation_sha256=evaluation[name], request_path=str(request_path),
                   request_sha256=file_hash(request_path), source_root=str(source),
                   dependencies=cards[name]["dependencies"], attempted=True)
        job["key"] = stable_hash(reuse._identity(job))
        jobs.append(job)
    parent_job, child = jobs
    parent_job.update(status="PASS", origin="new_attempt", directory=str(directory), attempt_id="producer",
                      result_sha256=file_hash(directory / "result.json"))
    child.update(status="BLOCKED", block_kind="prerequisite")
    atomic_json(directory / "request.json", parent_job)
    return parent_job, child


def test_as_served_ring_group_regrades_its_own_saved_extension(served, tmp_path):
    parent_job, child = _group(served, tmp_path, SERVED)
    certificate = reuse.grade_saved_group(parent_job, child)
    assert certificate["grade"]["status"] == "PASS" and certificate["task"] == variant_id("ring_extension", SERVED)
    # A clean-law receipt cannot stand in for the as-served producer's draws.
    raw_path = Path(parent_job["directory"]) / "raw.json"
    raw = read_json(raw_path)
    for row in raw["evidence"]["policy_observations"]:
        row["measured_draws"][0]["output_noise"] = False
    atomic_json(raw_path, raw)
    result = read_json(Path(parent_job["directory"]) / "result.json")
    atomic_json(Path(parent_job["directory"]) / "result.json", dict(result, raw_sha256=file_hash(raw_path)))
    parent_job["result_sha256"] = file_hash(Path(parent_job["directory"]) / "result.json")
    with pytest.raises(ValueError, match="original gate"):
        reuse.grade_saved_group(parent_job, child)


def test_scoped_ring_group_regrades_saved_extension_without_training(variants, tmp_path):
    parent_job, child = _group(variants, tmp_path)
    certificate = reuse.grade_saved_group(parent_job, child)
    assert certificate["grade"]["status"] == "PASS"
    assert certificate["grade"]["metrics"]["extension_passing"] == 300
    # The scoped pair still needs its unchanged original parents to form one group.
    tampered = deepcopy(variants["ring_extension"])
    tampered["execution"]["policy_parent_definition"]["evaluation"]["hold_budget"] = 1
    with pytest.raises(ValueError):
        reuse._check_group(variants["ring_hold"], tampered)
    with pytest.raises(ValueError, match="cannot pool policy cohorts"):
        reuse.grade_saved_group(*_mixed_cohort(parent_job, child, tmp_path))


def _mixed_cohort(parent_job, child, tmp_path):
    request = read_json(Path(parent_job["request_path"]))
    request["tasks"][child["task"]]["task_cohort"] = TIER1.id
    path = tmp_path / "mixed-request.json"
    atomic_json(path, request)
    jobs = []
    for job in (parent_job, child):
        job = dict(job, request_path=str(path), request_sha256=file_hash(path),
                   execution_sha256=task_execution_fingerprint(request["tasks"][job["task"]]),
                   evaluation_sha256=task_evaluation_fingerprint(request["tasks"][job["task"]]))
        job["key"] = stable_hash(reuse._identity(job))
        jobs.append(job)
    return jobs


def test_intake_prepares_both_laws_once_and_never_writes_the_repository_report(tmp_path):
    repository, directory = tmp_path / "repository", tmp_path / "run"
    sentinel = repository / runner.REPORT
    atomic_json(sentinel, {"records": ["unchanged"]})
    state = runner.prepare(repository, ROOT, directory, intake="policy-cloud")
    assert read_json(sentinel) == {"records": ["unchanged"]}
    jobs = list(state["jobs"].values())
    # 35 clean jobs plus 23 as-served jobs per family; 12 law-independent tests run once.
    assert state["intake"] == "policy-cloud" and len(jobs) == 2 * (35 + 23)
    assert {job["family"] for job in jobs} == {"policy-cloud-atlas", "policy-cloud-e22",
                                               "policy-cloud-as-served-atlas", "policy-cloud-as-served-e22"}
    assert len({job["key"] for job in jobs}) == len(jobs) and len({(job["family"], job["task"]) for job in jobs}) == len(jobs)
    for family in FAMILIES:
        mine = [job for job in jobs if job["owning_family"] == family]
        clean = [job for job in mine if job["cohort_id"] == COHORT]
        as_served = [job for job in mine if job["cohort_id"] == SERVED_COHORT]
        assert sorted(job["parent_task"] for job in clean) == sorted(PARENTS)
        assert sorted(job["parent_task"] for job in as_served) == sorted(SERVED_PARENTS)
        clean_scope, = {job["cohort_scope"] for job in clean}
        served_scope, = {job["cohort_scope"] for job in as_served}
        assert clean_scope != served_scope
        assert clean_scope == stable_hash([family, clean[0]["source_digest"], clean[0]["candidate_revision"], COHORT])
        assert {(j["source_digest"], j["candidate_revision"]) for j in mine} == {(clean[0]["source_digest"],
                                                                                 clean[0]["candidate_revision"])}
        for job in clean:
            shared = job["parent_task"] in LAW_INDEPENDENT
            assert job["measurement_law"] == ("shared" if shared else "clean") and job["label"] == LABEL
            assert job.get("shared_scopes") == ([served_scope] if shared else None)
            assert job["task"] == variant_id(job["parent_task"])
        for job in as_served:
            assert job["measurement_law"] == "as_served" and job["label"] == SERVED_LABEL
            assert job["task"] == variant_id(job["parent_task"], SERVED) and "shared_scopes" not in job
            for dependency in job["dependencies"]:
                # Checkpoints come from the same-law producer; the only shared
                # dependency is a gate on the clean clock audit.
                expected = SERVED if dependency["kind"] == "checkpoint" else (
                    COVERAGE if dependency["task"].startswith("clockfree_audit") else SERVED)
                assert dependency["task"].endswith(expected.suffix)
    for job in jobs:
        assert job["status"] == "PENDING" and job["diagnostic"] is True
        assert job["reporting_tier"] == next(t for t, names in TIERS.items() if job["parent_task"] in names)
        assert job["diagnostic_continuation"] == (job["parent_task"] in DIAGNOSTIC_CONTINUATIONS)
        assert job.get("exclusive_gpu", False) == (job["parent_task"] in
                                                   {"gaussian1d_smoke", "gaussian1d_stability", "ring16_acquisition"})
        if job["parent_task"] in BLOCKED_HOSTS:
            assert job["dependencies"] == []
    published = read_json(directory / "attempt-coverage.json")
    assert published["publication"] == "state_directory_only" and len(published["records"]) == len(jobs)
    assert all({"cohort_scope", "cohort_id", "measurement_law", "owning_family", "reporting_tier", "parent_task"}
               <= set(r) for r in published["records"])
    # A second preparation keeps the same jobs; the state cannot change intake.
    assert runner.prepare(repository, ROOT, directory)["jobs"].keys() == state["jobs"].keys()
    with pytest.raises(ValueError, match="belongs to the policy-cloud intake"):
        runner.prepare(repository, ROOT, directory, intake="selected")
    with pytest.raises(ValueError, match="state directory"):
        runner.publish(repository, state)
    with pytest.raises(ValueError, match="never refresh"):
        runner.drain(repository, directory, gpus=["0", "1"], refresh=True)
    # A clean-only (first-build) state directory cannot silently gain a law.
    old = tmp_path / "old"
    atomic_json(old / "policy-cloud.json", {"schema_version": 1, "cohort": COHORT, "view": VIEW,
                                            "families": list(FAMILIES), "source_root": str(ROOT)})
    atomic_json(old / "state.json", {"intake": "policy-cloud", "jobs": {}})
    with pytest.raises(ValueError, match="clean-only intake"):
        runner.prepare(repository, ROOT, old)


def test_selected_state_directory_cannot_switch_to_the_cohort(tmp_path):
    atomic_json(tmp_path / "state.json", {"jobs": {"key": {"family": "atlas", "task": "two_pole"}}})
    with pytest.raises(ValueError, match="belongs to the selected intake"):
        runner.prepare(tmp_path / "repository", ROOT, tmp_path, intake="policy-cloud")


def test_short_limit_jobs_run_alone_on_their_gpu(tmp_path, monkeypatch):
    root, directory = tmp_path / "repo", tmp_path / "runs"
    jobs = {str(i): dict(key=str(i), family="fixture", task=f"task{i}", status="PENDING",
                         resources=dict(gpus=1, gpu_memory_mb=1), dependencies=[]) for i in range(8)}
    for key in ("0", "5"):
        jobs[key].update(exclusive_gpu=True, required_gpu="1")
    atomic_json(directory / "state.json", dict(jobs=jobs))
    monkeypatch.setattr(runner, "device_slots", lambda *a, **kw: [
        dict(device=gpu, slot=slot, memory_mb=100) for gpu in ("0", "1") for slot in (0, 1)])
    monkeypatch.setattr(runner, "process_identity", lambda pid: str(pid))
    monkeypatch.setattr(runner.time, "sleep", lambda delay: None)
    running = []

    class Process:
        def __init__(self, command, **kwargs):
            path = Path(command[-1]).parent
            self.path, self.pid = path, 100 + len(running)
            running.append(self)
            self.polls = 0

        def poll(self):
            self.polls += 1
            if self.polls == 2:
                atomic_json(self.path / "result.json", dict(status="BLOCKED", attempted=True, reason="fixture"))
            return 0 if self.polls >= 2 else None

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    runner.drain(root, directory, gpus=["0", "1"])
    occupancy, exclusive = {}, set()
    for event in map(json.loads, (directory / "events.jsonl").read_text().splitlines()):
        gpu = event["gpu"]
        if event["event"] == "started":
            occupancy[gpu] = occupancy.get(gpu, 0) + 1
            if event["task"] in {"task0", "task5"}:
                assert gpu == "1" and occupancy[gpu] == 1
                exclusive.add(gpu)
            else:
                assert gpu not in exclusive
        else:
            occupancy[gpu] -= 1
            if event["task"] in {"task0", "task5"}:
                exclusive.discard(gpu)
    assert all(job["status"] == "BLOCKED" for job in read_json(directory / "state.json")["jobs"].values())


def test_cohort_records_project_one_row_per_law_and_family_with_shared_verdicts(tmp_path):
    from experiments.forge.research_tiers import _attempt_registry, policy_cohort_rows
    _attempt_registry.cache_clear()
    records = []
    for family in FAMILIES:
        scopes = {cohort: stable_hash([family, "source", "recipe", cohort]) for cohort in (COHORT, SERVED_COHORT)}
        common = dict(owning_family=family, source_digest="source", candidate_revision="recipe", attempted=True,
                      diagnostic=True)
        for tier, names in sorted(TIERS.items()):
            for name in names:
                status = ("BLOCKED" if name in BLOCKED_HOSTS else "PASS" if name == "img_stripes2" else "FAIL")
                shared = name in LAW_INDEPENDENT
                records.append(dict(common, cohort_scope=scopes[COHORT], cohort_id=COHORT, family="policy-cloud-" + family,
                                    task=variant_id(name), parent_task=name, reporting_tier=tier, status=status,
                                    label=LABEL, measurement_law="shared" if shared else "clean",
                                    **({"shared_scopes": [scopes[SERVED_COHORT]]} if shared else {}),
                                    **({"block_kind": "incompatible"} if status == "BLOCKED" else {})))
                if not shared:
                    # The as-served law passes one more (width-limited) test here.
                    served_status = "PASS" if name in {"img_stripes2", "vector_two_broad"} else "FAIL"
                    records.append(dict(common, cohort_scope=scopes[SERVED_COHORT], cohort_id=SERVED_COHORT,
                                        family="policy-cloud-as-served-" + family, task=variant_id(name, SERVED),
                                        parent_task=name, reporting_tier=tier, status=served_status,
                                        label=SERVED_LABEL, measurement_law="as_served"))
    atomic_json(tmp_path / runner.REPORT, {"records": records})
    rows = policy_cohort_rows(tmp_path, tmp_path / "board.md")
    assert set(rows) == set(FAMILIES)
    for family in FAMILIES:
        clean, as_served = rows[family]
        assert LABEL in clean and SERVED_LABEL in as_served
        # Both rows cover all 35 tests: the shared verdicts (two-pole, clock
        # audits, 9 incompatible hosts) appear in each row exactly once.
        assert "0/7 (3 incompatible)" in clean and "1/21 (6 incompatible)" in clean and "1/35 (9 incompatible)" in clean
        assert "0/7 (3 incompatible)" in as_served and "2/21 (6 incompatible)" in as_served
        assert "2/35 (9 incompatible)" in as_served
    records.append(dict(records[0], parent_task="duplicate", source_digest="other"))
    atomic_json(tmp_path / runner.REPORT, {"records": records})
    _attempt_registry.cache_clear()
    with pytest.raises(ValueError, match="cannot pool"):
        policy_cohort_rows(tmp_path, tmp_path / "board.md")
    records.pop()
    records.append(dict(records[0], shared_scopes=["no-such-row"]))
    records.pop(0)
    atomic_json(tmp_path / runner.REPORT, {"records": records})
    _attempt_registry.cache_clear()
    with pytest.raises(ValueError, match="no records of its own"):
        policy_cohort_rows(tmp_path, tmp_path / "board.md")
    _attempt_registry.cache_clear()


def test_attempt_media_renders_saved_images_and_paired_curves_without_sampling(tmp_path, variants):
    import torch
    from experiments.forge.tier1_media import render
    task = variants["img_bars4"]
    records = [{"step": step, "samples": torch.rand(32, 1, 8, 8), "targets": torch.rand(4, 1, 8, 8),
                "metrics": {"modes": 1, "hq": .5}} for step in (25, 50)]
    path = tmp_path / "observed-images.pt"
    torch.save(records, path)
    evidence = {"observations": [{"step": r["step"], "modes": 1, "hq": .5} for r in records],
                "saved_observer_outputs": {"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size},
                "scoring_weights": "state_selected", "sampling_law": task["evaluation"]["sampling_law"]}
    receipt = render(task, {"evidence": evidence, "gate_status": "FAIL"}, tmp_path, tmp_path / "images.gif")
    assert receipt["observation_count"] == 2 and receipt["sampling_draws_added"] == 0
    task = variants["target_shift_recovery"]
    curve = [{"step": step, "modes": 8, "hq": .95} for step in range(10, 101, 10)]
    evidence = {"active": {"diagnostic": curve}, "frozen": {"diagnostic": curve},
                "scoring_weights": "state_selected", "sampling_law": task["evaluation"]["sampling_law"]}
    receipt = render(task, {"evidence": evidence, "gate_status": "FAIL"}, tmp_path, tmp_path / "paired.gif")
    assert receipt["observation_count"] == 10 and (tmp_path / "paired.gif").is_file()


# --- As-served scoring law -------------------------------------------------

@pytest.mark.parametrize("name", SERVED_PARENTS)
def test_as_served_variant_differs_from_its_clean_variant_only_in_the_declared_measurement(name, variants, served):
    from experiments.forge.sampling import AS_SERVED_LAWS, SERVED_LEARNED_NOISE
    clean, task = variants[name], deepcopy(served[name])
    assert task["id"] == variant_id(name, SERVED) and task["task_cohort"] == SERVED_COHORT
    evaluation, observation = task["evaluation"], task["evaluation"]["policy_observation"]
    # The single declared difference, and the law names that follow from it.
    assert observation.pop("output_noise") is True and clean["evaluation"]["policy_observation"]["output_noise"] is False
    assert observation.pop("output_sigma").startswith("GANTrainer.sample(output_noise=True)")
    assert evaluation["eval_output_noise"] == SERVED_LEARNED_NOISE
    original_law = parent(name)["evaluation"]["sampling_law"]
    assert evaluation["sampling_law"] == "policy_cloud_as_served_" + AS_SERVED_LAWS[original_law]
    assert evaluation["sampling_law"] not in {original_law, clean["evaluation"]["sampling_law"]}
    # Undo exactly those fields and the cohort identity: the clean variant remains.
    observation["output_noise"] = False
    evaluation["eval_output_noise"] = clean["evaluation"]["eval_output_noise"]
    evaluation["sampling_law"] = clean["evaluation"]["sampling_law"]
    task.update(id=clean["id"], task_cohort=COHORT)
    task["execution"]["policy_contract"]["cohort"] = COHORT
    for dependency in task["dependencies"]:
        if dependency["task"].endswith(SERVED.suffix):
            dependency["task"] = dependency["task"].removesuffix(SERVED.suffix) + COVERAGE.suffix
        else:  # Only a gate may name a shared (law-independent) clean test.
            assert dependency["kind"] == "gate" and dependency["task"].removesuffix(COVERAGE.suffix) in LAW_INDEPENDENT
    if "continuation_of" in task["execution"]:
        assert task["execution"]["continuation_of"] == variant_id(parent(name)["execution"]["continuation_of"], SERVED)
        task["execution"]["continuation_of"] = clean["execution"]["continuation_of"]
    assert task == clean
    assert validate(served[name], root=ROOT) == parent(name)
    assert adapter_preflight(served[name], candidate("e22"), root=ROOT) == []


def test_law_independent_tests_are_shared_never_duplicated(variants):
    from experiments.forge.tier1_policy import PolicyCohort
    for name in LAW_INDEPENDENT:
        with pytest.raises(ValueError, match="law-independent"):
            materialize(parent(name), ROOT, SERVED)
        assert served_row_task(name) == variant_id(name)
        task = variants[name]
        # Its measurement scores no output samples, or a blocker refuses it first.
        assert (name in BLOCKED_HOSTS or task["adapter"] == "clockfree_audit"
                or task["evaluation"]["policy_observation"]["sampler"] == "served_snapshot")
    assert set(BLOCKED_HOSTS) <= set(LAW_INDEPENDENT)
    with pytest.raises(ValueError, match="same-law producer"):
        SERVED.dependency_task("two_pole", "checkpoint")
    with pytest.raises(ValueError, match="draws no output samples"):
        SERVED.sampling_law("learned_particles_and_critic_gradient")
    with pytest.raises(ValueError, match="law-independent parents stay"):
        PolicyCohort(id="x", parents=("two_pole",), sources=SOURCES, blocked_hosts={}, law_prefix="x_", label="x",
                     parameter_parents=frozenset(), clock_parents=frozenset(), measurement="as_served",
                     shared=frozenset({"two_pole"}), partner=COHORT)
    with pytest.raises(ValueError, match="clean or as_served"):
        PolicyCohort(id="x", parents=(), sources=(), blocked_hosts={}, law_prefix="x_", label="x", measurement="noisy")


class _FakeTrainer:
    completed_steps = 7

    def __init__(self):
        self.calls = []

    def output_sigma(self):
        return .031

    def sample(self, n, *, ema=False, generator=None, output_noise=False, fixed_first_n=False):
        import torch
        self.calls.append(output_noise)
        return torch.full((n, 2), float(output_noise))


def test_output_noise_is_permitted_only_for_the_as_served_cohort(variants, served):
    from experiments.forge.adapters import bind_measurement_law
    from experiments.forge.sampling import SERVED_LEARNED_NOISE, executed_receipt, task_blockers
    with pytest.raises(ValueError, match="registered policy task"):
        bind_measurement_law(_FakeTrainer(), parent("vector_spiral"))
    clean = _FakeTrainer()
    law = bind_measurement_law(clean, variants["vector_spiral"])
    clean.sample(8)
    clean.sample(8, output_noise=False)
    with pytest.raises(ValueError, match="without output noise"):
        clean.sample(8, output_noise=True)
    assert clean.calls == [False, False]
    assert law.receipt()["output_noise"] is False and law.receipt()["draws_without_output_noise"] == 2
    noisy = _FakeTrainer()
    served_law = bind_measurement_law(noisy, served["vector_spiral"])
    noisy.sample(8)
    noisy.sample(8, ema=True, output_noise=False)  # An evaluator's clean request takes the declared law.
    assert noisy.calls == [True, True]
    receipt = served_law.receipt()
    assert {key: receipt[key] for key in ("law", "output_noise", "draws", "draws_without_output_noise",
                                          "evaluator_clean_requests_served")} == {
        "law": SERVED_LEARNED_NOISE, "output_noise": True, "draws": 2, "draws_without_output_noise": 0,
        "evaluator_clean_requests_served": 1}
    assert receipt["output_sigma"] == {"first": .031, "last": .031, "min": .031, "max": .031}
    assert served_law.observed(1) == {"output_noise": True, "measured_draws": [
        {"role": "measured", "model": "ema", "n": 8, "completed_steps": 7, "output_noise": True, "output_sigma": .031}]}
    assert served_law.observed(2) == {}
    assert bind_measurement_law(noisy, served["vector_spiral"]) is served_law
    with pytest.raises(ValueError, match="another declared law"):
        bind_measurement_law(noisy, variants["vector_spiral"])
    # The as-served law name exists only under the registered as-served prefix.
    name = served["vector_spiral"]["evaluation"]["sampling_law"]
    assert executed_receipt(name, eval_output_noise=SERVED_LEARNED_NOISE)["sampling_law"] == name
    with pytest.raises(ValueError):
        executed_receipt(name, eval_output_noise="clean")
    with pytest.raises(ValueError):
        executed_receipt("public_prior_with_learned_output_noise", eval_output_noise=SERVED_LEARNED_NOISE)
    for task in (parent("vector_spiral"), variants["vector_spiral"]):
        impersonating = deepcopy(task)
        impersonating["evaluation"].update(sampling_law=name, eval_output_noise=SERVED_LEARNED_NOISE)
        assert task_blockers(impersonating)  # Neither an ordinary task nor the clean variant may claim it.
    assert task_blockers(served["vector_spiral"]) == [] and task_blockers(variants["vector_spiral"]) == []


def test_policy_evidence_requires_the_observed_output_noise_law(variants, served):
    task = served["vector_spiral"]
    evidence = _policy_evidence(task, 1600)
    assert validate_evidence(task, evidence) is None
    for change, status in (("drop_draws", "INCOMPLETE"), ("clean_draw", "INVALID"),
                           ("drop_summary", "INCOMPLETE"), ("observed_clean", "INVALID")):
        altered = deepcopy(evidence)
        row = altered["policy_observations"][0]
        if change == "drop_draws":
            row.pop("measured_draws")
        elif change == "clean_draw":
            row["measured_draws"][0]["output_noise"] = False
        elif change == "drop_summary":
            altered.pop("served_output_noise")
        else:
            row["output_noise"] = False
        assert validate_evidence(task, altered)["status"] == status, change
    ordinary = variants["vector_spiral"]
    evidence = _policy_evidence(ordinary, 1600)
    assert validate_evidence(ordinary, evidence) is None
    noisy = deepcopy(evidence)
    noisy["policy_observations"][0]["measured_draws"][0]["output_noise"] = True
    assert validate_evidence(ordinary, noisy)["status"] == "INVALID"
    legacy = deepcopy(evidence)
    legacy["policy_observations"][0].pop("measured_draws")  # Older clean receipts recorded no draws.
    assert validate_evidence(ordinary, legacy) is None


def test_policy_gaussian_frozen_control_is_measured_under_the_same_law(served, monkeypatch):
    from experiments.forge import policy_gaussian
    monkeypatch.setattr(policy_gaussian, "_ORIGINAL_BUILD",
                        lambda request, task, device, *, max_steps=1000: (object(), _FakeTrainer(), {}))

    def run(request, task, output, device, *, prerequisites=None, diagnostic=False):
        # The unchanged evaluator builds its active trainer, then its frozen control.
        _, active, _ = policy_gaussian._SCOPE["build"](request, task, device)
        _, frozen, _ = policy_gaussian._SCOPE["build"](request, task, device, max_steps=6000)
        active.sample(4096)
        frozen.sample(4096, generator=None, output_noise=False)
        return active._forge_measurement_law, frozen._forge_measurement_law
    monkeypatch.setattr(policy_gaussian, "_ORIGINAL_RUN", run)
    primary, control = policy_gaussian.run_gaussian({}, served["gaussian1d_stability"], None, "cuda")
    assert control.draws is primary.draws and [d["role"] for d in primary.draws] == ["measured", "frozen_control"]
    assert all(d["output_noise"] for d in primary.draws)
    assert primary.receipt()["control_draws"] == 1 and primary.receipt()["evaluator_clean_requests_served"] == 1
    _, lone, _ = policy_gaussian.build({}, served["gaussian1d_smoke"], "cuda")
    assert lone._forge_measurement_law.role == "measured"


@pytest.mark.parametrize("name,steps", [("vector_spiral", 24), ("img_bars4", 24)])
def test_both_laws_train_identically_and_only_the_as_served_draws_carry_noise(name, steps, tmp_path):
    """Short CPU software control through the real adapters (never a result)."""
    import torch
    from experiments.forge.adapters import _dispatch_task
    torch.set_num_threads(1)
    short = parent(name)
    short["execution"]["steps"] = steps
    receipts = {}
    for cohort in (COVERAGE, SERVED):
        task = materialize(short, ROOT, cohort)
        request = {"candidate": candidate("e22"), "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
        raw = _dispatch_task(request, {"task_id": task["id"]}, tmp_path / cohort.id, "cpu")
        assert grade_result(task, raw)["status"] in {"PASS", "FAIL"}
        receipts[cohort.measurement] = raw["evidence"]
    clean, as_served = receipts["clean"], receipts["as_served"]
    assert clean["served_output_noise"]["draws_without_output_noise"] == clean["served_output_noise"]["draws"]
    assert as_served["served_output_noise"]["draws_without_output_noise"] == 0
    assert as_served["served_output_noise"]["output_sigma"]["min"] > 0
    assert all(row["output_noise"] is True for row in as_served["policy_observations"])
    assert all(row["output_noise"] is False for row in clean["policy_observations"])
    # Evaluation is pure: both laws end in the same training state.
    assert clean["training_state"] == as_served["training_state"]
    assert clean["observations"] != as_served["observations"]


def test_policy_cloud_views_display_only_on_the_atlas_and_e22_pages():
    from experiments.forge.family_reports import _displayed_for
    tier1 = read_json(ROOT / "configs/forge/views/tier1_policy_coverage.json")
    for view in (view_declaration(), served_view_declaration()):
        assert _displayed_for(view, "atlas") and _displayed_for(view, "e22")
        assert not _displayed_for(view, "bcap-pure") and not _displayed_for(view, "k3p")
    assert all(_displayed_for(tier1, family) for family in ("atlas", "bcap-pure", "k3p"))
