"""Budget extensions retain trained prefix state, RNG and the original gates."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

import pytest
import torch

from experiments.forge import adapters
from experiments.forge.contracts import file_hash, stable_hash
from experiments.forge.sampling import executed_receipt, PUBLIC_PRIOR_CLEAN
from experiments.forge.state import state_digest
from experiments.forge.vector_budget_diagnostics import EVALUATOR, validate, validate_original
from experiments.forge.views import grade_result, load_tasks, load_view


ROOT = Path(__file__).resolve().parents[1]
IDS = ["gaussian1d_acquisition_3000_schedule1000_diagnostic_v1",
       "ring16_acquisition_1600_schedule400_diagnostic_v1"]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def miniature_pair(name):
    original = json.loads((ROOT / "configs/forge/tasks" / f"{name}.json").read_text())
    original["execution"].update(steps=24, produces_state=True)
    original["execution"]["host_definition"].update(steps=24, hidden=8, layers=1, batch=4, particles=12)
    diagnostic = deepcopy(original)
    diagnostic["id"] += "_mini_budget_diagnostic"
    diagnostic["execution"].update(steps=29, original_schedule_horizon=24,
        budget_diagnostic={"schema_version": 1, "kind": "preserve_vector_prefix_v1",
                           "original_task": name, "prefix_steps": 24})
    diagnostic["evaluation"].update(kind="transfer_budget_diagnostic", evaluator=EVALUATOR,
                                    observations=29, observation_steps=list(range(1, 30)))
    return original, diagnostic


def request(task):
    return {"candidate": {"recipe_overrides": {"reg_arm": "b_cap", "reg_coeff": 1.,
             "input_noise_std": .05, "output_noise_std": .02, "output_noise_warmup": .5,
             "lr": .001, "d_lr_mult": 2., "prior_lr_mult": .5}},
            "candidate_revision": "unit-test", "protocol": {"seed": 0}, "tasks": {task["id"]: task}}


def test_frozen_diagnostics_preserve_original_gates_and_do_not_enter_standard_view():
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "bcap_budget_diagnostics_v1")
    assert view["evidence_scope"] == "research_diagnostic"
    assert [row["task"] for row in view["assignments"]] == IDS
    assert all(row["importance"] == "diagnostic" for row in view["assignments"])
    standard = load_view(ROOT, "discriminator_stability")
    assert not set(IDS) & {row["task"] for row in standard["assignments"]}
    for identifier, prefix, limit, checks, timeout in zip(
            IDS, [1000, 400], [3000, 1600], [list(range(2600, 3001, 100)), list(range(1200, 1601, 100))], [360, 1200]):
        task = tasks[identifier]
        validate_original(task)
        contract = validate(task)
        assert contract["prefix_steps"] == contract["original_schedule_horizon"] == prefix
        assert contract["execution_steps"] == limit
        assert contract["prefix_observation_steps"] == [math.ceil(i * prefix / 24) for i in range(1, 25)]
        assert contract["terminal_observation_steps"] == checks
        assert task["resources"]["timeout_seconds"] == timeout
        assert adapters.adapter_preflight(task, request(task)["candidate"], root=ROOT) == []


@pytest.mark.parametrize("name", ["gaussian1d_acquisition", "ring16_acquisition"])
def test_extended_public_training_exactly_matches_original_prefix_including_eval_rng(tmp_path, name):
    original, diagnostic = miniature_pair(name)
    global_rng = torch.get_rng_state().clone()
    rows = []
    for task in (original, diagnostic):
        directory = tmp_path / task["id"]
        rows.append(adapters.run_task(request(task), {"task_id": task["id"]}, directory, "cpu"))
        assert torch.equal(torch.get_rng_state(), global_rng)
    base, extended = rows
    state = torch.load(tmp_path / original["id"] / "state.pt", weights_only=True)
    records = torch.load(tmp_path / original["id"] / "observed-samples.pt", weights_only=True)
    long_records = torch.load(tmp_path / diagnostic["id"] / "observed-samples.pt", weights_only=True)
    proof = extended["evidence"]["budget_diagnostic"]["prefix"]
    assert state_digest(state) == proof["context_except_execution_cap_sha256"]
    assert state_digest(state["streams"]) == proof["named_rng_sha256"]
    assert base["evidence"]["observations"] == extended["evidence"]["observations"][:24]
    assert state_digest(records) == state_digest(long_records[:24]) == proof["scored_samples_sha256"]
    assert stable_hash(base["evidence"]["observations"]) == proof["observations_sha256"]
    assert extended["recipe"]["total_steps"] == 24
    assert extended["cost"]["completed_steps"] == 29
    final = torch.load(tmp_path / diagnostic["id"] / "state.pt", weights_only=True)
    assert final["trainer"]["max_steps"] == final["trainer"]["completed_steps"] == 29
    assert extended["evidence"]["guards"]["unintended_rng_deviations"] == 0
    assert grade_result(diagnostic, extended)["gate_status"] in {"PASS", "FAIL"}
    if name == "gaussian1d_acquisition":
        from experiments.forge.tier1_media import render
        directory = tmp_path / diagnostic["id"]
        before = {name: file_hash(directory / name) for name in ("state.pt", "observed-samples.pt")}
        gif = tmp_path / "actual-training.gif"
        row = {**extended, "gate_status": grade_result(diagnostic, extended)["gate_status"]}
        media = render(diagnostic, row, directory, gif)
        assert gif.is_file() and media["observation_count"] == 29
        assert media["optimizer_updates_added"] == media["sampling_draws_added"] == 0
        assert media["selected_observation_indices"][-1] == 28
        assert before == {name: file_hash(directory / name) for name in before}


@pytest.mark.parametrize("mutation", ["schedule", "host_steps", "prefix_cadence", "terminal_count", "terminal_end"])
def test_invalid_budget_or_observation_contract_is_rejected(mutation):
    task = load_tasks(ROOT)[IDS[0]]
    if mutation == "schedule":
        task["execution"]["original_schedule_horizon"] = 3000
    elif mutation == "host_steps":
        task["execution"]["host_definition"]["steps"] = 3000
    elif mutation == "prefix_cadence":
        task["evaluation"]["observation_steps"][0] += 1
    elif mutation == "terminal_count":
        task["evaluation"]["observation_steps"].pop()
    else:
        task["evaluation"]["observation_steps"][-1] -= 1
    with pytest.raises(ValueError):
        validate(task)


@pytest.mark.parametrize("section,field,value", [
    ("evaluation", "thresholds", [["mean_error_sigma", "<=", 2.]]),
    ("execution", "initializer", "kaiming"),
    ("execution", "prior", {"kind": "mog", "sigma": .05, "standardize": False, "learnable": True}),
])
def test_diagnostic_cannot_silently_change_original_gate_or_training_cohort(section, field, value):
    task = load_tasks(ROOT)[IDS[0]]
    task[section][field] = value
    with pytest.raises(ValueError, match="changed original"):
        validate_original(task)


def passing_evidence(task):
    contract = validate(task)
    values = dict(sample_count=4096, finite_fraction=1., mean_error_sigma=0., std_ratio=1., cdf_ks=0.)
    points = [{"step": step, **values} for step in contract["observation_steps"]]
    prefix = {"completed_steps": contract["prefix_steps"], "observations_sha256": stable_hash(points[:24]),
              **{key: "a" * 64 for key in ("context_except_execution_cap_sha256", "named_rng_sha256", "scored_samples_sha256")}}
    return {"observations": points, "live": deepcopy(points[-1]),
            "budget_diagnostic": {**contract, "prefix": prefix,
                "recipe_schedule_horizon": contract["original_schedule_horizon"],
                "trainer_execution_limit": contract["execution_steps"], "completed_steps": contract["execution_steps"]},
            **executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean")}


def test_numerical_gate_requires_every_late_check_and_retains_original_failure():
    task = load_tasks(ROOT)[IDS[0]]
    evidence = passing_evidence(task)
    for point in evidence["observations"][:24]:
        point["cdf_ks"] = .2
    evidence["budget_diagnostic"]["prefix"]["observations_sha256"] = stable_hash(evidence["observations"][:24])
    result = grade_result(task, {"evidence": evidence})
    assert result["gate_status"] == "PASS"
    assert result["original_prefix_result"]["passing_suffix"] == 0
    evidence["observations"][-3]["cdf_ks"] = .06
    assert grade_result(task, {"evidence": evidence})["gate_status"] == "FAIL"
    evidence["observations"].pop(-3)
    assert grade_result(task, {"evidence": evidence})["gate_status"] == "INCOMPLETE"


def test_changed_prefix_identity_or_final_live_metrics_cannot_pass():
    task = load_tasks(ROOT)[IDS[0]]
    evidence = passing_evidence(task)
    evidence["budget_diagnostic"]["prefix"]["observations_sha256"] = "b" * 64
    assert grade_result(task, {"evidence": evidence})["gate_status"] == "INVALID"
    evidence = passing_evidence(task)
    evidence["live"]["std_ratio"] = .99
    assert grade_result(task, {"evidence": evidence})["gate_status"] == "INVALID"


@pytest.mark.parametrize("field,value", [("recipe_schedule_horizon", 3000), ("trainer_execution_limit", 1000),
                                         ("completed_steps", 1000)])
def test_stretched_schedule_or_incomplete_actual_training_cannot_pass(field, value):
    task = load_tasks(ROOT)[IDS[0]]
    evidence = passing_evidence(task)
    evidence["budget_diagnostic"][field] = value
    assert grade_result(task, {"evidence": evidence})["gate_status"] == "INVALID"


def frozen_budget_request(tmp_path, identifier):
    """Use the production queue and host boundary, with no training worker."""
    from test_forge_hostprofiles import prospective, bind_candidate, rebind
    from experiments.forge.sources import inspect_source, snapshot_source

    task = load_tasks(ROOT)[identifier]
    req = prospective(tmp_path, [task])
    relative = f"configs/forge/tasks/{task['execution']['budget_diagnostic']['original_task']}.json"
    checkout = tmp_path / "worktree"
    target = checkout / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative, target)
    files = set(req["source"]["files"]) | {relative}
    source = inspect_source(checkout, files)
    source["snapshot_path"] = str(snapshot_source(checkout, tmp_path / "queue", source))
    req["source"] = source
    bind_candidate(req)
    rebind(req)
    return req


@pytest.mark.parametrize("identifier", IDS)
def test_actual_queue_admission_accepts_valid_frozen_budget_extension_without_training(tmp_path, monkeypatch, identifier):
    from experiments.forge.queue import Queue
    from test_forge_queue import campaign

    req = frozen_budget_request(tmp_path, identifier)
    def forbidden(*args, **kwargs):
        raise AssertionError("admission must not construct models or train")
    monkeypatch.setattr("experiments.forge.api.FormulationContext.construct", forbidden)
    queue = Queue(tmp_path / "queue")
    assert queue.submit(req, campaign("budget-extension", budget=1560))["status"] == "queued"
    state = queue.inspect()
    assert all(not job["attempts"] and job["status"] == "pending" for job in state["jobs"].values())
    assert state["campaigns"]["budget-extension"]["spent_seconds"] == 0
    assert state["campaigns"]["budget-extension"]["reserved_seconds"] == 0


@pytest.mark.parametrize("mutation", ["unmarked", "horizon", "cadence", "host_steps", "host_width", "thresholds", "prior", "source_pin"])
def test_budget_exception_cannot_bypass_actual_queue_host_locks_even_with_rehashed_jobs(tmp_path, mutation):
    from experiments.forge.queue import Queue
    from test_forge_hostprofiles import rebind
    from test_forge_queue import campaign

    req = frozen_budget_request(tmp_path, IDS[0])
    task = req["tasks"][IDS[0]]
    if mutation == "unmarked":
        task["execution"].pop("budget_diagnostic")
        task["evaluation"]["kind"] = "transfer_sustained"
    elif mutation == "horizon":
        task["execution"]["original_schedule_horizon"] = 3000
    elif mutation == "cadence":
        task["evaluation"]["observation_steps"][0] += 1
    elif mutation == "host_steps":
        task["execution"]["host_definition"]["steps"] = 3000
    elif mutation == "host_width":
        task["execution"]["host_definition"]["hidden"] = 64
    elif mutation == "thresholds":
        task["evaluation"]["thresholds"][-1][-1] = .5
    elif mutation == "prior":
        task["execution"]["prior"]["sigma"] = .05
    else:
        relative = f"configs/forge/tasks/{task['execution']['budget_diagnostic']['original_task']}.json"
        task["evaluation"]["sources"][relative] = "0" * 64
    rebind(req)
    queue = Queue(tmp_path / "queue")
    with pytest.raises(ValueError, match="host profile blocked"):
        queue.submit(req, campaign("budget-extension", budget=1560))
    assert not (queue.root / "queue/state.json").exists()
