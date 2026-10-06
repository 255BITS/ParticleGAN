"""Declare, freeze and execute the finite BCAP dualnorm pacing study via Forge.

No trainer, task or evaluator is copied here. Conditional grids are pure
derivations of the committed contract; only one B and one C branch can run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.configuration_search import (
    enqueue_search, materialize_search, plan_search, report_search, select_configuration)
from experiments.forge.contracts import atomic_json, file_hash, file_lock, read_json, stable_hash, utc_now
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue, drain, lease_held

REPORT = ROOT / "reports/forge/dualnorm-pacing-v2"
CONTRACT = REPORT / "contract.json"
CAMPAIGN = "bcap-dualnorm-pacing-v2"
REQUIRED = ("gaussian1d_acquisition", "two_pole", "unused_token_hold", "ae_gan_hold",
            "ring16_acquisition", "five_word_joint_acquisition")
CLOCK = "clockfree_audit_measurement_v1"
TERMINAL = {"PASS", "FAIL", "INCOMPLETE", "INVALID", "BLOCKED"}
FINISHED_SUBMISSIONS = {"completed", "blocked", "terminal", "concluded"}


def emit(event, **fields):
    print(json.dumps({"time": utc_now(), "event": event, **fields}, sort_keys=True), flush=True)


def load_inputs(root=ROOT):
    root = Path(root)
    contract = read_json(root / "reports/forge/dualnorm-pacing-v2/contract.json")
    a = read_json(root / contract["stage_a_spec"])
    campaign = read_json(root / contract["campaign"])
    if (contract["schema_version"] != 1 or contract["id"] != CAMPAIGN
            or campaign != a["campaign"] or campaign["id"] != CAMPAIGN
            or contract["maximum_admitted_recipes"] != 25
            or contract["maximum_paid_reserved_seconds"] != 63000 or campaign["budget_seconds"] != 63000
            or campaign["candidate_budget_seconds"] != 2520
            or contract["maximum_elapsed_seconds"] != 43200
            or contract["automatic_retries"] != 0 or contract["automatic_expansions"] != 0
            or contract["required_tasks"] != list(REQUIRED) or contract["diagnostic_task"] != CLOCK
            or contract["rates_a"] != [.01] or contract["discriminator_ratios"] != [.5, .75, 1., 1.5, 2.]
            or contract["prior_steps"] != [.003, .01, .03, .1] or contract["rates_b"] != [.012, .016, .022]
            or contract["momenta_c"] != [.5, .9] or contract["cleanup_grace_seconds"] != 30
            or contract["momentum_base_candidate"] != "bcap-dualnorm-momentum-v1"
            or contract["qualification_through_tier"] != 1 or contract["default_adoption"] is not False
            or any(contract["activation"]["stage_" + stage]["minimum_required_passes"] != 3
                   or contract["activation"]["stage_" + stage]["at_least_current_control"] is not True
                   for stage in ("b", "c"))
            or any(contract["control"].get(key) != value for key, value in {
                   "lr": .01, "d_lr_mult": 1.5, "prior_step": .03, "optimizer_momentum": 0.}.items())):
        raise ValueError("study scope or budget differs from its finite reviewed contract")
    expected = {"lr": [.01], "d_lr_mult": contract["discriminator_ratios"],
                "prior_lr_mult": [step / .01 for step in contract["prior_steps"]]}
    if a["grid"] != expected:
        raise ValueError("Stage A differs from the master pacing grid")
    if any(a[key] != value for key, value in {
            "base_candidate": "bcap-dualnorm-zero-v1", "tuning_through_tier": 1,
            "protocol": "screening", "view": "discriminator_stability", "execution_backend": "cuda",
            "trainer_family": "bcap-dualnorm", "cuda_model": "NVIDIA RTX A6000"}.items()):
        raise ValueError("study must retain its declared pure BCAP technique and Tier 1 protocol")
    return contract, a


def branch_spec(stage, pace, contract, stage_a_spec):
    """An exact bounded Forge declaration, derived before any result exists."""
    spec = deepcopy(stage_a_spec)
    key = {"d_lr_mult": pace["d_lr_mult"], "prior_step": pace["prior_step"]}
    if stage == "C":
        key["lr"] = pace["lr"]
    elif stage != "B":
        raise ValueError("only B/C have conditional branches")
    spec["id"] = CAMPAIGN + "-" + stage.lower() + "-" + stable_hash(key)[:16]
    rates = contract["rates_b"] if stage == "B" else [pace["lr"]]
    spec["grid"] = {"player_rates": [{"lr": lr, "d_lr_mult": pace["d_lr_mult"],
                         "prior_lr_mult": pace["prior_step"] / lr} for lr in rates]}
    if stage == "C":
        spec["base_candidate"] = contract["momentum_base_candidate"]
        spec["grid"]["optimizer_momentum"] = contract["momenta_c"]
    spec["hypothesis"] = ("Resolve the .01-.03 generator-rate tradeoff at one selected whole recipe's "
        "fixed discriminator ratio and absolute sampled-prior pace, retaining all sustained gates."
        if stage == "B" else "At the selected whole pace, positive shared D/G/E momentum .5/.9 "
        "may improve complete sustained gates; the sampled prior retains zero momentum.")
    spec["rationale"] = ("Conditional branch fixed by reports/forge/dualnorm-pacing-v2/contract.json "
        "before spending. Activate only its registered PASS-count/hash whole-row branch after all "
        "current-tier attempts finish, selected evidence is valid PASS/FAIL, and the winner has "
        ">=3 required passes and >=the matched current-source control. No task-specific selection, "
        "new task laws, source changes, retries or expansion. Held pace: " + json.dumps(key, sort_keys=True))
    return spec


def declared_specs(contract, stage_a_spec):
    entries = [{"stage": "A", "pace": None, "spec": deepcopy(stage_a_spec)}]
    for ratio, prior in product(contract["discriminator_ratios"], contract["prior_steps"]):
        pace = {"lr": .01, "d_lr_mult": ratio, "prior_step": prior}
        entries.append({"stage": "B", "pace": pace,
                        "spec": branch_spec("B", pace, contract, stage_a_spec)})
    for lr, ratio, prior in product(contract["rates_a"] + contract["rates_b"],
                                   contract["discriminator_ratios"], contract["prior_steps"]):
        pace = {"lr": lr, "d_lr_mult": ratio, "prior_step": prior}
        entries.append({"stage": "C", "pace": pace,
                        "spec": branch_spec("C", pace, contract, stage_a_spec)})
    if len(entries) != 101 or len({entry["spec"]["id"] for entry in entries}) != 101:
        raise ValueError("conditional declaration space must contain exactly 101 bounded searches")
    return entries


def task_map(trial):
    return {task["task"]: task for task in trial["tasks"] if task["qualification_tier"] == 1}


def execution_complete(trial):
    tasks = task_map(trial)
    return (trial.get("submission_status", "").lower() in FINISHED_SUBMISSIONS
            and sum(task["qualification_tier"] == 1 for task in trial["tasks"]) == 7
            and set(tasks) == {*REQUIRED, CLOCK}
            and all(tasks[name]["importance"] == "required" for name in REQUIRED)
            and tasks[CLOCK]["importance"] == "diagnostic"
            and all(task.get("attempt_id") and task["gate_status"] in TERMINAL for task in tasks.values()))


def whole_selection(trials):
    if not trials:
        raise ValueError("whole selection requires at least one declared recipe")
    for key in ("source_digest", "runtime_cohort", "protocol_hash", "policy_fingerprint"):
        if any(not trial.get(key) for trial in trials) or len({stable_hash(t[key]) for t in trials}) != 1:
            raise ValueError("whole selection requires one frozen source/runtime/protocol/view cohort: " + key)
    if len({t["configuration_id"] for t in trials}) != len(trials):
        raise ValueError("whole selection must count each configuration exactly once")
    selection = select_configuration(trials, 1)
    if not all(execution_complete(trial) for trial in trials):
        selection.update(selection_complete=False, all_trials_terminal=False,
            selected_candidate_id=None, selected_configuration_id=None, qualified=False,
            selection_kind="pending", required_pass_count=None, required_total=None,
            required_passes_by_tier=None, source_digest=None, required_task_bindings=[])
    elif selection["selected_candidate_id"]:
        selected = next(t for t in trials if t["candidate_id"] == selection["selected_candidate_id"])
        valid = all(task["gate_status"] in {"PASS", "FAIL"} for task in task_map(selected).values())
        selection["selected_evidence_valid"] = valid
        if not valid:
            selection.update(qualified=False, selection_kind="best_observed_with_invalid_evidence")
    return selection


def pace_from_trial(trial, contract):
    recipe = trial["resolved_recipe"]
    actual = recipe["lr"] * recipe["prior_lr_mult"]
    priors = [p for p in contract["prior_steps"] if math.isclose(actual, p, rel_tol=0, abs_tol=1e-14)]
    if len(priors) != 1 or recipe["d_lr_mult"] not in contract["discriminator_ratios"]:
        raise ValueError("selected pace lies outside the predeclared absolute-prior grid")
    return {"lr": recipe["lr"], "d_lr_mult": recipe["d_lr_mult"], "prior_step": priors[0]}


def activation(stage, trials, control_trial, contract):
    selection = whole_selection(trials)
    result = {"eligible": False, "selection": selection, "pace": None}
    matching_control = [t for t in trials if t["candidate_id"] == control_trial["candidate_id"]]
    if len(matching_control) != 1 or stable_hash(matching_control[0]) != stable_hash(control_trial):
        return result | {"reason": "matched control must be the actual source-compatible whole row in this pool"}
    if not selection["selection_complete"]:
        return result | {"reason": "current stage has missing or unfinished required/diagnostic attempts"}
    selected = next(t for t in trials if t["candidate_id"] == selection["selected_candidate_id"])
    if not execution_complete(control_trial) or any(
            task["gate_status"] not in {"PASS", "FAIL"}
            for trial in (selected, control_trial) for task in task_map(trial).values()):
        return result | {"reason": "selected whole recipe or matched control has invalid/incomplete evidence"}
    rule = contract["activation"]["stage_" + stage.lower()]
    control_passes = sum(task_map(control_trial)[name]["gate_status"] == "PASS" for name in REQUIRED)
    count = selection["required_pass_count"]
    if count < rule["minimum_required_passes"] or (rule["at_least_current_control"] and count < control_passes):
        return result | {"reason": "whole winner misses the frozen >=3 and >=current-control activation predicate",
                         "control_required_passes": control_passes}
    return result | {"eligible": True, "reason": "frozen complete whole-row predicate satisfied",
                     "control_required_passes": control_passes, "pace": pace_from_trial(selected, contract)}


def branch_for(stage, pace, entries):
    keys = ("d_lr_mult", "prior_step") if stage == "B" else ("lr", "d_lr_mult", "prior_step")
    matches = [entry for entry in entries if entry["stage"] == stage
               and all(entry["pace"][key] == pace[key] for key in keys)]
    if len(matches) != 1:
        raise ValueError("selected conditional branch is not in the frozen finite space")
    return matches[0]


def deadline_admits(seconds, remaining, grace=30):
    return all(math.isfinite(value) for value in (seconds, remaining, grace)) and seconds > 0 and grace >= 0 and seconds + grace <= remaining


def validate_devices(devices):
    if len(devices) != 2 or any(not device.isdigit() for device in devices):
        raise ValueError("study requires exactly two distinct physical numeric GPU indices")
    devices = [str(int(device)) for device in devices]
    if len(set(devices)) != 2:
        raise ValueError("study requires two distinct GPUs, one worker each plus the automatic CPU worker")
    return devices


class DeadlineQueue(Queue):
    """Keep Forge's full reservations; do not requeue jobs that cannot fit."""
    def __init__(self, root, *, deadline, cleanup_grace_seconds=30, clock=time.monotonic, **kwargs):
        super().__init__(root, **kwargs)
        self.deadline, self.grace, self.clock = deadline, cleanup_grace_seconds, clock
        self.denied = {}

    def _deny(self, key, seconds):
        if key not in self.denied:
            self.denied[key] = {"budget_seconds": seconds, "remaining_seconds": max(0., self.deadline - self.clock())}
            emit("deadline_admission_denied", compatibility_key=key, **self.denied[key])

    def _available_budget(self, state, request, seconds, *, job_key=None, new_round_reservation=True):
        allowed, reason = super()._available_budget(state, request, seconds, job_key=job_key,
                                                   new_round_reservation=new_round_reservation)
        if not allowed:
            return allowed, reason
        if not deadline_admits(seconds, self.deadline - self.clock(), self.grace):
            self._deny(job_key or request.get("request_id", "unbound"), seconds)
            return False, "elapsed ceiling cannot reserve the unchanged full task timeout"
        return True, reason

    def _eligible(self, state, submission):
        eligible, reason, running = super()._eligible(state, submission)
        remaining = self.deadline - self.clock()
        kept = []
        for key in eligible:
            seconds = state["jobs"][key]["definition"]["budget_seconds"]
            if key not in self.denied and deadline_admits(seconds, remaining, self.grace):
                kept.append(key)
            else:
                self._deny(key, seconds)
        if eligible and not kept and not running:
            return [], "elapsed ceiling prevents full-timeout admission; unexecuted cells remain unknown", False
        return kept, reason, running


def verify_committed_bytes(commit, hashes, root=ROOT):
    """Verify actual bytes against Git before any queue admission."""
    references = "".join(commit + ":" + name + "\n" for name in hashes)
    payload = subprocess.check_output(["git", "cat-file", "--batch"], input=references.encode(), cwd=root)
    cursor = 0
    for name, expected in hashes.items():
        end = payload.index(b"\n", cursor)
        header = payload[cursor:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("input is not committed: " + name)
        size = int(header[2]); cursor = end + 1
        data = payload[cursor:cursor + size]; cursor += size + 1
        if hashlib.sha256(data).hexdigest() != expected or file_hash(Path(root) / name) != expected:
            raise ValueError("input differs from committed bytes: " + name)


def input_hashes(contract):
    names = ["reports/forge/dualnorm-pacing-v2/contract.json",
             "reports/forge/dualnorm-pacing-v2/run.py", contract["stage_a_spec"], contract["campaign"]]
    return {name: file_hash(ROOT / name) for name in names}


def prepare(entries):
    total = set()
    for index, entry in enumerate(entries, 1):
        total.update(materialize_search(ROOT, entry["spec"]))
        if index % 10 == 0 or index == len(entries):
            emit("prepared", branches=index, reachable_configurations=len(total))
    if len(total) != 240:
        raise ValueError("reachable space must contain exactly 240 distinct recipes")


def plan_all(queue_root, contract, entries):
    branches, source, cohort = [], None, None
    for index, entry in enumerate(entries, 1):
        summary = plan_search(ROOT, queue_root, entry["spec"])
        for trial in summary["trials"]:
            tasks = task_map(trial)
            if (set(tasks) != {*REQUIRED, CLOCK} or trial["declared_worst_case_seconds"] != 2520
                    or trial["submission_blockers"] or trial["submission_status"] != "READY"):
                raise ValueError("reachable recipe is blocked or changes the full Tier 1 scope: " + trial["candidate_id"])
        current = {key: summary[key] for key in ("source_digest", "runtime_cohort", "protocol_hash", "policy_fingerprint")}
        if cohort is None:
            cohort = current
            trial = summary["trials"][0]
            request = resolve_idea(ROOT, trial["candidate_id"], declaration=trial["declaration"],
                view_id=summary["view"], through_tier=1, execution_backend="cuda", cuda_model=entry["spec"]["cuda_model"])
            source = request["source"]
            if request["protocol"]["seed"] != 0 or request["execution_policy"]["mode"] != "complete_current_tier":
                raise ValueError("study must use seed 0 and complete independent current-tier peers")
        elif current != cohort:
            raise ValueError("reachable branches do not share one source/runtime/protocol/view")
        branches.append({"stage": entry["stage"], "pace": entry["pace"], "study_id": summary["study_id"],
            "spec_sha256": summary["spec_hash"], "declared_worst_case_seconds": summary["declared_worst_case_seconds"],
            "trials": [{key: trial[key] for key in ("candidate_id", "configuration_id", "candidate_revision",
                "scientific_signature", "settings", "declared_worst_case_seconds", "unreused_worst_case_seconds")}
                | {"task_bindings": [{key: task[key] for key in ("task", "importance", "compatibility_key")}
                                     for task in trial["tasks"] if task["qualification_tier"] == 1]}
                for trial in summary["trials"]]})
        emit("planned_branch", branch=index, total_branches=len(entries), stage=entry["stage"],
             study=summary["study_id"], configurations=len(summary["trials"]))
    control = [t for b in branches if b["stage"] == "A" for t in b["trials"]
               if t["settings"] == {"lr": .01, "d_lr_mult": 1.5, "prior_lr_mult": 3.}]
    if len(control) != 1:
        raise ValueError("Stage A requires exactly one matched current-source starter control")
    result = {"schema_version": 1, "campaign": CAMPAIGN, "qualification_input": False,
        "inputs_sha256": input_hashes(contract), "cohort": cohort, "source": source,
        "maximum_admitted_recipes": 25, "maximum_paid_reserved_seconds": 63000,
        "maximum_elapsed_seconds": 43200, "reachable_configurations": 240,
        "potential_branches": 101, "control_candidate_id": control[0]["candidate_id"], "branches": branches}
    result["input_digest"] = stable_hash(result)
    atomic_json(REPORT / "plan.json", result)
    return result


def freeze(queue_root, contract, entries):
    planned = plan_all(queue_root, contract, entries)
    source = planned["source"]
    verify_committed_bytes(source["origin_commit"], source["files"] | planned["inputs_sha256"])
    path = queue_root / "frozen-study.json"
    if path.exists() and read_json(path) != planned:
        raise ValueError("study freeze is immutable; restore inputs or declare a new campaign")
    atomic_json(path, planned)
    atomic_json(REPORT / "freeze.json", {key: planned[key] for key in (
        "campaign", "inputs_sha256", "cohort", "maximum_admitted_recipes", "maximum_paid_reserved_seconds",
        "maximum_elapsed_seconds", "reachable_configurations", "potential_branches", "control_candidate_id", "input_digest")}
        | {"source_commit": source["origin_commit"], "source_digest": source["digest"]})
    emit("frozen", source_commit=source["origin_commit"], source_digest=source["digest"],
         maximum_recipes=25, maximum_paid_reserved_seconds=63000, maximum_elapsed_seconds=43200)
    return planned


def verify_frozen(queue_root, contract):
    frozen = read_json(queue_root / "frozen-study.json")
    if frozen["input_digest"] != stable_hash({k: v for k, v in frozen.items() if k != "input_digest"}):
        raise ValueError("frozen study digest mismatch")
    if input_hashes(contract) != frozen["inputs_sha256"]:
        raise ValueError("driver or master declarations changed after freeze")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current != frozen["source"]["origin_commit"]:
        raise ValueError("retain the frozen source commit until conditional execution completes")
    verify_committed_bytes(current, frozen["source"]["files"] | frozen["inputs_sha256"])
    return frozen


def compact_trial(trial):
    fields = ("candidate_id", "configuration_id", "candidate_revision", "recipe_overrides", "resolved_recipe",
              "source_digest", "runtime_cohort", "protocol_hash", "policy_fingerprint", "submission_status",
              "submission_reason", "submission_blockers", "request_id", "attempt_ids", "attempt_history", "cost")
    return {key: trial.get(key) for key in fields} | {"tasks": [deepcopy(task) for task in trial["tasks"]
                                                               if task["qualification_tier"] == 1]}


def report(queue_root, state, queue=None):
    queue = queue or Queue(queue_root, report_root=ROOT / "reports/forge")
    studies = [report_search(ROOT, queue_root, stage["spec"], queue=queue)
               for stage in state["stages"] if stage.get("admitted") and
               (ROOT / "reports/forge/configuration-search" / (stage["spec"]["id"] + ".json")).is_file()]
    trials = [trial for study in studies for trial in study["trials"]]
    if len(trials) > 25 or len({t["configuration_id"] for t in trials}) != len(trials):
        raise ValueError("campaign exceeded its finite whole-recipe admission bound")
    selection = whole_selection(trials) if trials else None
    accounting = queue.inspect().get("campaigns", {}).get(CAMPAIGN, {})
    supervision = supervision_state(queue)
    admitted = sum(bool(t.get("request_id")) for t in trials)
    control = next((t for t in trials if t["candidate_id"] == state["control_candidate_id"]), None)
    result = {"schema_version": 1, "campaign": CAMPAIGN, "qualification_input": False,
        "status": state["status"], "stop_reason": state.get("stop_reason"),
        "study_binding_sha256": state["study_binding_sha256"], "source": state["source"],
        "started_at": state["started_at"], "deadline": state["deadline_utc"],
        "elapsed_seconds": max(0., time.monotonic() - state["started_monotonic"]),
        "maximum_elapsed_seconds": 43200, "maximum_paid_reserved_seconds": 63000,
        "admitted_recipes": admitted, "declared_recipes_in_registered_stages": len(trials),
        "automatic_retries": 0, "default_adoption": False,
        "running_workers": len(supervision["running_jobs"]),
        "all_attempts_supervised_finished": not supervision["running_jobs"] and not supervision["active_leases"],
        "attempt_status_counts": dict(Counter(read_json(Path(attempt["path"]) / "terminal.json")["attempt_status"]
            for job in supervision["jobs"] for attempt in job.get("attempts", [])
            if (Path(attempt["path"]) / "terminal.json").is_file())),
        "exit_code": state.get("exit_code", 0 if state["status"] == "completed" else 2),
        "control_candidate_id": state["control_candidate_id"],
        "control_required_passes": sum(task_map(control)[n]["gate_status"] == "PASS" for n in REQUIRED) if control else None,
        "stages": state["stages"], "selection": selection, "accounting": accounting,
        "required_counts": dict(Counter(task_map(t)[name]["gate_status"] for t in trials for name in REQUIRED)),
        "diagnostic_counts": dict(Counter(task_map(t)[CLOCK]["gate_status"] for t in trials)),
        "trials": [compact_trial(t) for t in trials]}
    atomic_json(REPORT / "results.json", result)
    terminal = {key: result[key] for key in ("campaign", "status", "stop_reason", "source", "started_at", "deadline",
        "elapsed_seconds", "admitted_recipes", "required_counts", "diagnostic_counts", "selection", "accounting",
        "running_workers", "all_attempts_supervised_finished", "attempt_status_counts", "exit_code",
        "control_candidate_id", "control_required_passes")}
    terminal["all_supervised_attempts_finished"] = result["all_attempts_supervised_finished"]
    terminal["final_status"] = result["status"]
    terminal["finished_at"] = state.get("finished_at")
    terminal["stages"] = [{k: stage[k] for k in ("stage", "admitted", "decision", "execution_complete", "selection") if k in stage}
                          | {"study_id": stage.get("spec", {}).get("id")} for stage in state["stages"]]
    terminal["terminal"] = state["status"] != "running" and result["all_attempts_supervised_finished"]
    terminal["results_path"] = "reports/forge/dualnorm-pacing-v2/results.json"
    atomic_json(queue_root / "terminal-summary.json", terminal)
    return studies, result


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def validate_resume_state(state, frozen, entries, clock_record):
    """Reject a changed stage roster or deadline before reattaching a worker."""
    if (clock_record.get("input_digest") != stable_hash({k: v for k, v in clock_record.items() if k != "input_digest"})
            or state.get("execution_clock_sha256") != clock_record["input_digest"]
            or state.get("study_binding_sha256") != frozen["input_digest"]
            or state.get("control_candidate_id") != frozen["control_candidate_id"]
            or state.get("source") != {"commit": frozen["source"]["origin_commit"], "digest": frozen["source"]["digest"]}):
        raise ValueError("resume changed the frozen study/source/control/clock binding")
    for key in ("study_binding_sha256", "source", "control_candidate_id", "boot_id", "started_at",
                "started_monotonic", "deadline_monotonic", "deadline_utc"):
        if state.get(key) != clock_record.get(key):
            raise ValueError("resume changed the original execution clock or source: " + key)
    if state["deadline_monotonic"] != state["started_monotonic"] + 43200:
        raise ValueError("resume changed the 12-hour monotonic ceiling")
    stages = state.get("stages")
    names = [s.get("stage") for s in stages] if isinstance(stages, list) else None
    if names is None or names != list(("A", "B", "C")[:len(names)]):
        raise ValueError("resume stage roster must be one ordered A/B/C sequence")
    if len(stages) > 3:
        raise ValueError("resume cannot add another stage")
    for recorded in stages:
        spec = recorded.get("spec")
        if recorded.get("admitted") or spec:
            matches = [e for e in entries if e["stage"] == recorded["stage"] and e["spec"] == spec]
            if len(matches) != 1:
                raise ValueError("resume contains a study outside the exact frozen reachable declarations")
            if recorded["stage"] != "A" and branch_for(recorded["stage"], recorded["decision"]["pace"], entries) != matches[0]:
                raise ValueError("resume changed the recorded whole-winner branch")


def supervision_state(queue):
    state = queue.inspect()
    submissions = {name for name, entry in state.get("submissions", {}).items()
                   if entry["request"]["campaign_id"] == CAMPAIGN}
    jobs = [job for job in state.get("jobs", {}).values()
            if (job.get("cost_owner") or {}).get("campaign") == CAMPAIGN
            or set(job.get("subscribers", [])) & submissions]
    running = [{"task": job["definition"]["task_id"], "device": job["worker"]["device"],
                "attempt_id": job["worker"]["attempt"]} for job in jobs if job["status"] == "running"]
    leases = [attempt["attempt_id"] for job in jobs for attempt in job.get("attempts", [])
              if lease_held(Path(attempt["path"]) / "execution.lock")]
    return {"running_jobs": running, "active_leases": leases, "jobs": jobs, "state": state}


def _progress(queue, stopped, interval=60):
    while not stopped.wait(interval):
        supervision = supervision_state(queue)
        rows = [row for job in supervision["jobs"] if job.get("result") for row in job["result"]["task_results"]]
        accounting = supervision["state"].get("campaigns", {}).get(CAMPAIGN, {})
        emit("progress", completed_attempts=sum(len(job.get("attempts", [])) for job in supervision["jobs"])
             - len(supervision["running_jobs"]), running=supervision["running_jobs"],
             pending_jobs=sum(job["status"] == "pending" for job in supervision["jobs"]),
             known_gate_counts=dict(Counter(row["gate_status"] for row in rows)),
             spent_seconds=accounting.get("spent_seconds", 0), reserved_seconds=accounting.get("reserved_seconds", 0))


def cancel_campaign(queue, reason):
    emit("cancel_campaign", reason=reason)
    for request_id, entry in queue.inspect().get("submissions", {}).items():
        if entry["request"]["campaign_id"] == CAMPAIGN and entry["status"] in {"queued", "running", "paused", "blocked"}:
            queue.cancel(request_id)
    queue.flush_events()


def finish_supervision(queue):
    """Collect existing supervisors without claiming or retrying another task."""
    while True:
        queue.collect()
        queue.flush_events()
        state = supervision_state(queue)
        if not state["running_jobs"] and not state["active_leases"]:
            return
        time.sleep(.2)


def _cancel_at_deadline(queue, stopped, deadline):
    while not stopped.wait(min(1., max(0., deadline - time.monotonic()))):
        if time.monotonic() < deadline:
            continue
        emit("elapsed_cleanup_started", maximum_elapsed_seconds=43200, cleanup_grace_seconds=30)
        cancel_campaign(queue, "cleanup grace before the 43200-second monotonic elapsed ceiling")
        return


def execute(queue_root, contract, entries, devices):
    devices = validate_devices(devices)
    frozen = verify_frozen(queue_root, contract)
    state_path = queue_root / "driver-state.json"
    clock_path = queue_root / "execution-clock.json"
    if state_path.exists():
        state = read_json(state_path)
        validate_resume_state(state, frozen, entries, read_json(clock_path))
        if state["study_binding_sha256"] != frozen["input_digest"] or state["boot_id"] != _boot_id():
            raise ValueError("resume requires the same frozen study and monotonic boot epoch")
        if state["status"] != "running":
            emit("already_terminal", status=state["status"], reason=state.get("stop_reason"))
            return report(queue_root, state)[1]
    else:
        now = time.monotonic(); wall = time.time()
        clock_record = {"schema_version": 1, "study_binding_sha256": frozen["input_digest"],
            "source": {"commit": frozen["source"]["origin_commit"], "digest": frozen["source"]["digest"]},
            "boot_id": _boot_id(), "started_at": utc_now(), "started_monotonic": now,
            "control_candidate_id": frozen["control_candidate_id"],
            "deadline_monotonic": now + 43200,
            "deadline_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wall + 43200))}
        if clock_path.exists():
            clock_record = read_json(clock_path)  # A crash before state persistence cannot reset elapsed time.
        else:
            clock_record["input_digest"] = stable_hash(clock_record)
            with clock_path.open("x", encoding="utf-8") as stream:
                json.dump(clock_record, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
        state = {k: deepcopy(v) for k, v in clock_record.items() if k != "input_digest"}
        state.update(status="running", stages=[], execution_clock_sha256=clock_record["input_digest"])
        validate_resume_state(state, frozen, entries, clock_record)
        if state["boot_id"] != _boot_id():
            raise ValueError("resume requires the same monotonic boot epoch")
        atomic_json(state_path, state)
    # Full timeout +30s launch headroom must end before the final30s cleanup window.
    queue = DeadlineQueue(queue_root, report_root=ROOT / "reports/forge",
                          deadline=state["deadline_monotonic"] - contract["cleanup_grace_seconds"],
                          cleanup_grace_seconds=contract["cleanup_grace_seconds"])
    os.environ["PARTICLEGAN_FORGE_OPTIMIZER_DIAGNOSTICS"] = "1"
    stopped = threading.Event()
    watchdog = threading.Thread(target=_cancel_at_deadline,
        args=(queue, stopped, state["deadline_monotonic"] - contract["cleanup_grace_seconds"]), daemon=True)
    watchdog.start()
    progress = threading.Thread(target=_progress, args=(queue, stopped), daemon=True)
    progress.start()
    try:
        for stage in ("A", "B", "C"):
            if time.monotonic() >= state["deadline_monotonic"] - contract["cleanup_grace_seconds"]:
                state["stop_reason"] = "elapsed ceiling reached before further stage admission"
                break
            recorded = next((s for s in state["stages"] if s["stage"] == stage), None)
            if recorded and not recorded.get("admitted") and recorded.get("spec"):
                recorded.update(admitted=True, admission_complete=False)
                atomic_json(state_path, state)
            if recorded and not recorded.get("admitted"):
                continue
            if recorded is None:
                if stage == "A":
                    entry, decision = entries[0], {"eligible": True, "reason": "fixed first-stage grid"}
                else:
                    studies, _ = report(queue_root, state, queue)
                    a_trials = studies[0]["trials"]
                    control = next(t for t in a_trials if t["candidate_id"] == frozen["control_candidate_id"])
                    considered = a_trials if stage == "B" else [t for study in studies for t in study["trials"]]
                    decision = activation(stage, considered, control, contract)
                    entry = branch_for(stage, decision["pace"], entries) if decision["eligible"] else None
                if time.monotonic() >= state["deadline_monotonic"]:
                    decision = {"eligible": False, "reason": "elapsed ceiling exhausted before stage admission"}
                    entry = None
                recorded = {"stage": stage, "decision": decision, "admitted": False}
                if entry:
                    recorded["spec"] = deepcopy(entry["spec"])
                    summary = plan_search(ROOT, queue_root, entry["spec"], queue=queue)
                    bound = next(b for b in frozen["branches"] if b["study_id"] == summary["study_id"])
                    if summary["source_digest"] != frozen["cohort"]["source_digest"] or (
                        [{k: t[k] for k in ("candidate_id", "configuration_id", "candidate_revision", "scientific_signature", "settings")}
                         for t in summary["trials"]] !=
                        [{k: t[k] for k in ("candidate_id", "configuration_id", "candidate_revision", "scientific_signature", "settings")}
                         for t in bound["trials"]]):
                        raise ValueError("conditional branch changed from the pre-spend plan")
                    recorded["admitted"] = True
                    recorded["admission_complete"] = False
                    state["stages"].append(recorded)
                    atomic_json(state_path, state)
                    # Forge's exact search registry is persisted before it submits any request.
                    verify_frozen(queue_root, contract)
                    enqueue_search(ROOT, queue_root, entry["spec"], queue=queue)
                    recorded["admission_complete"] = True
                    atomic_json(state_path, state)
                    emit("stage_admitted", stage=stage, study=entry["spec"]["id"], configurations=len(summary["trials"]))
                else:
                    state["stages"].append(recorded)
                    atomic_json(state_path, state)
                    emit("stage_skipped", stage=stage, reason=decision["reason"])
                    continue
            elif not recorded.get("admission_complete", True):
                # Explicit resume repairs an admission interruption by exact request
                # identity. Forge reuses existing submissions; it never retries a job.
                verify_frozen(queue_root, contract)
                enqueue_search(ROOT, queue_root, recorded["spec"], queue=queue)
                recorded["admission_complete"] = True
                atomic_json(state_path, state)
            verify_frozen(queue_root, contract)
            drain(queue, devices, workers_per_gpu=1, campaign=CAMPAIGN)
            summary = report_search(ROOT, queue_root, recorded["spec"], queue=queue)
            recorded["execution_complete"] = all(execution_complete(t) for t in summary["trials"])
            recorded["selection"] = whole_selection(summary["trials"])
            atomic_json(state_path, state)
            emit("stage_finished", stage=stage, execution_complete=recorded["execution_complete"],
                 selection=recorded["selection"])
            if not recorded["execution_complete"]:
                state["stop_reason"] = "stage incomplete; preserve unknown/error cells and do not activate conditional work"
                break
        finish_supervision(queue)
        admitted_stages = [s for s in state["stages"] if s["admitted"]]
        state["status"] = "completed" if admitted_stages and all(
            s.get("execution_complete", False) for s in admitted_stages) else "stopped_incomplete"
        state.setdefault("stop_reason", "finite declared sequence complete; conditional skips retained; no retries or expansion")
        state["deadline_admission_denials"] = queue.denied
        state["exit_code"] = 0 if state["status"] == "completed" else 2
        state["finished_at"] = utc_now()
        atomic_json(state_path, state)
        _, result = report(queue_root, state, queue)
        emit("campaign_terminal", status=state["status"], admitted_recipes=result["admitted_recipes"],
             counts=result["required_counts"], terminal_summary=str(queue_root / "terminal-summary.json"))
        return result
    except BaseException as error:
        emit("driver_error", error_type=type(error).__name__, reason=str(error))
        cancel_campaign(queue, "driver stopped: " + type(error).__name__)
        finish_supervision(queue)
        state.update(status="error", exit_code=1, stop_reason=f"{type(error).__name__}: {error}", finished_at=utc_now())
        atomic_json(state_path, state)
        try:
            return report(queue_root, state, queue)[1]
        except Exception as reporting_error:
            # A malformed registration must still produce truthful notification
            # data, with no inferred winner or fabricated numerical outcomes.
            supervision = supervision_state(queue)
            fallback = {"schema_version": 1, "campaign": CAMPAIGN, "terminal": True,
                "status": "error", "exit_code": 1, "stop_reason": state["stop_reason"],
                "reporting_error": str(reporting_error), "source": state["source"],
                "running_workers": len(supervision["running_jobs"]),
                "all_attempts_supervised_finished": not supervision["running_jobs"] and not supervision["active_leases"],
                "all_supervised_attempts_finished": not supervision["running_jobs"] and not supervision["active_leases"],
                "final_status": "error", "finished_at": state["finished_at"],
                "stages": state["stages"], "selection": None,
                "accounting": supervision["state"].get("campaigns", {}).get(CAMPAIGN, {})}
            atomic_json(queue_root / "terminal-summary.json", fallback)
            return fallback
    finally:
        stopped.set()
        watchdog.join(timeout=2)
        progress.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plan", "freeze", "run", "resume", "report"))
    parser.add_argument("--queue-root", type=Path, default=ROOT / "runs/forge/bcap-dualnorm-pacing-v2-queue")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    contract, a = load_inputs()
    entries = declared_specs(contract, a)
    queue_root = args.queue_root.resolve()
    if args.stage == "prepare":
        prepare(entries)
    elif args.stage == "plan":
        plan_all(queue_root, contract, entries)
    elif args.stage == "freeze":
        freeze(queue_root, contract, entries)
    elif args.stage == "report":
        report(queue_root, read_json(queue_root / "driver-state.json"))
    else:
        queue_root.mkdir(parents=True, exist_ok=True)
        with file_lock(queue_root / "driver.lock", blocking=False):
            result = execute(queue_root, contract, entries, args.gpus.split(","))
            raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
