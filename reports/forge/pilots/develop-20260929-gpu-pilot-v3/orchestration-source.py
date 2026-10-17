#!/usr/bin/env python3
"""Prepare/check or execute the already registered two-GPU Forge pilot.

Default: read-only preflight. --execute performs the registered work. This does
not register, change tasks, add steps, sleep a training process, sweep seeds,
or retry scientific failures. Supply final frozen paths/identities from root.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback


REGISTRATION = "develop-20260929-gpu-pilot-v3"
LINEAGES = {"k3p", "forge-no-critic-penalty"}
TASK = "vector_two_broad"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--fresh-clone", type=Path, required=True)
    p.add_argument("--queue-root", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--expected-source-digest", required=True)
    p.add_argument("--output", type=Path, required=True, help="new orchestration evidence directory")
    p.add_argument("--ownership-note", required=True, help="record the established joint device window")
    p.add_argument("--allow-compute-pid", type=int, action="append", default=[],
                   help="explicit preserved desktop PID (this pilot allows only the reviewed PIDs 11101 and 62189)")
    p.add_argument("--execute", action="store_true", help="enqueue and execute only this registered pilot")
    return p.parse_args()


def json_file(path):
    return json.loads(Path(path).read_text())


def maybe_json(path):
    try:
        return json_file(path)
    except (OSError, ValueError):
        return None


def command_output(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True, stderr=subprocess.PIPE).strip()


def gpu_inventory():
    rows = command_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.free", "--format=csv,noheader,nounits"])
    return {r[0].strip(): {"uuid": r[1].strip(), "name": r[2].strip(), "free_mb": int(r[3])}
            for r in csv.reader(rows.splitlines())}


def gpu_processes():
    rows = command_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"])
    return [{"pid": int(r[0]), "gpu_uuid": r[1].strip(), "used_memory_mb": r[2].strip()}
            for r in csv.reader(rows.splitlines()) if len(r) == 3 and r[0].strip().isdigit()]


def command_line(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
    except OSError:
        return None


class Pilot:
    def __init__(self, args):
        self.args = args
        self.root = args.checkout.resolve()
        self.clone = args.fresh_clone.resolve()
        self.queue_root = args.queue_root.resolve()
        self.output = args.output.resolve()
        self.campaign = "calibration-" + REGISTRATION
        self.coordinator = None
        self.coordinators = []
        self.started = time.monotonic()
        self.event_index = 0
        self.checks = {}
        self.missed = []
        self.request_ids = {}
        self.selected_keys = {}
        self.observed_children = {}
        self.did_cancel = False
        self.did_retry = False
        sys.path.insert(0, str(self.root))
        from experiments.forge.queue import Queue, lease_held, process_identity
        self.queue = Queue(self.queue_root, report_root=self.root / "reports/forge")
        self.lease_held, self.process_identity = lease_held, process_identity

    def event(self, event, **values):
        row = {"event": event, "time": time.time(), "elapsed_seconds": time.monotonic() - self.started, **values}
        print(json.dumps(row, sort_keys=True), flush=True)
        if self.output.exists():
            with (self.output / "orchestration.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def save(self, name, value):
        (self.output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def cli(self, root, *arguments):
        return [sys.executable, "-u", "-m", "experiments.forge", "--root", str(root),
                "--queue-root", str(self.queue_root), *arguments]

    def environment(self, root):
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root))
        return env

    def invoke(self, root, label, *arguments):
        with (self.output / f"{label}.stderr.log").open("w") as errors:
            result = subprocess.run(self.cli(root, *arguments), cwd=root, env=self.environment(root),
                                    stdout=subprocess.PIPE, stderr=errors, text=True, timeout=180)
        (self.output / f"{label}.json").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"{label} exited {result.returncode}; inspect its stderr log")
        return json.loads(result.stdout)

    def preflight(self):
        from experiments.forge.calibration_lane import plan_calibration
        from experiments.forge.sources import inspect_source
        if self.root == self.clone or self.output.exists():
            raise ValueError("require distinct checkouts and a new output directory")
        registration_path = Path("reports/forge/calibration-lanes") / REGISTRATION / "registration.json"
        registration = json_file(self.root / registration_path)
        declared_sources = [subject["base_request"]["source"] for subject in registration["subjects"].values()]
        if any(source["files"] != declared_sources[0]["files"] for source in declared_sources):
            raise ValueError("pilot lineages do not share one declared scientific source")
        commons = []
        manifests = []
        for root in (self.root, self.clone):
            commit = command_output(["git", "rev-parse", "HEAD"], root)
            if commit != self.args.expected_commit:
                raise ValueError(f"{root} is not the final expected commit: {commit}")
            common = command_output(["git", "rev-parse", "--git-common-dir"], root)
            commons.append((root / common).resolve())
            # The registered evaluator adds a pinned historical convergence
            # module outside default source directories; hash that input too.
            manifest = inspect_source(root, extra_paths=declared_sources[0]["files"])
            if manifest["digest"] != self.args.expected_source_digest:
                raise ValueError(f"{root} source differs from the final frozen digest")
            manifests.append(manifest)
        if commons[0] == commons[1]:
            raise ValueError("fresh checkout must be an independent clone, not another linked worktree")
        if manifests[0] != manifests[1]:
            raise ValueError("checkout source manifests differ")
        if registration != json_file(self.clone / registration_path):
            raise ValueError("fresh clone registration differs; clone the finalized registration")
        contract = registration["contract"]
        if contract["id"] != REGISTRATION or contract["qualification_reuse"] is not False:
            raise ValueError("unexpected diagnostic registration")
        if {row["lineage_id"]: row["tasks"] for row in contract["selections"]} != {name: [TASK] for name in LINEAGES}:
            raise ValueError("pilot must select exactly its two substantive vector cells")
        if contract["budgets"] != {"campaign_seconds": 5400, "candidate_seconds": 3600,
                                   "task_seconds": {TASK: 1800}}:
            raise ValueError("unexpected pilot budgets")
        if contract["execution_backend"] != "cuda" or contract["cuda_model"] != "NVIDIA RTX A6000":
            raise ValueError("pilot requires its registered physical CUDA cohort")
        requests = plan_calibration(self.root, REGISTRATION, self.queue_root, freeze_source=False)
        clone_requests = plan_calibration(self.clone, REGISTRATION, self.queue_root, freeze_source=False)
        if requests != clone_requests:
            raise ValueError("fresh clone and implementation checkout resolve different requests")
        state = self.queue.inspect()
        if self.campaign in state["campaigns"]:
            raise ValueError("pilot campaign already exists; inspect existing evidence instead of repeating")
        if any(job["status"] == "running" for job in state["jobs"].values()):
            raise ValueError("another Forge job is running in the common queue")
        for request in requests:
            lineage = request["calibration_lane"]["lineage_id"]
            if request["tasks"][TASK]["execution"]["steps"] != 1200:
                raise ValueError("pilot task no longer has exactly 1200 updates")
            selected = [job for job in request["jobs"] if job["task_id"] == TASK]
            if len(selected) != 1 or selected[0]["budget_seconds"] != 1800:
                raise ValueError("unexpected pilot job declaration")
            key = selected[0]["compatibility_key"]
            if state["jobs"].get(key, {}).get("attempts"):
                raise ValueError("compatible execution already exists; do not repeat it for a pilot")
            self.selected_keys[lineage] = key
        devices = gpu_inventory()
        if any(devices.get(index, {}).get("name") != contract["cuda_model"] for index in ("0", "1")):
            raise ValueError("physical devices 0 and 1 do not match the registered model")
        apps = gpu_processes()
        relevant = {devices[index]["uuid"] for index in ("0", "1")}
        allowed = set(self.args.allow_compute_pid)
        if not allowed <= {11101, 62189}:
            raise ValueError("only the two specifically reviewed desktop PIDs may be preserved by this script")
        self.preserved_desktop = {str(pid): {"pid": pid, "process_identity": self.process_identity(pid),
                                  "command_line": command_line(pid)} for pid in allowed}
        if any(not row["process_identity"] or not row["command_line"] for row in self.preserved_desktop.values()):
            raise ValueError("a supplied preserved desktop process no longer exists; review ownership again")
        if any(app["gpu_uuid"] in relevant and app["pid"] not in allowed for app in apps):
            raise ValueError("compute processes already occupy pilot GPUs; do not interfere with them")
        self.devices = devices
        return {"registration_id": REGISTRATION, "campaign_id": self.campaign,
                "commit": self.args.expected_commit, "source_digest": manifests[0]["digest"],
                "ownership_note": self.args.ownership_note, "devices": devices, "compute_processes": apps,
                "preserved_desktop_processes": self.preserved_desktop,
                "selected_keys": self.selected_keys, "queue_root": str(self.queue_root),
                "checkout": str(self.root), "fresh_clone": str(self.clone), "budgets": contract["budgets"],
                "training_launched": False, "execute_requested": self.args.execute}

    def enqueue(self, root, label):
        rows = self.invoke(root, label, "calibration-lane", "enqueue", REGISTRATION)
        ids = {row["calibration_lane"]["lineage_id"]: row["request_id"] for row in rows}
        if set(ids) != LINEAGES:
            raise ValueError("unexpected enqueue result")
        return ids

    def start_coordinator(self):
        index = len(self.coordinators) + 1
        path = self.output / f"coordinator-{index}.log"
        with path.open("ab", buffering=0) as log:
            process = subprocess.Popen(self.cli(self.root, "drain", "--gpus", "0,1", "--workers-per-gpu", "1",
                "--campaign", self.campaign, "--watch"), cwd=self.root, env=self.environment(self.root),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.coordinator = process
        self.coordinators.append({"pid": process.pid, "identity": self.process_identity(process.pid), "log": str(path)})
        self.event("coordinator_started", **self.coordinators[-1])

    def stop_coordinator(self, reason):
        process = self.coordinator
        if process is None or process.poll() is not None:
            return
        identity = self.coordinators[-1]["identity"]
        if self.process_identity(process.pid) != identity:
            raise RuntimeError("coordinator PID identity changed; refusing to signal")
        # Never signal the coordinator process group, worker, or runtime here.
        os.kill(process.pid, signal.SIGINT)
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            raise RuntimeError("coordinator did not release its lock after SIGINT; inspect manually")
        self.event("coordinator_interrupted", pid=process.pid, process_identity=identity,
                   reason=reason, returncode=code, signal="SIGINT", target="coordinator PID only")

    def selected(self):
        state = self.queue.inspect()
        return state, {lineage: state["jobs"][key] for lineage, key in self.selected_keys.items()}

    def live_runtime(self, lineage, job):
        if job["status"] != "running":
            return None
        worker = job["worker"]
        directory = Path(worker["directory"])
        child = maybe_json(directory / "child.json")
        if not child or child.get("phase") == "evaluation" or child.get("token") != worker["token"]:
            return None
        if self.process_identity(child["pid"]) != child.get("process_identity"):
            return None
        if self.process_identity(worker.get("pid")) != worker.get("process_identity"):
            return None
        if not self.lease_held(directory / "execution.lock"):
            return None
        try:
            observations = [json.loads(line) for line in (directory / "run.log").read_text().splitlines()
                            if line.startswith('{"event": "observation"')]
        except (OSError, ValueError):
            return None
        if not observations:
            return None
        snapshot = {"lineage": lineage, "attempt": worker["attempt"], "token": worker["token"],
                    "device": worker["device"], "worker_pid": worker["pid"],
                    "worker_identity": worker["process_identity"], "child": child,
                    "latest_step": observations[-1]["step"], "directory": str(directory), "lease_held": True}
        self.observed_children[worker["attempt"]] = deepcopy(snapshot)
        return snapshot

    def both_physical(self, jobs):
        alive = {lineage: self.live_runtime(lineage, job) for lineage, job in jobs.items()}
        if any(row is None for row in alive.values()) or {row["device"] for row in alive.values()} != {"0", "1"}:
            return None
        apps = gpu_processes()
        for row in alive.values():
            matches = [app for app in apps if app["pid"] == row["child"]["pid"]
                       and app["gpu_uuid"] == self.devices[row["device"]]["uuid"]]
            if not matches:
                return None
            row["physical_gpu_process"] = matches[0]
        return alive

    @staticmethod
    def identity(row):
        return {key: row[key] for key in ("attempt", "token", "device", "worker_pid", "worker_identity", "child")}

    def await_overlap(self):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            _, jobs = self.selected()
            witness = self.both_physical(jobs)
            if witness:
                self.checks["physical_overlap"] = self.event("two_real_gpu_children_observed", children=witness)
                return witness
            if any(job["status"] == "terminal" for job in jobs.values()):
                raise RuntimeError("an original attempt finished before both physical children were witnessed")
            if self.coordinator.poll() is not None:
                raise RuntimeError("coordinator exited before both original workers started")
            time.sleep(.05)  # Coordinator observation only; no training process is delayed.
        raise RuntimeError("two physical workers were not observed within the startup window")

    def restart_and_cancel(self, original):
        self.stop_coordinator("registered live-worker recovery check")
        status = self.queue_root / "status.json"
        previous_mtime = status.stat().st_mtime_ns if status.exists() else None
        _, jobs = self.selected()
        for lineage, job in jobs.items():
            live = self.live_runtime(lineage, job)
            if live is None or self.identity(live) != self.identity(original[lineage]):
                raise RuntimeError("original child finished before coordinator restart; do not repeat science")
        self.start_coordinator()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            _, jobs = self.selected()
            witness = self.both_physical(jobs)
            refreshed = status.exists() and status.stat().st_mtime_ns != previous_mtime
            if refreshed and witness and all(self.identity(witness[name]) == self.identity(original[name]) for name in LINEAGES):
                self.checks["recovery"] = self.event("same_live_leases_recovered", children=witness,
                    coordinator_pid=self.coordinator.pid, status_mtime_ns=status.stat().st_mtime_ns)
                break
            if any(job["status"] == "terminal" for job in jobs.values()):
                raise RuntimeError("live recovery window ended before restart evidence; no extra attempts")
            if self.coordinator.poll() is not None:
                raise RuntimeError("replacement coordinator exited before recovery")
            time.sleep(.025)
        else:
            raise RuntimeError("replacement coordinator did not demonstrate recovery")
        target = "forge-no-critic-penalty"
        # Recheck the exact runtime identity immediately before Queue.cancel.
        _, jobs = self.selected()
        live = self.live_runtime(target, jobs[target])
        if live is None or self.identity(live) != self.identity(original[target]):
            raise RuntimeError("penalty child finished before cancellation; never retry its scientific result")
        self.event("cancellation_requested", request_id=self.request_ids[target], child=live)
        self.queue.cancel(self.request_ids[target])
        self.did_cancel = True
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            _, jobs = self.selected()
            job = jobs[target]
            if job["status"] == "terminal":
                if job["result"]["raw"]["attempt_status"] != "cancelled":
                    raise RuntimeError("cancellation lost its runtime window; result retained without retry")
                if self.process_identity(live["child"]["pid"]) or self.process_identity(live["worker_pid"]):
                    raise RuntimeError("cancelled attempt still has a live original process")
                if self.lease_held(Path(live["directory"]) / "execution.lock"):
                    raise RuntimeError("cancelled execution lease remains held")
                if job.get("reserved_seconds", 0) != 0:
                    raise RuntimeError("cancelled job reservation was not released")
                self.checks["cancellation"] = self.event("cancelled_attempt_collected", attempt=live["attempt"],
                    charged_seconds=job["result"]["raw"]["elapsed_seconds"], original_processes_exited=True)
                break
            time.sleep(.025)
        else:
            raise RuntimeError("cancelled attempt was not collected; no repair attempted")
        ids = self.enqueue(self.root, "reattach-after-cancellation")
        if ids != self.request_ids:
            raise RuntimeError("reattachment changed request identity")
        _, jobs = self.selected()
        if len(jobs[target]["attempts"]) != 1:
            raise RuntimeError("repair would exceed one deliberate cancellation plus one retry")
        self.queue.retry(self.selected_keys[target], reason=("Repair the registered deliberate cancellation after the physical two-GPU "
            "coordinator-recovery check; original cancelled receipt and cost retained. One repair only."))
        self.did_retry = True
        self.event("one_repair_authorized", compatibility_key=self.selected_keys[target], cancelled_attempt=live["attempt"])

    def finish_existing(self):
        if self.coordinator is None or self.coordinator.poll() is not None:
            self.start_coordinator()
        deadline = time.monotonic() + 5460
        last_update = 0
        while time.monotonic() < deadline:
            state, jobs = self.selected()
            terminal = all(job["status"] == "terminal" for job in jobs.values())
            if terminal and state["campaigns"][self.campaign]["reserved_seconds"] == 0:
                self.stop_coordinator("registered selected work completed")
                return
            if self.coordinator.poll() is not None:
                raise RuntimeError("drain exited unexpectedly; bounded workers retain their own deadlines")
            if time.monotonic() - last_update > 15:
                self.event("progress", jobs={name: {"status": job["status"], "attempts": len(job["attempts"])}
                    for name, job in jobs.items()}, campaign=state["campaigns"][self.campaign])
                last_update = time.monotonic()
            entries = [state["submissions"][value] for value in self.request_ids.values()]
            if not any(job["status"] == "running" for job in jobs.values()) and any(entry["status"] == "blocked" for entry in entries):
                self.stop_coordinator("campaign blocked; no added work")
                raise RuntimeError("pilot blocked; inspect frozen blockers")
            time.sleep(.1)
        raise RuntimeError("orchestration wall ceiling reached; inspect bounded workers and do not launch more")

    def gather(self):
        from experiments.forge.telemetry import summarize_automation
        state, jobs = self.selected()
        self.save("queue-final.json", state)
        paid = [row for row in state.get("charges", []) if row["owner"]["campaign"] == self.campaign]
        attempts = []
        for lineage, job in jobs.items():
            for attempt in job["attempts"]:
                directory = Path(attempt["path"])
                copy = self.output / "attempts" / attempt["attempt_id"]
                copy.mkdir(parents=True, exist_ok=True)
                for name in ("request.json", "process.json", "child.json", "heartbeat.json", "terminal.json", "run.log",
                             "cancel.json", "raw-result.json", "graded-result.json", "adapter-receipt.json"):
                    source = directory / name
                    if source.is_file():
                        shutil.copyfile(source, copy / name)
                durable = self.root / "reports/forge/attempts" / attempt["attempt_id"]
                if durable.is_dir():
                    shutil.copytree(durable, copy / "durable", dirs_exist_ok=True)
                process = maybe_json(directory / "process.json") or {}
                child = maybe_json(directory / "child.json") or {}
                terminal = maybe_json(directory / "terminal.json") or {}
                attempts.append({"lineage": lineage, "attempt": attempt["attempt_id"], "path": str(directory),
                    "raw_status": terminal.get("attempt_status"), "elapsed_seconds": terminal.get("elapsed_seconds"),
                    "device": process.get("device"), "worker_alive": self.process_identity(process.get("pid")) is not None,
                    "last_child_alive": self.process_identity(child.get("pid")) is not None,
                    "lease_held": self.lease_held(directory / "execution.lock"), "telemetry": terminal.get("telemetry")})
        for name, source in (("events.jsonl", self.queue_root / "events.jsonl"),
                             ("campaign-progress.jsonl", self.queue_root / self.campaign / "progress.jsonl")):
            if source.exists():
                shutil.copyfile(source, self.output / name)
        self.save("automation-global.json", summarize_automation(self.root, self.queue_root))
        campaign = state["campaigns"][self.campaign]
        self.checks["zero_reservations"] = campaign["reserved_seconds"] == 0 and all(job.get("reserved_seconds", 0) == 0 for job in jobs.values())
        self.checks["no_live_selected_workers"] = all(not (row["worker_alive"] or row["last_child_alive"] or row["lease_held"]) for row in attempts)
        self.checks["paid_once_per_attempt"] = len(paid) == len(attempts) and len({row["attempt_id"] for row in paid}) == len(paid)
        self.checks["within_registered_budget"] = campaign["spent_seconds"] <= 5400 and all(
            sum(row["elapsed_seconds"] or 0 for row in attempts if row["lineage"] == name) <= 3600 for name in LINEAGES)
        self.checks["exact_bounded_execution_count"] = (len(attempts) == 3 and
            sum(row["raw_status"] == "cancelled" for row in attempts) == 1 and
            sum(row["raw_status"] == "completed" for row in attempts) == 2 and self.did_retry)
        desktop_after = {pid: {"pid": row["pid"], "process_identity": self.process_identity(row["pid"]),
                              "command_line": command_line(row["pid"])} for pid, row in self.preserved_desktop.items()}
        self.checks["preserved_desktop_processes_unchanged"] = desktop_after == self.preserved_desktop
        summary = {"registration": REGISTRATION, "campaign": self.campaign, "checks": self.checks,
            "missed_checks": self.missed, "attempts": attempts, "charges": paid, "campaign_cost": campaign,
            "request_ids": self.request_ids, "selected_keys": self.selected_keys,
            "verdicts": {name: (job.get("result") or {}).get("task_results") for name, job in jobs.items()},
            "coordinators": self.coordinators, "qualification_reuse": False,
            "preserved_desktop_before": self.preserved_desktop, "preserved_desktop_after": desktop_after,
            "scientific_adoption": "not established by this operational pilot",
            "scope_note": "automation-global.json covers all durable history; campaign_cost and attempts above are pilot-specific.",
            "tail": f"tail -F {self.output / 'orchestration.jsonl'} {self.queue_root / 'events.jsonl'}"}
        summary["operational_acceptance"] = (not self.missed and all(bool(self.checks.get(key)) for key in (
            "duplicate_submission", "physical_overlap", "recovery", "cancellation", "zero_reservations",
            "no_live_selected_workers", "paid_once_per_attempt", "within_registered_budget", "exact_bounded_execution_count",
            "preserved_desktop_processes_unchanged")))
        self.save("summary.json", summary)
        self.event("pilot_finished", operational_acceptance=summary["operational_acceptance"],
                   summary=str(self.output / "summary.json"), campaign_spent_seconds=campaign["spent_seconds"])
        return summary

    def execute(self, preflight):
        self.output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(Path(__file__), self.output / "orchestration-source.py")
        self.save("preflight.json", preflight)
        self.event("pilot_started", registration=REGISTRATION, ownership_note=self.args.ownership_note)
        self.request_ids = self.enqueue(self.root, "enqueue-checkout")
        before = self.queue.inspect()
        duplicate = self.enqueue(self.clone, "enqueue-fresh-clone")
        after = self.queue.inspect()
        selected_before = {key: before["jobs"][key] for key in self.selected_keys.values()}
        selected_after = {key: after["jobs"][key] for key in self.selected_keys.values()}
        if duplicate != self.request_ids or selected_before != selected_after:
            raise RuntimeError("duplicate enqueue created a different request, execution, subscriber or reservation")
        if after["campaigns"][self.campaign]["reserved_seconds"] != 0:
            raise RuntimeError("enqueue unexpectedly reserved compute before a drain")
        self.checks["duplicate_submission"] = self.event("fresh_clone_deduplicated", request_ids=duplicate,
                                                         attempts=0, reserved_seconds=0)
        try:
            self.start_coordinator()
            witness = self.await_overlap()
            self.restart_and_cancel(witness)
        except Exception as error:
            self.missed.append(str(error))
            self.event("operational_check_missed", error=str(error), extra_scientific_repeats=False)
            (self.output / "operational-error.txt").write_text(traceback.format_exc())
        try:
            self.finish_existing()
        except Exception as error:
            self.missed.append(str(error))
            self.event("completion_problem", error=str(error))
        return self.gather()


def main():
    args = arguments()
    pilot = Pilot(args)
    preflight = pilot.preflight()
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    return 0 if pilot.execute(preflight)["operational_acceptance"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
