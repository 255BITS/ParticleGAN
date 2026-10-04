"""Dedicated policy execution ownership, sharing Forge's admission transaction.

Policy receipts retain their cloud/served cohort. They are not Queue jobs and
cannot supply MoG qualification. The shared ledger owns physical attempts; study
archives own their existing scientific gates and conservative budget readouts.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .contracts import atomic_json, read_json, stable_hash, utc_now
from .queue import Queue, fence_orphan_group, host_capacity, lease_held, process_identity
from .sources import inspect_source, snapshot_source, verify_snapshot


def freeze_source(root, queue_root, expected):
    root = Path(root).resolve()
    # Providers load published host scripts and examples lazily. Snapshot those
    # sources, not their checkpoints, observation streams or training artifacts.
    extras = [str(path.relative_to(root)) for directory in ("examples", "reports")
              for path in (root / directory).rglob("*.py")]
    catalog = root / "reports/toy_audit/catalog.json"
    if catalog.is_file():
        extras.append(str(catalog.relative_to(root)))
    manifest = inspect_source(root, extras)
    if any(manifest["files"].get(path) != digest for path, digest in expected["files_sha256"].items()):
        raise ValueError("source changed between policy planning and submission")
    # A docs-only commit during submission cannot replace the planned origin.
    # The content digest still binds every captured execution file.
    manifest["origin_commit"] = expected["commit"]
    # Source bytes deduplicate science; an origin-specific storage namespace
    # preserves the planned commit after docs-only commits with identical bytes.
    namespace = Path(queue_root) / "policy/source-origins" / stable_hash(expected["commit"])
    destination = snapshot_source(root, namespace, manifest)
    return {**manifest, "snapshot_path": str(destination)}


def physical_device(device):
    if str(device) == "cpu":
        return "cpu"
    index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        members = [item.strip() for item in visible.split(",") if item.strip()]
        if index >= len(members):
            raise ValueError("requested CUDA index is outside CUDA_VISIBLE_DEVICES")
        if not all(member.isdigit() for member in members):
            raise ValueError("shared policy admission requires numeric CUDA_VISIBLE_DEVICES indices; UUID/MIG masks are unresolved")
        return str(int(members[index]))
    return str(index)


def device_key(device):
    value = str(device)
    return str(int(value)) if value.isdigit() else value


def scientific_runtime(runtime):
    """Physical CUDA placement is admission, not a new scientific cohort."""
    return {**runtime, "device": str(runtime.get("device", "cpu")).split(":", 1)[0]}


def _released_attempt(entry):
    directory = Path(entry["lease_path"]).parent
    terminal_path = directory / "supervisor-terminal.json"
    terminal = read_json(terminal_path) if terminal_path.exists() else None
    if terminal and terminal.get("token") != entry.get("token"):
        raise RuntimeError("fenced policy supervisor published a stale terminal receipt")
    measured = terminal.get("paid_wall_seconds", 0.) if terminal else 0.
    if type(measured) not in (int, float) or not math.isfinite(measured) or measured < 0:
        raise ValueError("invalid durable policy cost")
    if terminal and terminal["attempt_status"] == "completed":
        entry.update(status="awaiting_certification", terminal=terminal,
                     charged_seconds=measured, completed_at=utc_now())
    else:
        entry.update(status="interrupted", charged_seconds=max(entry["allowance_seconds"], measured), terminal=terminal,
                     completed_at=utc_now(), result=None,
                     reason="released execution lease without a committed terminal receipt")


def recover_attempts(state):
    """Preserve live attempts before their durable deadline; fence after it."""
    for entry in state.get("policy_attempts", {}).values():
        if entry["status"] != "running":
            continue
        directory = Path(entry["lease_path"]).parent
        if lease_held(Path(entry["lease_path"])):
            if entry.get("deadline_monotonic") and time.monotonic() >= entry["deadline_monotonic"]:
                child_path = directory / "child.json"
                if child_path.exists():
                    child = read_json(child_path)
                    if child.get("token") == entry.get("token"):
                        fence_orphan_group(child, directory)
                supervisor = entry.get("supervisor")
                supervisor_path = directory / "supervisor.json"
                if supervisor is None and supervisor_path.exists():
                    recorded = read_json(supervisor_path)
                    if recorded.get("token") == entry.get("token"):
                        supervisor = recorded
                if supervisor and process_identity(supervisor["pid"]) == supervisor["process_identity"]:
                    try:
                        os.kill(supervisor["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            # Killing a group does not prove its descriptors are closed yet.
            if lease_held(Path(entry["lease_path"])):
                continue
        _released_attempt(entry)


def active_reservations(state):
    return [entry for entry in state.get("policy_attempts", {}).values() if entry["status"] == "running"]


@contextmanager
def execution_lease(path):
    """Close without LOCK_UN: inherited children retain ownership after a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        yield handle


class PolicyCoordinator:
    def __init__(self, root, *, report_root=None):
        self.queue = Queue(root, report_root=report_root)
        self.root = self.queue.root

    def register(self, packet, output, family, runtime):
        """Different output names attach to one canonical compatible study."""
        spec = {key: value for key, value in packet["spec"].items() if key != "id"}
        spec["representation_card"] = {"sha256": spec["representation_card"]["sha256"]}
        identity = {"cohort": "cloud-served-policy-v1", "spec": spec,
                    "source": packet["execution_source"]["digest"], "cases": packet["case_definitions"],
                    "family": family, "runtime": scientific_runtime(runtime)}
        key = stable_hash(identity)
        output = Path(output).resolve()
        with self.queue.state() as state:
            studies = state.setdefault("policy_studies", {})
            outputs = state.setdefault("policy_outputs", {})
            if str(output) in outputs and outputs[str(output)] != key:
                saved = read_json(output / "study.json")
                if saved.get("executed_family") != family or saved.get("lane_runtime") != runtime:
                    raise ValueError("one family/fixed runtime per output; use separate family archives")
                raise ValueError("study identity changed; use a new study ID/output")
            existing = studies.get(key)
            if existing:
                canonical = Path(existing["output"])
            else:
                canonical = output
                registration = canonical / "study.json"
                if registration.exists():
                    saved = read_json(registration)
                    for field in ("spec_sha256", "spec", "source", "case_definitions", "capacity_preflight",
                                  "runtime_contract", "family_paid_budget_seconds"):
                        if saved.get(field) != packet.get(field):
                            raise ValueError("study identity changed; use a new study ID/output")
                    if saved.get("executed_family") != family or saved.get("lane_runtime") != runtime:
                        raise ValueError("one family/fixed runtime per output; use separate family archives")
                    saved.setdefault("execution_source", packet["execution_source"])
                    packet = saved
                elif canonical.exists() and any(canonical.iterdir()):
                    raise ValueError("unregistered nonempty archive cannot be reused as a fresh study")
                packet.update(executed_family=family, lane_runtime=runtime)
                packet.setdefault("spent_seconds", 0.)
                packet["coordinator"] = {"study_key": key, "queue_root": str(self.root),
                                         "canonical_output": str(canonical)}
                atomic_json(registration, packet)
                studies[key] = {"identity": identity, "output": str(canonical), "created_at": utc_now()}
            outputs[str(output)] = key
        return key, canonical

    def publish_attachment(self, key, canonical, requested):
        packet = read_json(Path(canonical) / "study.json")
        if Path(requested).resolve() != Path(canonical).resolve():
            packet["coordinator"] = {**packet["coordinator"], "attached": True}
            atomic_json(Path(requested) / "study.json", packet)
        return packet

    def study_lease(self, key):
        return execution_lease(self.root / "policy/studies" / f"{key}.lock")

    def recover(self):
        with self.queue.state() as state:
            recover_attempts(state)

    def retained(self, key):
        with self.queue.state() as state:
            recover_attempts(state)
            return state.get("policy_attempts", {}).get(key)

    def attempt_key(self, packet, trial, row):
        identity = {"cohort": "cloud-served-policy-v1", "case": packet["case_definitions"][row["id"]],
                    "family": trial["family"], "recipe": trial["recipe_overrides"],
                    "source_digest": packet["execution_source"]["digest"],
                    "runtime": scientific_runtime(packet["lane_runtime"]), "timeout_seconds": row["timeout_seconds"],
                    "export_grace_seconds": packet["spec"]["export_grace_seconds"],
                    "frames": packet["spec"].get("frames", 9)}
        # Only this source-bound one-case envelope has a recognized fresh repeat.
        # Ordinary policy studies retain their released identity byte-for-byte.
        schema = packet.get("schema", "")
        if ((isinstance(schema, str) and schema.startswith("pg_canonical_two_pole_first_case"))
                or str(row.get("id", "")).startswith("canonical-two-pole-full-atlas-")
                or "scientific_repeat" in packet or "scientific_repeat" in packet.get("protocol", {})):
            from .canonical_two_pole_repeat import coordinator_repeat
            identity["scientific_repeat"] = coordinator_repeat(packet, trial, row)
        return stable_hash(identity)

    @contextmanager
    def admit(self, key, packet, row, device):
        """Reserve complete allowance and exclusive GPU admission atomically."""
        self.queue.collect(only_orphans=True)
        self.recover()
        path = self.root / "policy/attempts" / key / "execution.lock"
        with execution_lease(path) as lease:
            if lease is None:
                yield {"status": "busy", "reason": "compatible physical attempt has a live execution lease"}, None
                return
            with self.queue.state() as state:
                recover_attempts(state)
                existing = state.setdefault("policy_attempts", {}).get(key)
                if existing and existing["status"] == "running":
                    # We acquired the old lease; there is no surviving owner.
                    _released_attempt(existing)
                if existing:
                    decision = existing
                else:
                    capacity = host_capacity()
                    resources = packet["spec"].get("resources", {})
                    needed = {"cpu_threads": packet["lane_runtime"]["torch_threads"],
                              "host_memory_mb": resources.get("host_memory_mb", 512)}
                    if (type(needed["cpu_threads"]) is not int or needed["cpu_threads"] < 1
                            or type(needed["host_memory_mb"]) not in (int, float)
                            or not 512 <= needed["host_memory_mb"] <= capacity["memory_mb"]):
                        raise ValueError("invalid policy host reservation; declare at least 512 MiB")
                    core = [entry["worker"] for entry in state["jobs"].values() if entry["status"] == "running"]
                    active = [entry["worker"] for entry in active_reservations(state)]
                    workers = core + active
                    target = physical_device(device)
                    occupied = target != "cpu" and any(device_key(worker["device"]) == target for worker in workers)
                    reservations = [worker.get("host_reservation", {"cpu_threads": 1, "host_memory_mb": 512})
                                    for worker in workers]
                    if (occupied or needed["cpu_threads"] + sum(r["cpu_threads"] for r in reservations) > capacity["cpu_threads"]
                            or needed["host_memory_mb"] + sum(r["host_memory_mb"] for r in reservations) > capacity["available_memory_mb"]):
                        decision = {"status": "busy", "reason": "shared Forge host/GPU admission unavailable"}
                    else:
                        decision = {"status": "running", "lease_path": str(path),
                                    "allowance_seconds": row["timeout_seconds"] + packet["spec"]["export_grace_seconds"],
                                    "token": os.urandom(16).hex(), "started_monotonic": time.monotonic(),
                                    "source": packet["source"],
                                    "runtime": packet["lane_runtime"],
                                    "started_at": utc_now(), "charged_seconds": 0., "result": None,
                                    "worker": {"device": target, "slot": 0, "exclusive_device": target != "cpu",
                                               "host_reservation": needed}}
                        decision["deadline_monotonic"] = decision["started_monotonic"] + decision["allowance_seconds"]
                        state["policy_attempts"][key] = decision
            yield decision, lease if (decision["status"] == "running" and existing is None
                                       or decision["status"] == "awaiting_certification") else None

    def launch(self, command, packet, log, leases, allowance):
        snapshot = Path(packet["execution_source"]["snapshot_path"])
        verify_snapshot(snapshot, packet["execution_source"])
        expected = {key: value for key, value in packet["execution_source"].items() if key != "snapshot_path"}
        if read_json(snapshot / "forge-source.json") != expected:
            raise ValueError("policy source provenance metadata changed")
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(snapshot), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        directory = Path(leases[-1].name).parent
        key = directory.name
        with self.queue.state() as state:
            entry = state["policy_attempts"][key]
            if entry["status"] != "running" or entry["allowance_seconds"] != allowance:
                raise RuntimeError("policy reservation changed before supervised launch")
            entry.update(command=command, log_path=str(log))
            request = {"command": command, "source": packet["execution_source"], "log_path": str(log),
                       "token": entry["token"], "started_monotonic": entry["started_monotonic"],
                       "deadline_monotonic": entry["deadline_monotonic"],
                       "lease_fds": [lease.fileno() for lease in leases]}
            atomic_json(directory / "supervisor-request.json", request)
            with Path(log).open("wb") as stream:
                supervisor = subprocess.Popen([sys.executable, "-u", "-m", "experiments.forge.policy_execution",
                                                str(directory / "supervisor-request.json")],
                                               cwd=snapshot, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True, pass_fds=tuple(request["lease_fds"]))
            entry["supervisor"] = {"pid": supervisor.pid, "process_identity": process_identity(supervisor.pid)}
        try:
            supervisor.wait(timeout=max(0., request["deadline_monotonic"] - time.monotonic()) + .5)
        except BaseException:
            child_path = directory / "child.json"
            if child_path.exists():
                fence_orphan_group(read_json(child_path), directory)
            if process_identity(supervisor.pid) == entry["supervisor"]["process_identity"]:
                supervisor.kill()
            supervisor.wait()
            raise
        terminal_path = directory / "supervisor-terminal.json"
        if not terminal_path.exists() and time.monotonic() >= request["deadline_monotonic"]:
            error = subprocess.TimeoutExpired(command, allowance)
            error.paid_wall_seconds = time.monotonic() - request["started_monotonic"]
            raise error
        terminal = read_json(terminal_path)
        if terminal["token"] != entry["token"]:
            raise RuntimeError("fenced policy supervisor terminal receipt")
        if terminal["attempt_status"] == "timeout":
            error = subprocess.TimeoutExpired(command, allowance)
            error.paid_wall_seconds = terminal["paid_wall_seconds"]
            raise error
        if terminal["attempt_status"] != "completed":
            raise RuntimeError(terminal.get("reason", "policy supervisor failed"))
        result = subprocess.CompletedProcess(command, terminal["child_returncode"])
        result.paid_wall_seconds = terminal["paid_wall_seconds"]
        return result

    def complete(self, key, result):
        with self.queue.state() as state:
            entry = state["policy_attempts"][key]
            if entry["status"] not in {"running", "awaiting_certification"}:
                raise RuntimeError("policy attempt was fenced before terminal publication")
            entry.update(status="completed", charged_seconds=result["paid_wall_seconds"] +
                         result.get("unmeasured_interrupt_reserved_seconds", 0.),
                         completed_at=utc_now(), result=result)


def supervise(path):
    """Frozen independent deadline owner; a killed submitter cannot extend it."""
    request = read_json(path)
    directory = Path(path).parent
    terminal = {"token": request["token"], "attempt_status": "error", "child_returncode": None}
    child = None
    identity = None
    cancelled = False
    def cancel(signum, frame):
        nonlocal cancelled
        cancelled = True
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    def expired(signum, frame):
        raise TimeoutError("absolute policy allowance exhausted")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, request["deadline_monotonic"] - time.monotonic()))
    try:
        atomic_json(directory / "supervisor.json", {"pid": os.getpid(), "process_identity": process_identity(os.getpid()),
                                                    "token": request["token"], "deadline_monotonic": request["deadline_monotonic"]})
        snapshot = Path(request["source"]["snapshot_path"])
        verify_snapshot(snapshot, request["source"])
        expected = {key: value for key, value in request["source"].items() if key != "snapshot_path"}
        if read_json(snapshot / "forge-source.json") != expected:
            raise ValueError("policy source provenance metadata changed")
        if time.monotonic() >= request["deadline_monotonic"]:
            terminal.update(attempt_status="timeout", reason="allowance exhausted before child launch")
        else:
            # The payload cannot execute before its durable process identity is
            # written. A supervisor killed in this window closes the pipe and
            # the bootstrap exits instead of launching untracked training.
            read_fd, write_fd = os.pipe()
            bootstrap = ("import os,sys,json; fd=int(sys.argv[1]); permitted=os.read(fd,1); os.close(fd); "
                         "command=json.loads(sys.argv[2]); "
                         "sys.exit(125) if permitted != b'1' else os.execvpe(command[0],command,os.environ)")
            try:
                child = subprocess.Popen([sys.executable, "-u", "-c", bootstrap, str(read_fd), json.dumps(request["command"])],
                                         cwd=snapshot, start_new_session=True,
                                         pass_fds=tuple(request["lease_fds"]) + (read_fd,))
                identity = {"pid": child.pid, "process_identity": process_identity(child.pid), "token": request["token"],
                            "supervisor_pid": os.getpid(), "deadline_monotonic": request["deadline_monotonic"]}
                atomic_json(directory / "child.json", identity)
                if time.monotonic() < request["deadline_monotonic"]:
                    os.write(write_fd, b"1")
            finally:
                os.close(read_fd); os.close(write_fd)
            while child.poll() is None:
                if cancelled or time.monotonic() >= request["deadline_monotonic"]:
                    fence_orphan_group(identity, directory)
                    child.wait()
                    terminal.update(attempt_status="cancelled" if cancelled else "timeout",
                                    reason="supervised policy child interrupted or wall allowance exhausted")
                    break
                time.sleep(.01)
            else:
                terminal["attempt_status"] = "completed"
            terminal["child_returncode"] = child.returncode
            # A successful leader must not leave descendants holding resources.
            fence_orphan_group(identity, directory)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if child:
            fence_orphan_group(identity or {"pid": child.pid, "process_identity": None}, directory)
            child.wait()
        terminal.update(attempt_status="timeout" if isinstance(error, TimeoutError) else "error",
                        reason=f"{type(error).__name__}: {error}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal["paid_wall_seconds"] = max(0., time.monotonic() - request["started_monotonic"])
        atomic_json(directory / "supervisor-terminal.json", terminal)
    return 0 if terminal["attempt_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(supervise(Path(sys.argv[1])))
