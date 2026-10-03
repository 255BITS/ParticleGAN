"""Dedicated policy execution ownership, sharing Forge's admission transaction.

Policy receipts retain their cloud/served cohort. They are not Queue jobs and
cannot supply MoG qualification. The shared ledger owns physical attempts; study
archives own their existing scientific gates and conservative budget readouts.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import subprocess

from .contracts import atomic_json, read_json, stable_hash, utc_now
from .queue import Queue, host_capacity, lease_held
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
        return members[index]
    return str(index)


def scientific_runtime(runtime):
    """Physical CUDA placement is admission, not a new scientific cohort."""
    return {**runtime, "device": str(runtime.get("device", "cpu")).split(":", 1)[0]}


def recover_attempts(state):
    """Called under queue.lock. A held kernel lease always wins over age/PID."""
    for entry in state.get("policy_attempts", {}).values():
        if entry["status"] == "running" and not lease_held(Path(entry["lease_path"])):
            entry.update(status="interrupted", charged_seconds=entry["allowance_seconds"],
                         completed_at=utc_now(), result=None,
                         reason="released execution lease without a committed terminal receipt")


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
        return stable_hash({"cohort": "cloud-served-policy-v1", "case": packet["case_definitions"][row["id"]],
                            "family": trial["family"], "recipe": trial["recipe_overrides"],
                            "source_digest": packet["execution_source"]["digest"],
                            "runtime": scientific_runtime(packet["lane_runtime"]), "timeout_seconds": row["timeout_seconds"],
                            "export_grace_seconds": packet["spec"]["export_grace_seconds"],
                            "frames": packet["spec"].get("frames", 9)})

    @contextmanager
    def admit(self, key, packet, row, device):
        """Reserve complete allowance and exclusive GPU admission atomically."""
        self.queue.collect(only_orphans=True)
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
                    existing.update(status="interrupted", charged_seconds=existing["allowance_seconds"],
                                    result=None, completed_at=utc_now(),
                                    reason="released execution lease without a committed terminal receipt")
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
                    occupied = target != "cpu" and any(str(worker["device"]) == target for worker in workers)
                    reservations = [worker.get("host_reservation", {"cpu_threads": 1, "host_memory_mb": 512})
                                    for worker in workers]
                    if (occupied or needed["cpu_threads"] + sum(r["cpu_threads"] for r in reservations) > capacity["cpu_threads"]
                            or needed["host_memory_mb"] + sum(r["host_memory_mb"] for r in reservations) > capacity["available_memory_mb"]):
                        decision = {"status": "busy", "reason": "shared Forge host/GPU admission unavailable"}
                    else:
                        decision = {"status": "running", "lease_path": str(path),
                                    "allowance_seconds": row["timeout_seconds"] + packet["spec"]["export_grace_seconds"],
                                    "source": packet["source"],
                                    "runtime": packet["lane_runtime"],
                                    "started_at": utc_now(), "charged_seconds": 0., "result": None,
                                    "worker": {"device": target, "slot": 0, "exclusive_device": target != "cpu",
                                               "host_reservation": needed}}
                        state["policy_attempts"][key] = decision
            yield decision, lease if decision["status"] == "running" and existing is None else None

    def launch(self, command, packet, log, leases, allowance):
        snapshot = Path(packet["execution_source"]["snapshot_path"])
        verify_snapshot(snapshot, packet["execution_source"])
        expected = {key: value for key, value in packet["execution_source"].items() if key != "snapshot_path"}
        if read_json(snapshot / "forge-source.json") != expected:
            raise ValueError("policy source provenance metadata changed")
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(snapshot), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        with Path(log).open("wb") as stream:
            child = subprocess.Popen(command, cwd=snapshot, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=tuple(lease.fileno() for lease in leases))
            try:
                child.wait(timeout=allowance)
            except BaseException:
                # Terminate descendants too; close-only leases retain admission
                # if the coordinator itself is killed before this cleanup.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                raise
        return child

    def complete(self, key, result):
        with self.queue.state() as state:
            entry = state["policy_attempts"][key]
            if entry["status"] != "running":
                raise RuntimeError("policy attempt was fenced before terminal publication")
            entry.update(status="completed", charged_seconds=result["paid_wall_seconds"] +
                         result.get("unmeasured_interrupt_reserved_seconds", 0.),
                         completed_at=utc_now(), result=result)
