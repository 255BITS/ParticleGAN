"""Independent timeout/cancellation supervisor for one immutable Forge attempt."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .contracts import atomic_json, read_json, utc_now
from .queue import process_identity
from .sources import compute_profile, runtime_manifest, verify_snapshot


def stop_group(process: subprocess.Popen, grace: float = 2.0):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    # A leader can exit while leaving descendants alive. Kill the remaining group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def execute(path: Path) -> int:
    resolved = read_json(path)
    worker, request, job = resolved["worker"], resolved["request"], resolved["job"]
    directory = path.parent
    started = time.monotonic()
    terminal = {"schema_version": 1, "token": worker["token"], "attempt_status": "error", "result": {}}
    cancelled = False
    child = None

    def cancel(signum, frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        verify_snapshot(Path(request["source"]["snapshot_path"]), request["source"])
        if request.get("runtime") != runtime_manifest():
            raise ValueError("runtime changed since submission; resolve a new request")
        expected_compute = job.get("science", {}).get("compute")
        if expected_compute and compute_profile("cpu" if worker["device"] == "cpu" else "cuda", expected_compute.get("model")) != expected_compute:
            raise ValueError("compute backend/hardware changed since submission; resolve a compatible request")
        lease_fd = int(os.environ["FORGE_LEASE_FD"])
        # The runner inherits the lease too: SIGKILL of this supervisor cannot
        # convince recovery that a still-running training descendant has died.
        child = subprocess.Popen([sys.executable, "-u", "-m", "experiments.forge.runtime", str(path)],
                                 start_new_session=True, pass_fds=(lease_fd,))
        atomic_json(directory / "child.json", {"pid": child.pid, "process_identity": process_identity(child.pid),
                    "supervisor_pid": os.getpid(), "token": worker["token"]})
        deadline = started + job["budget_seconds"]
        while child.poll() is None:
            atomic_json(directory / "heartbeat.json", {"timestamp": utc_now(), "pid": os.getpid(),
                "child_pid": child.pid, "token": worker["token"], "elapsed_seconds": time.monotonic() - started})
            if cancelled or (directory / "cancel.json").exists():
                stop_group(child)
                terminal.update(attempt_status="cancelled", reason="request cancelled")
                break
            if time.monotonic() >= deadline:
                stop_group(child)
                terminal.update(attempt_status="timeout", reason="task wall-time budget exhausted")
                break
            time.sleep(.1)
        else:
            result_path = directory / "raw-result.json"
            if result_path.exists():
                terminal["result"] = read_json(result_path)
            if child.returncode == 0 and result_path.exists():
                terminal.update(attempt_status="completed")
            else:
                terminal.update(attempt_status="error", reason=f"runner exited {child.returncode}; inspect run.log",
                                exit_code=child.returncode)
        # Clean descendants on every path, including successful leader exit.
        stop_group(child, grace=0)
        if terminal["attempt_status"] == "completed" and request.get("requires_independent_grading"):
            child = subprocess.Popen([sys.executable, "-u", "-m", "experiments.forge.evaluate", str(path)],
                                     start_new_session=True, pass_fds=(lease_fd,))
            atomic_json(directory / "child.json", {"pid": child.pid, "process_identity": process_identity(child.pid),
                        "supervisor_pid": os.getpid(), "token": worker["token"], "phase": "evaluation"})
            while child.poll() is None:
                if cancelled or (directory / "cancel.json").exists() or time.monotonic() >= deadline:
                    stop_group(child)
                    terminal.update(attempt_status="cancelled" if cancelled or (directory / "cancel.json").exists() else "timeout",
                                    reason="independent evaluation interrupted or over budget")
                    break
                time.sleep(.1)
            else:
                graded_path = directory / "graded-result.json"
                if child.returncode != 0 or not graded_path.exists():
                    terminal.update(attempt_status="error", reason="independent evaluator failed; raw evidence retained")
                else:
                    terminal["grading"] = read_json(graded_path)
            stop_group(child, grace=0)
    except BaseException as error:
        if child:
            stop_group(child)
        traceback.print_exc()
        terminal.update(attempt_status="error", reason=f"{type(error).__name__}: {error}")
    finally:
        terminal.update(elapsed_seconds=time.monotonic() - started, finished_at=utc_now())
        atomic_json(directory / "terminal.json", terminal)
        print(f"{terminal['finished_at']} {job['task_id']} {terminal['attempt_status']} "
              f"{terminal['elapsed_seconds']:.3f}s {terminal.get('reason', '')}", flush=True)
    return 0 if terminal["attempt_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(execute(Path(sys.argv[1]).resolve()))
