"""Queue ownership and real process lease lifetime, without training/model calls."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.queue import Queue, drain
from experiments.forge.worker import inherited_lease_fds
from tests.test_forge_queue import SLOTS, campaign, grade, request

ROOT = Path(__file__).resolve().parents[1]


def test_claim_filter_leaves_other_owners_work_pending(tmp_path):
    queue = Queue(tmp_path / "queue", grader=grade)
    foreign = request(tmp_path, candidate="other-owner", cap=1)
    queue.submit(foreign, campaign())
    ours = deepcopy(foreign)
    ours["candidate"]["id"] = "our-study"
    for job in ours["jobs"]:
        job["compatibility_key"] += "-ours"
    admitted = queue.submit(ours, campaign())
    assert queue.claim(SLOTS, request_filter="unknown-request") is None
    claim = queue.claim(SLOTS, request_filter=admitted["request"]["request_id"])
    assert claim["request"]["candidate"]["id"] == "our-study"
    assert queue.inspect()["jobs"][foreign["jobs"][0]["compatibility_key"]]["status"] == "pending"


def test_normal_drain_limits_request_and_forwards_explicit_fleet(tmp_path):
    from tests.test_forge_queue import finish

    queue = Queue(tmp_path / "queue", grader=grade)
    foreign = request(tmp_path, candidate="other-owner", cap=1)
    queue.submit(foreign, campaign())
    ours = deepcopy(foreign)
    ours["candidate"]["id"] = "our-study"
    for job in ours["jobs"]:
        job["compatibility_key"] += "-ours"
    admitted = queue.submit(ours, campaign())
    launched = []

    class Finished:
        def poll(self):
            return 0

    with (tmp_path / "fleet.lock").open("w+b") as lease:
        def launch(claim, *, shared_fleet_fds):
            assert shared_fleet_fds == (lease.fileno(),)
            assert claim["request"]["request_id"] == admitted["request"]["request_id"]
            launched.append(claim)
            finish(claim)
            return Finished()

        with patch.object(queue, "launch", side_effect=launch):
            drain(queue, ["cpu"], request_filter=admitted["request"]["request_id"],
                  shared_fleet_fds=(lease.fileno(),), poll_seconds=0)
    assert len(launched) == 1
    state = queue.inspect()
    assert state["submissions"][admitted["request"]["request_id"]]["status"] == "completed"
    assert state["jobs"][foreign["jobs"][0]["compatibility_key"]]["status"] == "pending"


def test_normal_drain_checks_fleet_before_claiming(tmp_path):
    queue = Queue(tmp_path / "queue")
    with patch.object(queue, "claim") as claim:
        with pytest.raises(OSError):
            drain(queue, ["cpu"], shared_fleet_fds=(999999,))
        claim.assert_not_called()


def test_cli_exposes_exact_request_and_explicit_fleet():
    from experiments.forge.__main__ import parser
    args = parser().parse_args(["drain", "--gpus", "cpu", "--request", "our-request",
                                "--shared-fleet-fd", "3", "--shared-fleet-fd", "4"])
    assert args.request == "our-request"
    assert args.shared_fleet_fd == [3, 4]


@contextmanager
def launch_fixture(tmp_path, *, source=None):
    queue = Queue(tmp_path / "queue")
    directory = tmp_path / "attempt"
    directory.mkdir()
    worker = dict(directory=str(directory), token="fence", device="cpu")
    resolved = dict(worker=worker, job=dict(compatibility_key="key"),
                    request=dict(source=dict(snapshot_path=str(source or ROOT),
                        files={"experiments/forge/worker.py": file_hash(ROOT / "experiments/forge/worker.py")})))
    state = dict(jobs=dict(key=dict(status="running", worker=dict(worker))))

    @contextmanager
    def state_context():
        yield state

    with patch.object(queue, "state", state_context):
        yield queue, resolved


def test_default_launch_clears_stale_outer_shared_descriptors(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_SHARED_SLOT_FD", "999999")
    monkeypatch.setenv("FORGE_SHARED_FLEET_FDS", "999998,999997")
    with launch_fixture(tmp_path) as (queue, resolved), patch("experiments.forge.queue.subprocess.Popen") as popen:
        popen.return_value.pid = os.getpid()
        queue.launch(resolved)
        options = popen.call_args.kwargs
        assert len(options["pass_fds"]) == 1
        assert "FORGE_SHARED_SLOT_FD" not in options["env"]
        assert "FORGE_SHARED_FLEET_FDS" not in options["env"]


def test_external_fleet_refuses_an_old_immutable_worker(tmp_path):
    with launch_fixture(tmp_path) as (queue, resolved), (tmp_path / "lease").open("w+b") as lease, \
         patch("experiments.forge.queue.subprocess.Popen") as popen:
        resolved["request"]["source"]["files"]["experiments/forge/worker.py"] = "old-worker"
        with pytest.raises(RuntimeError, match="FD-forwarding worker"):
            queue.launch(resolved, shared_fleet_fds=(lease.fileno(),))
        popen.assert_not_called()


@pytest.mark.parametrize("argument", ["shared_slot_fd", "shared_fleet_fds"])
@pytest.mark.parametrize("descriptor", [True, -1, "3", 999999])
def test_invalid_descriptor_fails_before_process_creation(tmp_path, argument, descriptor):
    value = descriptor if argument == "shared_slot_fd" else (descriptor,)
    with launch_fixture(tmp_path) as (queue, resolved), patch("experiments.forge.queue.subprocess.Popen") as popen:
        with pytest.raises((ValueError, OSError)):
            queue.launch(resolved, **{argument: value})
        popen.assert_not_called()


def test_worker_deduplicates_and_checks_inherited_descriptors(tmp_path):
    with (tmp_path / "one").open("w+b") as one, (tmp_path / "two").open("w+b") as two:
        env = dict(FORGE_LEASE_FD=str(one.fileno()), FORGE_SHARED_SLOT_FD=str(two.fileno()),
                   FORGE_SHARED_FLEET_FDS=f"{one.fileno()},{two.fileno()}")
        with patch.dict(os.environ, env, clear=True):
            assert inherited_lease_fds() == (one.fileno(), two.fileno())
        stale = two.fileno()
    with patch.dict(os.environ, {"FORGE_LEASE_FD": str(stale)}, clear=True):
        with pytest.raises(OSError):
            inherited_lease_fds()


def test_worker_passes_fleet_to_both_runtime_and_independent_grader(tmp_path):
    from experiments.forge import worker

    path = tmp_path / "request.json"
    atomic_json(path, dict(worker=dict(token="fence", device="cpu"),
                          request=dict(source=dict(snapshot_path=str(ROOT)), runtime={},
                                       requires_independent_grading=True),
                          job=dict(task_id="software-fixture", budget_seconds=30)))
    phases = []

    class Finished:
        pid = os.getpid()
        returncode = 0

        def poll(self):
            return 0

    with (tmp_path / "attempt.lock").open("w+b") as attempt, (tmp_path / "fleet.lock").open("w+b") as fleet:
        def launch(argv, *, pass_fds, **kwargs):
            assert pass_fds == (attempt.fileno(), fleet.fileno())
            phase = argv[3]
            phases.append(phase)
            atomic_json(tmp_path / ("raw-result.json" if phase.endswith("runtime") else "graded-result.json"), {})
            return Finished()

        env = dict(FORGE_LEASE_FD=str(attempt.fileno()), FORGE_SHARED_FLEET_FDS=str(fleet.fileno()))
        with patch.dict(os.environ, env, clear=True), patch.object(worker, "verify_snapshot"), \
             patch.object(worker, "runtime_manifest", return_value={}), patch.object(worker, "stop_group"), \
             patch.object(worker.subprocess, "Popen", side_effect=launch):
            assert worker.execute(path) == 0
    assert phases == ["experiments.forge.runtime", "experiments.forge.evaluate"]


def test_real_descendant_keeps_whole_fleet_after_supervisor_crash(tmp_path):
    """Use actual queue launch/worker FD collection and OS locks across two hops."""
    source = tmp_path / "source"
    source.mkdir()
    ready = tmp_path / "descendant.json"
    probe = (
        "import json, os, pathlib, subprocess, sys, time\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from experiments.forge.worker import inherited_lease_fds\n"
        "fds = inherited_lease_fds()\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "pass_fds=fds, start_new_session=True)\n"
        f"pathlib.Path({str(ready)!r}).write_text(json.dumps(dict(pid=child.pid, fds=list(fds))))\n"
        "time.sleep(30)\n"
    )
    (source / "lease_probe.py").write_text(probe)
    paths = [tmp_path / f"gpu{gpu}-slot{slot}.lock" for gpu, count in ((0, 10), (1, 2)) for slot in range(count)]
    for path in paths:
        path.write_bytes(b"preserved lease inode")
    before = [(path.stat().st_ino, path.stat().st_size, path.stat().st_mode) for path in paths]
    process = None
    child_pid = None
    try:
        with ExitStack() as leases, launch_fixture(tmp_path, source=source) as (queue, resolved):
            handles = [leases.enter_context(path.open("r+b")) for path in paths]
            for handle in handles:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            process = queue.launch(resolved, module="lease_probe", shared_slot_fd=handles[0].fileno(),
                                   shared_fleet_fds=tuple(handle.fileno() for handle in handles))
            deadline = time.monotonic() + 15
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert ready.exists(), (tmp_path / "attempt/run.log").read_text()
            receipt = json.loads(ready.read_text())
            child_pid = receipt["pid"]
            assert len(receipt["fds"]) == 13  # Twelve fleet tokens plus the attempt lease.
        # Closing our descriptors and killing the supervisor must not release a live child's fleet.
        process.kill()
        process.wait(timeout=5)
        for path in paths:
            with path.open("r+b") as checker:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(checker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert before == [(path.stat().st_ino, path.stat().st_size, path.stat().st_mode) for path in paths]
        os.kill(child_pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with paths[0].open("r+b") as checker:
                    fcntl.flock(checker, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(.02)
        for path in paths:
            with path.open("r+b") as checker:
                fcntl.flock(checker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert path.read_bytes() == b"preserved lease inode"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
