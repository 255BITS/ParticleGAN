"""Ownership controls with fake children; no GAN research experiments."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from benchmarks.toy_audit import api_family_search as search, api_run
from experiments.forge import policy_execution as execution, queue as queue_module
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.queue import Queue
from experiments.forge.sources import inspect_source, snapshot_source


def packet_for_admission():
    return {"source": {"commit": "software-control", "files_sha256": {}},
            "spec": {"export_grace_seconds": 1.}, "lane_runtime": {"torch_threads": 1}}


def core_request(tmp_path, backend):
    from experiments.forge.api import FormulationContext
    root = tmp_path / "worktree"
    (root / "particlegan").mkdir(parents=True)
    (root / "particlegan/fixture.py").write_text("value = 1\n")
    manifest = inspect_source(root)
    manifest["snapshot_path"] = str(snapshot_source(root, tmp_path / "queue", manifest))
    rng = FormulationContext().streams.manifest()
    protocol = {"id": "screening", "seed": 0}
    return {"candidate": {"id": "control"}, "candidate_revision": "control-source", "source": manifest,
            "protocol": protocol, "rng": rng, "through_tier": 1,
            "view": {"goal": "control", "assignments": [{"task": "one", "qualification_tier": 1,
                       "importance": "required", "order": 0}]},
            "tasks": {"one": {"id": "one", "dependencies": []}},
            "jobs": [{"task_id": "one", "compatibility_key": stable_hash(["control", backend]),
                      "science": {"protocol": protocol, "seed": 0, "rng": rng}, "budget_seconds": 11.,
                      "resources": {"allow_cpu": backend == "cpu", "gpus": int(backend == "cuda"),
                                    "cpu_threads": 1, "host_memory_mb": 512, "memory_mb": 0}}]}


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("first", ["policy", "core"])
def test_core_queue_and_policy_share_host_and_exclusive_gpu_admission(tmp_path, monkeypatch, backend, first):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    capacity = {"cpu_threads": 1 if backend == "cpu" else 2, "memory_mb": 4096,
                "available_memory_mb": 4096}
    monkeypatch.setattr(execution, "host_capacity", lambda: capacity)
    monkeypatch.setattr(queue_module, "host_capacity", lambda: capacity)
    coordinator = execution.PolicyCoordinator(tmp_path / "queue")
    request = core_request(tmp_path, backend)
    coordinator.queue.submit(request, {"id": "control", "budget_seconds": 100., "candidate_budget_seconds": 100.})
    slot = {"device": "cpu" if backend == "cpu" else "0", "slot": 0,
            "memory_mb": 4096, "capacity_mb": 4096}
    device = "cpu" if backend == "cpu" else "cuda:0"
    packet, row = packet_for_admission(), {"timeout_seconds": 10.}
    if first == "core":
        # The real drain owns this lease throughout its claim-to-launch gap.
        with execution.execution_lease(coordinator.root / "coordinator.lock"):
            assert coordinator.queue.claim([slot]) is not None
            with coordinator.admit("policy-control", packet, row, device) as (decision, lease):
                assert decision["status"] == "busy" and lease is None
        assert coordinator.queue.inspect().get("policy_attempts", {}) == {}
    else:
        with coordinator.admit("policy-control", packet, row, device) as (decision, lease):
            assert decision["allowance_seconds"] == 11. and lease is not None
            assert coordinator.queue.claim([slot]) is None
            state = coordinator.queue.inspect()
            assert next(iter(state["submissions"].values()))["status"] == "queued"
            assert state["campaigns"]["control"]["reserved_seconds"] == 0.
            coordinator.complete("policy-control", {"paid_wall_seconds": 1.})
        assert coordinator.queue.claim([slot]) is not None


def test_visible_cuda_device_maps_to_shared_physical_index(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,7")
    assert execution.physical_device("cuda:1") == "7"
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        execution.physical_device("cuda:2")


def test_cuda_placement_reuses_science_but_gpu_model_is_a_distinct_cohort(tmp_path):
    coordinator = execution.PolicyCoordinator(tmp_path / "queue")
    packet = {**packet_for_admission(), "execution_source": {"digest": "source"},
              "case_definitions": {"one": {"id": "one"}}}
    packet["lane_runtime"].update(device="cuda:0", cuda_device_model="one-model")
    trial, row = {"family": "atlas", "recipe_overrides": {}}, {"id": "one", "timeout_seconds": 10.}
    key = coordinator.attempt_key(packet, trial, row)
    other = deepcopy(packet)
    other["lane_runtime"]["device"] = "cuda:1"
    assert coordinator.attempt_key(other, trial, row) == key
    other["lane_runtime"]["cuda_device_model"] = "another-model"
    assert coordinator.attempt_key(other, trial, row) != key


def test_policy_runtime_binds_actual_single_thread_child_despite_parent_threads(monkeypatch):
    import torch
    monkeypatch.setattr(torch, "get_num_threads", lambda: 24)
    assert search._runtime("cpu")["torch_threads"] == 1


def test_surviving_child_fences_recovery_then_interruption_charges_complete_allowance(tmp_path):
    coordinator = execution.PolicyCoordinator(tmp_path / "queue")
    packet = packet_for_admission()
    child = None
    try:
        with coordinator.admit("inherited", packet, {"timeout_seconds": 10.}, "cpu") as (decision, lease):
            assert decision["allowance_seconds"] == 11.
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                     pass_fds=(lease.fileno(),))
        # The supervisor's descriptor is closed, as it would be after a crash.
        coordinator.recover()
        assert coordinator.queue.inspect()["policy_attempts"]["inherited"]["status"] == "running"
        with coordinator.admit("inherited", packet, {"timeout_seconds": 10.}, "cpu") as (decision, lease):
            assert decision["status"] == "busy" and lease is None
        child.terminate(); child.wait(timeout=5)
        coordinator.recover(); coordinator.recover()
        saved = coordinator.queue.inspect()["policy_attempts"]["inherited"]
        assert saved["status"] == "interrupted" and saved["charged_seconds"] == 11.
        with coordinator.admit("inherited", packet, {"timeout_seconds": 10.}, "cpu") as (decision, lease):
            assert decision["status"] == "interrupted" and lease is None
        assert len(coordinator.queue.inspect()["policy_attempts"]) == 1
    finally:
        if child and child.poll() is None:
            child.kill(); child.wait(timeout=5)


def wait_for_file(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"child did not publish {path}")
        time.sleep(.01)


def frozen_control(tmp_path):
    root = tmp_path / "source"
    (root / "particlegan").mkdir(parents=True)
    (root / "particlegan/__init__.py").write_text("")
    code = root / "particlegan/control.py"
    code.write_text("VALUE = 'frozen'\n")
    expected = {"commit": "control", "files_sha256": {"particlegan/control.py": api_run.file_hash(code)}}
    manifest = execution.freeze_source(root, tmp_path / "queue", expected)
    return root, code, {**packet_for_admission(), "execution_source": manifest}


def test_actual_child_imports_frozen_source_after_concurrent_worktree_edit(tmp_path):
    root, code, packet = frozen_control(tmp_path)
    coordinator = execution.PolicyCoordinator(tmp_path / "queue")
    ready, release, observed = [tmp_path / name for name in ("ready", "release", "observed")]
    script = f"""from pathlib import Path
import time
Path({str(ready)!r}).touch()
deadline = time.monotonic() + 10
while not Path({str(release)!r}).exists():
    if time.monotonic() > deadline: raise RuntimeError('control timeout')
    time.sleep(.01)
from particlegan.control import VALUE
Path({str(observed)!r}).write_text(VALUE)
"""
    with coordinator.admit("immutable", packet, {"timeout_seconds": 10.}, "cpu") as (_, lease):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(coordinator.launch, [sys.executable, "-c", script], packet,
                                 tmp_path / "child.log", (lease,), 11.)
            try:
                wait_for_file(ready)
                code.write_text("VALUE = 'edited'\n")
            finally:
                release.touch()
            assert future.result(timeout=15).returncode == 0
        assert observed.read_text() == "frozen"
        assert str(root) not in (tmp_path / "child.log").read_text()
        coordinator.complete("immutable", {"paid_wall_seconds": .1})


@pytest.mark.parametrize("mutation", ["code", "origin"])
def test_tampered_snapshot_blocks_before_child_start(tmp_path, monkeypatch, mutation):
    _, _, packet = frozen_control(tmp_path)
    snapshot = Path(packet["execution_source"]["snapshot_path"])
    if mutation == "code":
        (snapshot / "particlegan/control.py").write_text("VALUE = 'tampered'\n")
    else:
        metadata = read_json(snapshot / "forge-source.json")
        metadata["origin_commit"] = "tampered"
        atomic_json(snapshot / "forge-source.json", metadata)
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("tampered source launched"))
    with pytest.raises(ValueError, match="changed"):
        execution.PolicyCoordinator(tmp_path / "queue").launch([], packet, tmp_path / "log", (), 11.)


def test_identical_source_bytes_keep_distinct_planned_origin_metadata(tmp_path):
    root, code, first = frozen_control(tmp_path)
    expected = {"commit": "later-docs-origin", "files_sha256": {"particlegan/control.py": api_run.file_hash(code)}}
    second = execution.freeze_source(root, tmp_path / "queue", expected)
    assert second["digest"] == first["execution_source"]["digest"]
    assert second["snapshot_path"] != first["execution_source"]["snapshot_path"]
    assert read_json(Path(second["snapshot_path"]) / "forge-source.json")["origin_commit"] == "later-docs-origin"


@pytest.mark.parametrize("live", [False, True])
def test_policy_admission_recovers_abandoned_core_reservation_and_preserves_live_lease(tmp_path, monkeypatch, live):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    coordinator = execution.PolicyCoordinator(tmp_path / "queue")
    request = core_request(tmp_path, "cuda")
    coordinator.queue.submit(request, {"id": "control", "budget_seconds": 100., "candidate_budget_seconds": 100.})
    claim = coordinator.queue.claim([{"device": "0", "slot": 0, "memory_mb": 4096, "capacity_mb": 4096}])
    lease_path = Path(claim["worker"]["directory"]) / "execution.lock"
    def check():
        with coordinator.admit("policy-after-orphan", packet_for_admission(), {"timeout_seconds": 10.}, "cuda:0") as (decision, lease):
            state = coordinator.queue.inspect()
            core = state["jobs"][claim["job"]["compatibility_key"]]
            if live:
                assert decision["status"] == "busy" and lease is None
                assert core["status"] == "running"
                assert state["campaigns"]["control"]["reserved_seconds"] == 11.
            else:
                assert decision["status"] == "running" and lease is not None
                assert core["status"] == "terminal"
                assert core["result"]["task_results"][0]["gate_status"] == "INCOMPLETE"
                assert state["campaigns"]["control"]["reserved_seconds"] == 0.
                assert len(state["charges"]) == 1
                coordinator.complete("policy-after-orphan", {"paid_wall_seconds": 1.})
    if live:
        with execution.execution_lease(lease_path):
            check()
    else:
        check()


def test_cross_output_concurrent_study_submitters_launch_one_compatible_attempt(tmp_path, monkeypatch):
    # Retain all eight configurations/case denominators; scope this software
    # control to one admitted fake child and block the other configurations.
    from test_toy_api_family_search import planned
    from benchmarks.toy_audit import api_contract
    monkeypatch.setenv("PARTICLEGAN_FORGE_QUEUE", str(tmp_path / "queue"))
    import torch
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    cases = api_contract.discover()
    packet = planned(tmp_path, cases, monkeypatch)
    target = next(trial for trial in packet["trials"] if trial["family"] == "atlas")
    for trial in packet["trials"]:
        if trial is not target:
            trial["status"] = "BLOCKED"
    monkeypatch.setattr(search, "plan_study", lambda spec: deepcopy(packet))
    context = multiprocessing.get_context("fork")
    started, release = context.Event(), context.Event()
    count = tmp_path / "physical-launches"
    def child(self, command, *args):
        with count.open("a") as stream:
            stream.write("physical attempt\n")
        started.set()
        assert release.wait(10)
        output = Path(command[command.index("--output") + 1]) / command[command.index("--case") + 1]
        api_run.write_json(output / "receipt.json", {"status": "INCOMPLETE", "verdict": "FAIL", "failed_bounds": ["software cap"]})
        return type("Child", (), {"returncode": 0})()
    monkeypatch.setattr(execution.PolicyCoordinator, "launch", child)
    first = context.Process(target=search.run_study, args=(packet["spec"], tmp_path / "first"),
                            kwargs={"family": "atlas", "device": "cpu"})
    second = context.Process(target=search.run_study, args=(packet["spec"], tmp_path / "second"),
                             kwargs={"family": "atlas", "device": "cpu"})
    try:
        first.start()
        assert started.wait(10)
        second.start(); second.join(10)
        assert second.exitcode == 0
        active = read_json(tmp_path / "second/study.json")
        assert active["coordinator"]["attached"] is True
        assert active["coordinator"]["canonical_output"] == str(tmp_path / "first")
        assert count.read_text().count("physical attempt") == 1
        release.set(); first.join(10)
        assert first.exitcode == 0
        final = search.run_study(packet["spec"], tmp_path / "second", family="atlas", device="cpu")
        assert count.read_text().count("physical attempt") == 1
        assert final["spent_seconds"] > 0
        assert final["default_adoption"] is False and "cloud/served-policy" in final["scope"]
        assert len(Queue(tmp_path / "queue").inspect()["policy_attempts"]) == 1
    finally:
        release.set()
        for process in (first, second):
            if process.pid and process.is_alive():
                process.kill(); process.join(5)


def test_attempt_identity_separates_law_recipe_source_and_runtime(tmp_path):
    coordinator = execution.PolicyCoordinator(tmp_path / "queue")
    packet = {**packet_for_admission(), "execution_source": {"digest": "source-one"},
              "case_definitions": {"one": {"id": "one", "sampling": "served-cloud"}}}
    trial, row = {"family": "atlas", "recipe_overrides": {"lr": .006375}}, {"id": "one", "timeout_seconds": 10.}
    key = coordinator.attempt_key(packet, trial, row)
    for field in ("sampling", "source", "runtime"):
        changed = deepcopy(packet)
        if field == "sampling": changed["case_definitions"]["one"]["sampling"] = "clean-mog"
        if field == "source": changed["execution_source"]["digest"] = "source-two"
        if field == "runtime": changed["lane_runtime"]["torch_threads"] = 2
        assert coordinator.attempt_key(changed, trial, row) != key
    assert coordinator.attempt_key(packet, {**trial, "family": "e22"}, row) != key
    assert coordinator.attempt_key(packet, {**trial, "recipe_overrides": {"lr": .0085}}, row) != key


def test_overlapping_studies_reuse_original_cost_and_never_retry_interruption(tmp_path):
    coordinator = execution.PolicyCoordinator(tmp_path / "queue")
    packet = {**packet_for_admission(), "execution_source": {"digest": "same-source"},
              "case_definitions": {"one": {"id": "one", "sampling": "served-cloud"}}}
    packet["spec"]["id"] = "first-grid"
    trial, row = {"family": "atlas", "recipe_overrides": {"lr": .006375}}, {"id": "one", "timeout_seconds": 10.}
    key = coordinator.attempt_key(packet, trial, row)
    other = deepcopy(packet)
    other["spec"].update(id="another-grid", grid={"lr": [.006375, .0053125]})
    assert coordinator.attempt_key(other, trial, row) == key
    with coordinator.admit(key, packet, row, "cpu") as (admission, lease):
        assert admission["allowance_seconds"] == 11. and lease is not None
        coordinator.complete(key, {"status": "INCOMPLETE", "paid_wall_seconds": 2.})
    with coordinator.admit(key, other, row, "cpu") as (admission, lease):
        assert lease is None and admission["charged_seconds"] == 2.
        assert admission["result"]["status"] == "INCOMPLETE"
    assert len(coordinator.queue.inspect()["policy_attempts"]) == 1


def test_recovery_uses_central_terminal_before_study_projection_save(tmp_path, monkeypatch):
    from test_toy_api_family_search import planned
    from benchmarks.toy_audit import api_contract
    monkeypatch.setenv("PARTICLEGAN_FORGE_QUEUE", str(tmp_path / "queue"))
    import torch
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    packet = planned(tmp_path, api_contract.discover(), monkeypatch)
    target = next(trial for trial in packet["trials"] if trial["family"] == "atlas")
    for trial in packet["trials"]:
        if trial is not target:
            trial["status"] = "BLOCKED"
    monkeypatch.setattr(search, "plan_study", lambda spec: deepcopy(packet))
    from experiments.forge.policy_execution import freeze_source
    coordinator = execution.PolicyCoordinator(tmp_path / "queue")
    packet["execution_source"] = freeze_source(api_contract.ROOT, coordinator.root, packet["source"])
    key, output = coordinator.register(packet, tmp_path / "study", "atlas", search._runtime("cpu"))
    frozen = read_json(output / "study.json")
    row = next(trial for trial in frozen["trials"] if trial["family"] == "atlas")["cases"][0]
    attempt = coordinator.attempt_key(frozen, target, row)
    row.update(status="RUNNING", attempt_key=attempt)
    search._save(output / "study.json", frozen)
    with coordinator.admit(attempt, frozen, row, "cpu") as (_, lease):
        assert lease is not None
        coordinator.complete(attempt, {**row, "status": "INCOMPLETE", "reason": "software terminal", "paid_wall_seconds": 2.})
    monkeypatch.setattr(execution.PolicyCoordinator, "launch", lambda *args: pytest.fail("terminal attempt reran"))
    recovered = search.run_study(packet["spec"], output, family="atlas", device="cpu")
    target = next(trial for trial in recovered["trials"] if trial["family"] == "atlas")
    assert target["status"] == "INCOMPLETE"
    assert recovered["spent_seconds"] == 2.
    assert recovered["unmeasured_interrupt_reservation_seconds"] == 0.
    assert target["cases"][0]["reused_physical_attempt"] is True
