"""Capacity discovery and atomic reservations; no CUDA or training execution."""
from copy import deepcopy
import multiprocessing

import pytest

from experiments.forge import queue as module
from experiments.forge.contracts import stable_hash
from experiments.forge.queue import Queue, device_slots, host_capacity
from test_forge_queue import campaign, finish, grade, request


MIB = 1024 * 1024


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value))


def linux(tmp_path, membership="0::/parent/worker\n", mounts=None):
    proc, group = tmp_path / "proc", tmp_path / "cgroup"
    put(proc / "meminfo", "MemTotal: 8388608 kB\nMemAvailable: 6291456 kB\n")
    put(proc / "self/cgroup", membership)
    if mounts:
        put(proc / "self/mountinfo", mounts)
    return proc, group


def test_host_capacity_honors_ancestor_memory_quota_cpuset_and_affinity(tmp_path):
    proc, group = linux(tmp_path)
    put(group / "parent/worker/memory.max", 2048 * MIB)
    put(group / "parent/worker/memory.current", 256 * MIB)
    put(group / "parent/memory.max", 4096 * MIB)
    put(group / "parent/memory.current", 3584 * MIB)
    put(group / "parent/worker/cpu.max", "max 100000")
    put(group / "parent/cpu.max", "250000 100000")
    put(group / "parent/cpuset.cpus.effective", "2-5")
    result = host_capacity(proc_root=proc, cgroup_root=group, affinity={1, 2, 3, 4})
    assert result["cpu_threads"] == 2 and result["cpu_affinity"] == [2, 3, 4]
    assert result["memory_mb"] == 2048 - 102
    assert result["available_memory_mb"] == 512 - 102


def test_cgroup_mount_root_and_namespace_paths_are_resolved(tmp_path):
    mounts = "20 1 0:1 /delegated /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
    proc, group = linux(tmp_path, "0::/delegated/worker\n", mounts)
    put(group / "worker/memory.max", 1024 * MIB)
    put(group / "worker/memory.current", 128 * MIB)
    result = host_capacity(proc_root=proc, cgroup_root=group, affinity={0, 1})
    assert result["memory_mb"] == 1024 - 51
    put(proc / "self/cgroup", "0::/worker\n")
    assert host_capacity(proc_root=proc, cgroup_root=group, affinity={0, 1}) == result


def test_cgroup_v1_limits_and_unlimited_sentinels(tmp_path):
    mounts = ("20 1 0:1 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"
              "21 1 0:2 / /sys/fs/cgroup/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct\n")
    proc, group = linux(tmp_path, "7:memory:/job\n6:cpu,cpuacct:/job\n", mounts)
    put(group / "memory/job/memory.limit_in_bytes", 3072 * MIB)
    put(group / "memory/job/memory.usage_in_bytes", 1024 * MIB)
    put(group / "memory/memory.limit_in_bytes", 2**63 - 4096)
    put(group / "memory/memory.usage_in_bytes", 0)
    put(group / "cpu,cpuacct/job/cpu.cfs_quota_us", 100000)
    put(group / "cpu,cpuacct/job/cpu.cfs_period_us", 100000)
    result = host_capacity(proc_root=proc, cgroup_root=group, affinity=set(range(8)))
    assert result["cpu_threads"] == 1
    assert result["memory_mb"] == 3072 - 153
    assert result["available_memory_mb"] == 2048 - 153


def test_unknown_memory_fails_closed_and_limited_missing_usage_is_not_free(tmp_path):
    with pytest.raises(ValueError, match="physical host memory"):
        host_capacity(proc_root=tmp_path)
    proc, group = linux(tmp_path)
    put(group / "parent/worker/memory.max", 1024 * MIB)
    result = host_capacity(proc_root=proc, cgroup_root=group, affinity={0})
    assert result["available_memory_mb"] == 0


@pytest.fixture
def capacity(monkeypatch):
    value = {"cpu_threads": 4, "memory_mb": 8192, "available_memory_mb": 8192,
             "memory_headroom_mb": 512, "cpu_affinity": [0, 1, 2, 3]}
    monkeypatch.setattr(module, "host_capacity", lambda: deepcopy(value))
    return value


def slots():
    return [{"device": str(i), "slot": 0, "memory_mb": 4096, "capacity_mb": 8192} for i in (0, 1)] + [
        {"device": "cpu", "slot": 0, "memory_mb": 8192, "capacity_mb": 8192}]


def enqueue(root, q, name, *, threads=1, host_memory=512, cpu=False, memory=256):
    value = request(root, name, cap=1)
    value["candidate_revision"] = name
    for job in value["jobs"]:
        job["compatibility_key"] = stable_hash((name, job["task_id"]))
        job["resources"] = {"gpus": 0 if cpu else 1, "allow_cpu": cpu, "backend": "cpu" if cpu else "cuda",
                            "memory_mb": memory, "cpu_threads": threads, "host_memory_mb": host_memory}
    return q.submit(value, campaign())["request"]["request_id"]


def test_cpu_threads_reserved_across_distinct_gpu_workers_and_released(capacity, tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    enqueue(tmp_path, q, "a", threads=3)
    first = q.claim(slots())
    assert first["worker"]["host_reservation"]["cpu_threads"] == 3
    second_id = enqueue(tmp_path, q, "b", threads=2)
    assert q.claim(slots()) is None
    assert q.inspect()["submissions"][second_id]["status"] == "queued"
    finish(first)
    q.collect()
    assert q.claim(slots())["request"]["candidate"]["id"] == "b"


def test_cpu_worker_and_gpu_worker_share_the_same_host_memory_reservations(capacity, tmp_path):
    capacity.update(memory_mb=4096, available_memory_mb=3000)
    q = Queue(tmp_path / "queue", grader=grade)
    enqueue(tmp_path, q, "gpu", host_memory=1800)
    first = q.claim(slots())
    enqueue(tmp_path, q, "cpu", host_memory=1800, cpu=True)
    assert q.claim(slots()) is None
    assert first["worker"]["host_reservation"]["host_memory_mb"] == 1800
    finish(first)
    q.collect()
    assert q.claim(slots())["worker"]["device"] == "cpu"


@pytest.mark.parametrize("resources", [{"threads": 5}, {"host_memory": 9000}, {"threads": True}, {"host_memory": -1}])
def test_impossible_host_request_blocks_without_spending_or_hanging(capacity, tmp_path, resources):
    q = Queue(tmp_path / "queue", grader=grade)
    identity = enqueue(tmp_path, q, "impossible", **resources)
    assert q.claim(slots()) is None
    state = q.inspect()
    assert state["submissions"][identity]["status"] == "blocked"
    assert state["submissions"][identity]["reason"].startswith("resource:")
    assert state["campaigns"]["pilot"]["reserved_seconds"] == 0
    assert all(not job["attempts"] for job in state["jobs"].values())


def test_nonfinite_resource_declaration_is_rejected_before_queue_write(capacity, tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    with pytest.raises(ValueError, match="JSON compliant"):
        enqueue(tmp_path, q, "nonfinite", host_memory=float("inf"))
    assert not q.inspect()["submissions"]


def test_external_ram_pressure_without_owned_work_blocks_until_explicit_drain_retry(capacity, tmp_path):
    capacity["available_memory_mb"] = 400
    q = Queue(tmp_path / "queue", grader=grade)
    identity = enqueue(tmp_path, q, "waiting")
    assert q.claim(slots()) is None
    assert "currently available host RAM" in q.inspect()["submissions"][identity]["reason"]
    capacity["available_memory_mb"] = 4096
    assert q.claim(slots())


def test_claims_recheck_capacity_under_queue_lock_and_do_not_double_book(capacity, tmp_path):
    capacity["cpu_threads"] = 2
    q = Queue(tmp_path / "queue", grader=grade)
    enqueue(tmp_path, q, "first", threads=2)
    enqueue(tmp_path, q, "second", threads=2)
    assert q.claim(slots())
    recovered = Queue(q.root, grader=grade)
    assert recovered.claim(slots()) is None
    assert sum(job["status"] == "running" for job in recovered.inspect()["jobs"].values()) == 1


def concurrent_claim(root, barrier, output):
    barrier.wait(timeout=5)
    result = Queue(root).claim(slots())
    output.put(result["worker"]["attempt"] if result else None)


def test_concurrent_claimants_reserve_the_same_host_capacity_atomically(capacity, tmp_path):
    capacity["cpu_threads"] = 2
    q = Queue(tmp_path / "queue", grader=grade)
    enqueue(tmp_path, q, "first", threads=2)
    enqueue(tmp_path, q, "second", threads=2)
    context = multiprocessing.get_context("fork")
    barrier, output = context.Barrier(2), context.Queue()
    processes = [context.Process(target=concurrent_claim, args=(q.root, barrier, output)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=8)
        assert process.exitcode == 0
    assert sum(output.get(timeout=2) is not None for _ in processes) == 1
    assert sum(job["status"] == "running" for job in q.inspect()["jobs"].values()) == 1


def test_slots_use_real_host_ram_and_gpu_sharing_keeps_headroom(capacity, monkeypatch):
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "8192, 12288, Fixture GPU\n")
    with pytest.raises(ValueError, match="sharing"):
        device_slots(["0"], 2)
    gpu = device_slots(["0"], 2, allow_sharing=True)
    assert [row["memory_mb"] for row in gpu[:2]] == [(8192 - 512) // 2] * 2
    assert gpu[0]["capacity_mb"] == (12288 - 512) // 2
    assert gpu[-1]["device"] == "cpu" and gpu[-1]["memory_mb"] == capacity["available_memory_mb"]
    assert len(device_slots(["cpu"], 2, allow_sharing=True)) == 1
    assert len(device_slots(["0"])) == 2  # one GPU worker plus one CPU slot


def test_gpu_free_memory_pressure_is_not_mistaken_for_capacity_or_spinning(capacity, tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    identity = enqueue(tmp_path, q, "gpu", memory=2000)
    unavailable = [{"device": "0", "slot": 0, "memory_mb": 500, "capacity_mb": 8192}]
    assert q.claim(unavailable) is None
    assert q.inspect()["submissions"][identity]["status"] == "blocked"
    unavailable[0]["memory_mb"] = 4096
    assert q.claim(unavailable)


def test_shared_gpu_requires_positive_memory_reservation(capacity, tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    identity = enqueue(tmp_path, q, "unbounded", memory=0)
    shared = [{**row, "sharing": True} for row in slots() if row["device"] != "cpu"]
    assert q.claim(shared) is None
    assert q.inspect()["submissions"][identity]["status"] == "blocked"


def test_resource_threads_must_match_frozen_scientific_profile(capacity, tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    value = request(tmp_path, cap=1)
    value["jobs"][0]["science"]["compute"] = {"threads": 2}
    value["jobs"][0]["resources"]["cpu_threads"] = 1
    identity = q.submit(value, campaign())["request"]["request_id"]
    assert q.claim(slots()) is None
    assert "frozen scientific compute" in q.inspect()["submissions"][identity]["reason"]


def test_multi_gpu_task_is_blocked_before_single_gpu_worker_claim(capacity, tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    value = request(tmp_path, cap=1)
    value["jobs"][0]["resources"]["gpus"] = 2
    identity = q.submit(value, campaign())["request"]["request_id"]
    assert q.claim(slots()) is None
    state = q.inspect()
    assert state["submissions"][identity]["status"] == "blocked"
    assert "zero or one GPU" in state["submissions"][identity]["reason"]
    assert not any(job["attempts"] for job in state["jobs"].values())
