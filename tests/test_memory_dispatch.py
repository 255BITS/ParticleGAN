import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import pytest

from experiments.memory_dispatch import add, initialize


DISPATCH = Path(__file__).resolve().parents[1]/"experiments/memory_dispatch.py"
STUB = '''import argparse, json, os, pathlib, sys
p = argparse.ArgumentParser()
for flag in ("config", "out", "device"):
    p.add_argument("--"+flag, required=True)
a = p.parse_args()
c = json.loads(pathlib.Path(a.config).read_text())
print("progress " + c["name"], flush=True)
print("diagnostic " + c["name"], file=sys.stderr, flush=True)
if "ready" in c:
    with open(c["ready"], "w") as f:
        f.write(json.dumps([c["name"], a.device])+"\\n")
    fd = os.open(c["gate"], os.O_RDONLY)
    os.read(fd, 1)
    os.close(fd)
if c.get("fail"):
    sys.exit(7)
out = pathlib.Path(a.out)
out.mkdir()
(out/"summary.json").write_text(json.dumps({"metrics": {"generated_256": {"pass": 0.5}}}))
'''


def config(tmp_path, name, **kwargs):
    path = tmp_path/f"{name}.json"
    path.write_text(json.dumps({"name": name, **kwargs}))
    return path


def command(queue, *args):
    return [sys.executable, str(DISPATCH), *args, "--queue", str(queue)]


def start(queue):
    return subprocess.Popen(command(queue, "drain", "--devices", "cpu:0", "cpu:1"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def seal(queue):
    subprocess.run(command(queue, "seal"), check=True, capture_output=True, timeout=10)


@pytest.fixture
def setup(tmp_path):
    queue = tmp_path/"queue"
    initialize(queue)
    trainer = tmp_path/"trainer.py"
    trainer.write_text(STUB)
    return queue, trainer


def test_parallel_drain_central_logs_and_failure(tmp_path, setup):
    queue, trainer = setup
    ready, gate = tmp_path/"ready", tmp_path/"gate"
    os.mkfifo(ready)
    os.mkfifo(gate)
    ready_fd = os.open(ready, os.O_RDWR | os.O_NONBLOCK)
    gate_fd = os.open(gate, os.O_RDWR)
    paths = [config(tmp_path, name, fail=fail, ready=str(ready), gate=str(gate))
             for name, fail in (("success", False), ("failure", True))]
    add(queue, paths+[config(tmp_path, "after_failure")], trainer)
    seal(queue)
    process = start(queue)
    try:
        lines, deadline = b"", time.monotonic()+10
        while lines.count(b"\n") < 2:
            available, _, _ = select.select([ready_fd], [], [], max(0, deadline-time.monotonic()))
            assert available, "Both workers must launch concurrently"
            lines += os.read(ready_fd, 4096)
        assert {json.loads(line)[1] for line in lines.splitlines()} == {"cpu:0", "cpu:1"}
        os.write(gate_fd, b"!!")
        stdout, stderr = process.communicate(timeout=15)
    finally:
        os.write(gate_fd, b"!!")
        if process.poll() is None:
            process.kill()
            process.communicate()
        os.close(ready_fd)
        os.close(gate_fd)
    assert process.returncode == 1, stderr
    events = [json.loads(line) for line in stdout.splitlines()]
    assert sorted(e["event"] for e in events) == ["completed", "completed", "failed", "queue_complete"]
    assert events[-1]["failed"] == ["failure"]
    assert len(list((queue/"done").glob("*.json"))) == 2
    assert len(list((queue/"failed").glob("*.json"))) == 1
    assert not list((queue/"running").glob("*.json"))
    central = (queue/"train.log").read_text()
    for event in events[:-1]:
        name, device = event["name"], event["device"]
        assert f"[{name} {device}] progress {name}\n" in central
        assert f"[{name} {device}] diagnostic {name}\n" in central
        assert (queue/f"{name}.console.log").read_text() == f"progress {name}\ndiagnostic {name}\n"
    assert "launch" in (queue/"queue.log").read_text()
    assert "queue_complete" in central


def test_dynamic_add_wakes_empty_drain_and_duplicate_is_rejected(tmp_path, setup):
    queue, trainer = setup
    process = start(queue)
    try:
        # Blocks until the drain has opened its notification FIFO, with an outer timeout.
        subprocess.run([sys.executable, "-c", "import os,sys; os.close(os.open(sys.argv[1],os.O_WRONLY))",
                        str(queue/"wake")], check=True, timeout=10)
        path = config(tmp_path, "late")
        add(queue, [path], trainer)
        with pytest.raises(ValueError, match="Duplicate"):
            add(queue, [path], trainer)
        seal(queue)
        stdout, stderr = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 0, stderr
    assert [json.loads(line)["event"] for line in stdout.splitlines()] == ["completed", "queue_complete"]
    assert "[late cpu:" in (queue/"train.log").read_text()
    with pytest.raises(ValueError, match="sealed"):
        add(queue, [config(tmp_path, "too_late")], trainer)


def test_reporter_runs_after_success_before_notification(tmp_path, setup):
    queue, trainer = setup
    reporter = tmp_path/'reporter.py'
    reporter.write_text('''import argparse, json, pathlib
p = argparse.ArgumentParser()
p.add_argument('--source', type=pathlib.Path)
p.add_argument('--out', type=pathlib.Path)
a = p.parse_args()
assert len(list((a.source.parent/'done').glob('*.json'))) == 1
a.out.mkdir(exist_ok=True)
(a.out/'ready').write_text('complete')
''')
    report = tmp_path/'report'
    add(queue, [config(tmp_path, 'success')], trainer)
    seal(queue)
    process = subprocess.run(command(queue, 'drain', '--devices', 'cpu:0',
                                    '--reporter', str(reporter), '--report-out', str(report)),
                             capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert (report/'ready').read_text() == 'complete'
    assert [json.loads(line)['event'] for line in process.stdout.splitlines()] == ['completed', 'queue_complete']
