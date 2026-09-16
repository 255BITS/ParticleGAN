"""Append configs to a durable shared GPU queue; drain emits completion events only.

add --queue Q --configs a.json b.json [--trainer experiments/memory_scout.py]
drain --queue Q [--devices cuda:0 cuda:1]
seal --queue Q   # exit after all submitted jobs finish

Workers block on child completion or a FIFO notification, never tail training logs.
Tail Q/train.log for labelled output from every job plus lifecycle events. Drain
stdout contains only completed/failed/queue_complete notifications.
Pending jobs survive a drain restart. Interrupted running jobs require inspection;
they are deliberately not retried automatically into an existing output directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import threading
import time


@contextmanager
def locked(path):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def initialize(queue):
    queue.mkdir(parents=True, exist_ok=True)
    for name in ("pending", "running", "done", "failed", "runs"):
        (queue/name).mkdir(exist_ok=True)
    try:
        os.mkfifo(queue/"wake")
    except FileExistsError:
        pass


def wake(queue, count):
    fd = os.open(queue/"wake", os.O_RDWR | os.O_NONBLOCK)
    try:
        os.write(fd, b"!"*count)
    except BlockingIOError:
        pass  # A full FIFO already guarantees workers will wake.
    finally:
        os.close(fd)


def add(queue, configs, trainer):
    trainer = trainer.resolve(strict=True)
    entries = []
    for config in configs:
        config = config.resolve(strict=True)
        name = json.loads(config.read_text())["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Unsafe run name: {name}")
        entries.append((name, {"name": name, "config": str(config), "trainer": str(trainer),
                               "cwd": str(Path.cwd()), "submitted": time.time()}))
    with locked(queue/"queue.lock"):
        if (queue/"SEALED").exists():
            raise ValueError("Queue is sealed; create a new queue")
        existing = {json.loads(p.read_text())["name"] for state in ("pending", "running", "done", "failed")
                    for p in (queue/state).glob("*.json")}
        names = [name for name, _ in entries]
        if len(set(names)) != len(names) or existing.intersection(names):
            raise ValueError("Duplicate run name")
        for name, job in entries:
            (queue/"pending"/f"{time.time_ns()}_{name}.json").write_text(json.dumps(job)+"\n")
    wake(queue, len(entries))


def drain(queue, devices, reporter=None, report_out=None):
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Each device may have only one worker")
    guard = (queue/"drain.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((queue/"running").glob("*.json")):
        guard.close()
        raise RuntimeError("Unresolved running jobs: inspect before restarting drain")
    event_lock = threading.Lock()
    report_lock = threading.Lock()
    failures = []
    fd = os.open(queue/"wake", os.O_RDWR | os.O_NONBLOCK)
    with (queue/"queue.log").open("a", buffering=1) as log, \
            (queue/"train.log").open("a", buffering=1) as training:
        def event(notify=False, **row):
            line = json.dumps({"time": time.time(), **row}, allow_nan=False)
            with event_lock:
                log.write(line+"\n")
                training.write(line+"\n")
                if notify:
                    print(line, flush=True)

        def claim():
            with locked(queue/"queue.lock"):
                pending = sorted((queue/"pending").glob("*.json"))
                if pending:
                    path = pending[0]
                    target = queue/"running"/path.name
                    path.rename(target)
                    return target, False
                return None, (queue/"SEALED").exists()

        def worker(device):
            while True:
                path, sealed = claim()
                if path is None:
                    if sealed:
                        return
                    select.select([fd], [], [])
                    try:
                        os.read(fd, 1)
                    except BlockingIOError:
                        pass
                    continue
                job = json.loads(path.read_text())
                name, out = job["name"], queue/"runs"/job["name"]
                job.update(device=device, out=str(out), started=time.time())
                with locked(queue/"queue.lock"):
                    path.write_text(json.dumps(job)+"\n")
                event(event="launch", name=name, device=device, out=str(out))
                error, metrics = None, None
                try:
                    with (queue/f"{name}.console.log").open("w", buffering=1) as console:
                        with subprocess.Popen([sys.executable, "-u", job["trainer"], "--config", job["config"],
                                               "--out", str(out), "--device", device],
                                              cwd=job["cwd"], stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, text=True, errors="replace") as child:
                            for line in child.stdout:
                                console.write(line)
                                with event_lock:
                                    training.write(f"[{name} {device}] {line.rstrip(chr(10))}\n")
                            code = child.wait()
                    if code:
                        raise RuntimeError(f"Trainer exited {code}; see {queue/name}.console.log")
                    result = json.loads((out/"summary.json").read_text())
                    metrics = result["metrics"]["generated_256"]
                except Exception as exc:
                    error = str(exc)
                    failures.append(name)
                state = "failed" if error else "done"
                job.update(finished=time.time(), error=error, metrics=metrics)
                with locked(queue/"queue.lock"):
                    path.write_text(json.dumps(job, allow_nan=False)+"\n")
                    path.rename(queue/state/path.name)
                if not error and reporter is not None:
                    # Only completed jobs enter reports. Serialize report writes,
                    # while the other GPU and its log reader continue working.
                    with report_lock:
                        report = subprocess.run([sys.executable, str(reporter),
                            "--source", str(queue/"runs"), "--out", str(report_out)],
                            capture_output=True, text=True)
                        if report.returncode:
                            failures.append(f"report:{name}")
                            event(True, event="report_failed", name=name,
                                  error=report.stderr[-4000:])
                event(True, event="failed" if error else "completed", name=name, device=device,
                      out=str(out), error=error, metrics=metrics)

        try:
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                list(pool.map(worker, devices))
            event(True, event="queue_complete", failed=failures)
        finally:
            os.close(fd)
            guard.close()
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("add", "drain", "seal"):
        part = sub.add_parser(command)
        part.add_argument("--queue", type=Path, required=True)
        if command == "add":
            part.add_argument("--configs", nargs="+", type=Path, required=True)
            part.add_argument("--trainer", type=Path, default=Path("experiments/memory_scout.py"))
        elif command == "drain":
            part.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
            part.add_argument("--reporter", type=Path)
            part.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    queue = args.queue.resolve()
    initialize(queue)
    if args.command == "add":
        add(queue, args.configs, args.trainer)
    elif args.command == "seal":
        with locked(queue/"queue.lock"):
            (queue/"SEALED").touch()
        wake(queue, 256)
    else:
        if bool(args.reporter) != bool(args.report_out):
            parser.error("--reporter and --report-out must be provided together")
        sys.exit(drain(queue, args.devices,
                       args.reporter.resolve(strict=True) if args.reporter else None,
                       args.report_out.resolve() if args.report_out else None))


if __name__ == "__main__":
    main()
