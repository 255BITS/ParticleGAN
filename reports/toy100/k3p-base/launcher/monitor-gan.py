#!/usr/bin/env python3
"""Live, read-only terminal dashboard for try-gan.sh batches (standard library only).

Follows every registered batch at once, so concurrent Codex and Claude/opus
searches share one screen. Default: gan-attempts/active-batches.txt, falling
back to current-batch.txt; --batch pins one or more batches instead.
"""

import argparse
from collections import Counter, deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parent
GATES = (
    "trajectory", "mode_hold", "residual_student", "img_stripes2", "img_bars4",
    "vector_overlap", "img_blobs4", "img_intensity2", "vector_unequal_mass",
    "vector_unequal_width",
)
EXECUTED = {"PASS", "FAIL", "ERROR"}
AGENT_LOGS = ("codex.log", "claude.log", "grok.log", "agent.log")
KEEP_FINISHED_SECONDS = 1800  # Keep a just-finished batch on screen for a while.
COSTS = {}


def read_text(path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return ""


class Ledger:
    """Read appended complete lines; never retain bulky sample/observation arrays."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.offset = 0
        self.identity = None
        self.modified = None
        self.pending = b""
        self.counts = Counter()
        self.candidates = set()
        self.latest = {}
        self.recent = deque(maxlen=3)
        self.regression = None
        self.bad_lines = 0

    def update(self, path):
        try:
            with path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if self.identity != identity or stat.st_size < self.offset or (
                    stat.st_size == self.offset and self.modified != stat.st_mtime_ns
                ):
                    self.reset()
                self.identity = identity
                self.modified = stat.st_mtime_ns
                stream.seek(self.offset)
                chunk = stream.read()
                self.offset = stream.tell()
        except OSError:
            return
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a receipt object")
                candidate, gate, status = (row.get(k) for k in ("candidate", "gate", "status"))
                if not all(isinstance(v, str) for v in (candidate, gate, status)):
                    raise ValueError("Missing receipt identity")
                metrics = row.get("metrics") or {}
                if not isinstance(metrics, dict):
                    raise ValueError("Invalid metrics object")
                if candidate == "regression":
                    self.regression = row
                    continue
                self.counts[status] += 1
                compact = {
                    "candidate": candidate, "gate": gate, "status": status,
                    "live": next((metrics[k] for k in ("live", "final")
                                  if isinstance(metrics.get(k), dict)), metrics),
                }
                self.latest[(candidate, gate)] = compact
                if status in EXECUTED:
                    self.candidates.add(candidate)
                    self.recent.append(compact)
            except (ValueError, TypeError, AttributeError):
                self.bad_lines += 1

    def leader(self, gates=GATES):
        def prefix(candidate):
            passed = 0
            for gate in gates:
                if self.latest.get((candidate, gate), {}).get("status") != "PASS":
                    break
                passed += 1
            return passed
        if not self.candidates:
            return "waiting for measured candidates"
        name = max(sorted(self.candidates), key=prefix)
        passed = prefix(name)
        if passed == 0 and self.recent:
            row = self.recent[-1]
            live = row["live"]
            values = []
            for key in ("step", "modes", "hq"):
                if isinstance(live.get(key), (int, float)):
                    values.append(f"{key} {live[key]:.6g}")
            detail = "; " + ", ".join(values) if values else ""
            return f"latest {row['gate']} {row['status']}{detail} | {row['candidate']}"
        detail = "focused screen complete" if passed == len(gates) else "awaiting " + gates[passed]
        if passed < len(gates):
            next_row = self.latest.get((name, gates[passed]), {})
            if next_row.get("status") in {"FAIL", "ERROR"}:
                detail = next_row["status"] + " " + gates[passed]
        ring = self.latest.get((name, "mode_hold"), {}).get("live", {})
        if isinstance(ring.get("modes"), (int, float)):
            detail += f"; modes {ring['modes']:g}"
        return f"{passed}/{len(gates)}; {detail} | {name}"


def engine_of(record, run):
    """Batches predate the engine field; a run's own run.txt settles it."""
    engine = record.get("engine")
    if not engine:
        for line in read_text(run / "run.txt").splitlines():
            if line.startswith("engine="):
                engine = line.split("=", 1)[1].strip()
    return engine or "codex"


def run_cost(run):
    """Dollars reported by one finished Claude/Grok attempt (unavailable for Codex)."""
    path = next((run / name for name in ('grok-usage.json', 'claude-usage.json')
                 if (run / name).exists()), run / 'claude-usage.json')
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return 0.0
    if COSTS.get(path, (None,))[0] != stamp:
        try:
            COSTS[path] = (stamp, float(json.loads(path.read_text()).get("total_cost_usd") or 0))
        except (OSError, ValueError, TypeError):
            COSTS[path] = (stamp, 0.0)
    return COSTS[path][1]


def process_alive(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return False
        return True
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError, IndexError):
        return False


def state(record, run):
    status = read_text(run / "status.txt")
    if status.startswith("completed"):
        return "completed"
    if status.startswith("timed out"):
        return "timeout"
    if status.startswith("failed"):
        return "failed"
    return "running" if process_alive(record.get("pid")) else "exited"


def activity_age(run):
    modified = []
    for name in AGENT_LOGS + ("tests.jsonl", "status.txt", "result.md"):
        try:
            modified.append((run / name).stat().st_mtime)
        except OSError:
            pass
    if not modified:
        return "--"
    age = max(0, int(time.time() - max(modified)))
    return f"{age}s" if age < 60 else f"{age // 60}m"


def collect(batch, records, ledgers):
    """One row per attempt slot: its newest run, its engine and its ledger."""
    attempts = []
    for record in (r for r in records if not r.get("stopped_by_supervisor")):
        directory = Path(record["directory"])
        runs = sorted(p for p in directory.glob("20*") if p.is_dir())
        if not runs:
            continue
        run = runs[-1]
        ledger = ledgers.setdefault(run, Ledger())
        ledger.update(run / "tests.jsonl")
        attempts.append(dict(lane=record["lane"], run=run, state=state(record, run),
                             ledger=ledger, engine=engine_of(record, run), tries=len(runs),
                             cost=sum(run_cost(r) for r in runs),
                             gates=record.get("gate_order", GATES)))
    return attempts


def fresh(batch, attempts):
    """Drop batches that finished a while ago; a live pool always stays."""
    if any(a["state"] == "running" for a in attempts):
        return True
    try:
        newest = max((a["run"].stat().st_mtime for a in attempts), default=batch.stat().st_mtime)
    except OSError:
        return False
    return time.time() - newest < KEEP_FINISHED_SECONDS


def render(batches, ledgers, interval):
    boards, running, total_gates, spend, excluded = [], 0, 0, 0.0, 0
    for batch, records in batches:
        attempts = collect(batch, records, ledgers)
        if not fresh(batch, attempts):
            continue
        boards.append((batch, attempts))
        running += sum(a["state"] == "running" for a in attempts)
        total_gates += sum(sum(a["ledger"].counts[k] for k in EXECUTED) for a in attempts)
        spend += sum(a["cost"] for a in attempts)
        excluded += len(records) - len(attempts)
    lines = [
        f"ParticleGAN | {running} running | {total_gates} gate executions"
        + (f" | claude ${spend:.2f} billed" if spend else "")
        + f" | {datetime.now():%H:%M:%S}",
        f"Refresh {interval:g}s. Ctrl-C closes this display; experiments keep running.",
    ]
    for index, (batch, attempts) in enumerate(boards, 1):
        engines = sorted({a["engine"] for a in attempts}) or ["-"]
        lines += ["", f"[b{index}] {batch.name}  [{'+'.join(engines)}]",
                  f"{'Attempt':30} {'Engine':7} {'State':10} {'Try':>3} {'Cands':>5} "
                  f"{'Gates':>6} {'Pass':>5} {'Fail':>5} {'Err':>4} {'Idle':>5}"]
        for attempt in attempts:
            c = attempt["ledger"].counts
            lines.append(f"{attempt['lane'][:30]:30} {attempt['engine'][:7]:7} {attempt['state']:10} "
                         f"{attempt['tries']:3} {len(attempt['ledger'].candidates):5} "
                         f"{sum(c[k] for k in EXECUTED):6} {c['PASS']:5} {c['FAIL']:5} "
                         f"{c['ERROR']:4} {activity_age(attempt['run']):>5}")
    lines += ["", "Focused gate progress (full 22-toy and stability qualification are separate):"]
    for index, (_, attempts) in enumerate(boards, 1):
        for attempt in attempts:
            lines.append(f"  b{index} {attempt['lane']}: {attempt['ledger'].leader(attempt['gates'])}")
    lines += ["", "Latest completed gate / latest regression run:"]
    for index, (_, attempts) in enumerate(boards, 1):
        for attempt in attempts:
            ledger = attempt["ledger"]
            last = ledger.recent[-1] if ledger.recent else None
            receipt = (f"{last['candidate']} {last['gate']} {last['status']}"
                       if last else "waiting for gates")
            unit = ledger.regression
            if unit:
                metrics = unit.get("metrics") or {}
                passed = metrics.get("passed")
                if passed is None and "tests" in metrics:
                    try:
                        passed = int(metrics["tests"]) - sum(
                            int(metrics.get(key, 0)) for key in ("failures", "errors", "skipped"))
                    except (TypeError, ValueError):
                        pass
                if passed is None:
                    match = re.search(r"\b(\d+) passed\b", str(metrics.get("summary", "")))
                    passed = int(match[1]) if match else "?"
                unit_text = (f"unit {unit['status']}: {passed} passed, "
                             f"{metrics.get('failures', metrics.get('failed', 0))} failed")
                if passed == "?":
                    unit_text = f"unit {unit['status']} (count not supplied)"
            else:
                unit_text = "unit results pending"
            lines.append(f"  b{index} {attempt['lane']}: {receipt} | {unit_text}")
    skipped = sum(a["ledger"].counts["SKIPPED"] for _, attempts in boards for a in attempts)
    bad = sum(a["ledger"].bad_lines for _, attempts in boards for a in attempts)
    lines += ["", f"Skipped gates: {skipped} (not executed). Excluded stopped attempts: {excluded}.",
              f"Batches: {len(boards)} of {len(batches)} registered."]
    if bad:
        lines.append(f"Unreadable complete receipt lines: {bad}; counts may be incomplete.")
    if boards and running == 0:
        lines.append("No attempts running. Inspect result.md/final.md. Waiting for a new batch.")
    if not boards:
        lines.append("No live batches. Launch one, or pass --batch to pin a finished batch.")
    return "\n".join(lines)


def registered(pinned):
    """Batches to follow: pinned ones, else every live batch, else the current one."""
    if pinned:
        return [Path(p).expanduser().resolve() for p in pinned]
    paths, seen = [], set()
    for name in ("gan-attempts/active-batches.txt", "gan-attempts/current-batch.txt"):
        for line in read_text(ROOT / name).splitlines():
            line = line.strip()
            if not line:
                continue
            batch = Path(line).expanduser().resolve()
            if batch not in seen and (batch / "batch.json").exists():
                seen.add(batch)
                paths.append(batch)
        if paths:
            break
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append",
                        help="Pin a batch (repeatable); default follows every live batch")
    parser.add_argument("--interval", type=float, default=2, help="Refresh seconds (default: 2)")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 0.2:
        parser.error("--interval must be a finite number of at least 0.2 seconds")
    terminal = sys.stdout.isatty() and not args.once
    ledgers = {}
    previous = None
    if terminal:
        sys.stdout.write("\033[?1049h\033[?25l")
    try:
        while True:
            try:
                paths = registered(args.batch)
                if not paths:
                    raise ValueError("No batch registered yet")
                if paths != previous:
                    ledgers.clear()
                    previous = paths
                batches = []
                for batch in paths:
                    try:
                        batches.append((batch, json.loads((batch / "batch.json").read_text())))
                    except (OSError, ValueError):
                        continue
                if not batches:
                    raise ValueError("No readable batch.json yet")
                screen = render(batches, ledgers, args.interval)
            except (OSError, ValueError, KeyError, TypeError) as error:
                screen = f"Waiting for GAN batch data: {error}\nCtrl-C exits the monitor."
            if terminal:
                columns, rows = shutil.get_terminal_size((100, 30))
                # Prevent long candidate IDs/paths from wrapping or scrolling the screen.
                screen = "\n".join(line[:max(1, columns - 1)]
                                   for line in screen.splitlines()[:max(1, rows - 1)])
                sys.stdout.write("\033[H\033[2J")
            print(screen, flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        if terminal:
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
