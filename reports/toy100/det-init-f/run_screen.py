"""CPU screen for family F. One JSON line per job on stdout and in progress.jsonl.

    python -u reports/toy100/det-init-f/run_screen.py determinism
    python -u reports/toy100/det-init-f/run_screen.py baseline
    python -u reports/toy100/det-init-f/run_screen.py wave g
    python -u reports/toy100/det-init-f/run_screen.py wave x
    python -u reports/toy100/det-init-f/run_screen.py priority hid_q_r2 qr_pb_pq_sobol

Tails: ``tail -f results/det-init-f/progress.jsonl``
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "det-init-f"
PROGRESS = OUT / "progress.jsonl"
OFFSETS = (0, 101, 202, 303, 404, 505, 606, 707)
SCREEN_GATES = ("ring", "unequal")
SEQUENCES = ("r2", "sobol", "halton", "fib", "r1", "strat", "lhs", "weyl")
MAPS = ("g", "x", "b", "s")
ARMS = ("hid_q", "qr_pb_pq")


def init_name(arm: str, seq: str, mp: str) -> str:
    if arm == "hid_q" and seq == "weyl" and mp == "g":
        return "hid_q"
    if arm == "qr_pb_pq" and seq == "r2" and mp == "g":
        return "qr_pb_pq"
    if mp == "g":
        return f"{arm}_{seq}"
    return f"{arm}_{seq}_{mp}"


def names_for_map(mp: str) -> list[str]:
    return [init_name(arm, seq, mp) for arm in ARMS for seq in SEQUENCES]


def log(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True)
    with PROGRESS.open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def done_keys() -> set[tuple]:
    if not PROGRESS.exists():
        return set()
    found = set()
    for line in PROGRESS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "done" and row.get("status") != "ERROR":
            found.add((row.get("phase"), row.get("init"), row.get("gate"), row.get("seed_offset")))
    return found


def run_one(phase: str, init: str | None, gate: str, offset: int, init_only: bool = False) -> dict:
    label = init or "k3p"
    key = (phase, label, gate, offset)
    if key in done_keys():
        return {"event": "skip", "phase": phase, "init": label, "gate": gate, "seed_offset": offset}
    output = OUT / "runs" / label / f"s{offset}" / ("init-" + gate if init_only else gate)
    log_path = OUT / "logs" / f"{label}-{gate}-s{offset}{'-init' if init_only else ''}.log"
    if output.exists():
        subprocess.run(["rm", "-rf", str(output)], check=True)
    cmd = [sys.executable, "-u", "-m", "benchmarks.toy100.det_init_screen",
           "--gate", gate, "--output", str(output), "--log", str(log_path),
           "--seed-offset", str(offset)]
    if init:
        cmd.extend(["--init", init])
    if init_only:
        cmd.append("--init-only")
    log({"event": "start", "phase": phase, "init": label, "gate": gate, "seed_offset": offset,
         "init_only": init_only, "log": str(log_path)})
    started = time.time()
    done = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    row = {"event": "done", "phase": phase, "init": label, "gate": gate, "seed_offset": offset,
           "init_only": init_only, "wall_seconds": round(time.time() - started, 1),
           "driver_returncode": done.returncode}
    text = done.stdout.strip().splitlines()
    if text:
        try:
            row.update({k: v for k, v in json.loads(text[-1]).items() if k not in row})
        except json.JSONDecodeError:
            row["status"] = "ERROR"
            row["driver_tail"] = text[-1][:400]
    else:
        row["status"] = "ERROR"
    # Checkpoints are reproducible from the recipe; the JSON row is the screen record.
    for name in ("final-state.pt", "initial-values.pt"):
        path = output / name
        if path.exists():
            path.unlink()
    log(row)
    return row


def check_determinism() -> None:
    rows = [json.loads(line) for line in PROGRESS.read_text().splitlines() if line.strip()]
    grouped = {}
    for row in rows:
        if row.get("event") != "done" or row.get("phase") != "determinism":
            continue
        grouped.setdefault((row.get("init"), row.get("gate")), {})[row.get("seed_offset")] = row
    bad = 0
    for key, offsets in sorted(grouped.items()):
        a, b = offsets.get(0), offsets.get(101)
        if not a or not b or a.get("status") == "ERROR" or b.get("status") == "ERROR":
            bad += 1
            log({"event": "determinism_fail", "init": key[0], "gate": key[1], "reason": "missing"})
            continue
        if a.get("param_sha256") != b.get("param_sha256") or not a.get("param_sha256"):
            bad += 1
            log({"event": "determinism_fail", "init": key[0], "gate": key[1],
                 "sha0": a.get("param_sha256"), "sha101": b.get("param_sha256")})
    log({"event": "determinism_summary", "checked": len(grouped), "bad": bad})
    if bad:
        raise SystemExit(1)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    phase = sys.argv[1]
    if phase == "check-determinism":
        check_determinism()
        return
    from particlegan.det_init import torch_build
    log({"event": "build", "phase": phase, **torch_build()})
    jobs = int(os.environ.get("K3P_SCREEN_JOBS", "3"))
    tasks = []
    if phase == "determinism":
        selected = sys.argv[2:] or [init_name(arm, seq, mp) for arm in ARMS for seq in SEQUENCES for mp in MAPS]
        for name in selected:
            for gate in SCREEN_GATES:
                for offset in (0, 101):
                    tasks.append(("determinism", name, gate, offset, True))
    elif phase == "baseline":
        for gate in SCREEN_GATES:
            for offset in OFFSETS:
                tasks.append(("baseline", None, gate, offset, False))
    elif phase == "wave":
        mp = sys.argv[2]
        if mp not in MAPS:
            raise SystemExit(f"map must be one of {MAPS}")
        for name in names_for_map(mp):
            for gate in SCREEN_GATES:
                for offset in OFFSETS:
                    tasks.append((f"wave-{mp}", name, gate, offset, False))
    elif phase == "priority":
        for name in sys.argv[2:]:
            for gate in ("ring", "unequal", "stripes", "blobs", "hold", "shift", "grid100", "rotated100"):
                tasks.append(("priority", name, gate, 0, False))
    elif phase == "names":
        mp = sys.argv[2] if len(sys.argv) > 2 else "g"
        for name in names_for_map(mp):
            print(name)
        return
    else:
        raise SystemExit(__doc__)
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(lambda item: run_one(*item), tasks))


if __name__ == "__main__":
    main()
