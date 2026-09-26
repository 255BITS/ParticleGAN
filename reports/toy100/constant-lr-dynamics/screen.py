"""Constant-LR ring, hold, stay, and unequal for one or more dynamics.

Learning rates stay at K3P's pre-anneal values (network 0.00425, prior 0.0085):
``lr_anneal_start`` is 0 and both floors are 1. ``--dynamics baseline`` is that
schedule with no mechanism. Other names set ``K3P_DYNAMICS`` for the child.

CPU numbers do not rank against the A6000. One JSON line per finished job.

    python -u reports/toy100/constant-lr-dynamics/screen.py --dynamics baseline --gates ring
    python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics unit_rms --gates ring hold shift unequal
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
K3P = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p"
OFFSETS = (0, 101, 202, 303, 404, 505, 606, 707)
GATES = ("ring", "hold", "shift", "unequal")
DYNAMICS = ("baseline", "unit_rms", "pair_chord", "shared_batch", "unrolled")


def constant_config(source: Path, dest: Path) -> dict:
    config = json.loads(source.read_text())
    config["lr_anneal_start"] = 0.0
    config["lr_floor"] = 1.0
    config["network_lr_floor"] = 1.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(config, indent=2) + "\n")
    sys.path.insert(0, str(ROOT))
    from benchmarks.toy100.schedule import policy_multipliers
    for step in (0, 720, 721, 1199, 2400, 4000):
        network, prior = policy_multipliers(
            step, 1200, 0.0, 1.0, config["network_lr_horizon_cap"], network_lr_floor=1.0,
        )
        if network != 1.0 or prior != 1.0:
            raise SystemExit(f"constant schedule is not 1 at step {step}: {network}, {prior}")
    return config


def command(gate: str, config: Path, output: Path, backend: str) -> list[str]:
    common = ["--repo", str(ROOT), "--config", str(config), "--backend", backend,
              "--init", "hid_q", "--output", str(output)]
    if gate == "ring":
        return [sys.executable, "-u", str(K3P / "probe.py"), *common, "--task", "mode_hold"]
    if gate == "unequal":
        return [sys.executable, "-u", str(K3P / "probe.py"), *common, "--task", "vector_unequal_mass"]
    script = K3P / ("hold.py" if gate == "hold" else "shift.py")
    return [sys.executable, "-u", str(script), *common, "--task", "mode_hold",
            "--network-floor", "1", "--prior-floor", "1", "--anneal-start", "0"]


def summarize(gate: str, output: Path, code: int) -> dict:
    row = {"gate": gate, "returncode": code}
    path = output / "result.json"
    if not path.exists():
        row["status"] = "ERROR"
        return row
    payload = json.loads(path.read_text())
    row["status"] = payload.get("status")
    row["seconds"] = payload.get("seconds")
    if gate == "hold":
        gate_row = payload.get("gate") or {}
        row["hold_checks"] = gate_row.get("hold_checks")
        row["converged_step"] = gate_row.get("converged_step")
        row["pass_1200"] = gate_row.get("hold_checks") == 1200 and row["status"] == "PASS"
    elif gate == "shift":
        hold = payload.get("continued_hold") or {}
        row["stay"] = f"{hold.get('passing_checks')}/{hold.get('checks')}"
        row["pass_all"] = hold.get("pass_all")
        row["stay_pass"] = hold.get("passing_checks") == 120 and hold.get("checks") == 120 and bool(hold.get("pass_all"))
    else:
        live = (payload.get("result") or {}).get("live") or {}
        row["modes"] = live.get("modes")
        row["hq"] = live.get("hq")
        conv = (payload.get("result") or {}).get("convergence") or {}
        row["passing_suffix"] = conv.get("passing_suffix")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamics", nargs="+", default=["baseline"], choices=DYNAMICS)
    parser.add_argument("--gates", nargs="+", default=["ring", "hold", "shift"], choices=GATES)
    parser.add_argument("--offsets", nargs="+", type=int, default=list(OFFSETS))
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--root", type=Path, default=Path("/tmp/k3p-clr"))
    args = parser.parse_args()
    config_path = args.root / "config.json"
    config = constant_config(K3P / "config.json", config_path)
    print(json.dumps({
        "event": "constant_lr",
        "lr": config["lr"],
        "prior_lr": config["lr"] * config["prior_lr_mult"],
        "lr_anneal_start": config["lr_anneal_start"],
        "floors": [config["lr_floor"], config["network_lr_floor"]],
        "dynamics": args.dynamics,
        "gates": args.gates,
        "backend": args.backend,
    }), flush=True)
    pending = [(name, gate, offset)
               for name in args.dynamics for gate in args.gates for offset in args.offsets]
    print(f"queued {len(pending)}", flush=True)
    running = []
    index = 0
    rows = []
    while index < len(pending) or running:
        while index < len(pending) and len(running) < args.jobs:
            name, gate, offset = pending[index]
            index += 1
            out = args.root / name / f"{gate}-s{offset}"
            log = args.root / "logs" / f"{name}-{gate}-s{offset}.log"
            if out.exists():
                raise SystemExit(f"output exists: {out}")
            log.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
            env["PYTHONPATH"] = str(HERE) + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            if name != "baseline":
                env["K3P_DYNAMICS"] = name
            else:
                env.pop("K3P_DYNAMICS", None)
            cmd = command(gate, config_path, out, args.backend)
            if offset:
                env["K3P_SEED_OFFSET"] = str(offset)
                shim = ROOT / "benchmarks/toy100/det_init_seedshim.py"
                cmd = [sys.executable, "-u", str(shim), *cmd[2:]]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)
            running.append((name, gate, offset, proc, time.time(), log, out))
            print(f"START {name} {gate} s{offset}", flush=True)
        still = []
        for name, gate, offset, proc, t0, log, out in running:
            if proc.poll() is None:
                still.append((name, gate, offset, proc, t0, log, out))
                continue
            row = summarize(gate, out, proc.returncode)
            row.update(dynamics=name, offset=offset, seconds_wall=round(time.time() - t0, 1))
            rows.append(row)
            print("DONE " + json.dumps(row, sort_keys=True), flush=True)
        running = still
        if running:
            time.sleep(0.5)
    summary = args.root / "summary.jsonl"
    with summary.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
