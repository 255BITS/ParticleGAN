"""Screen one relief over the 8 hid_q offsets and four gates.

CPU screening only. Each child is ``det_init_screen`` with ``K3P_RELIEF`` set,
so the default path is unchanged unless that variable is present.

    python -u reports/toy100/diag-sensitivity/screen.py --relief optimistic
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OFFSETS = (0, 101, 202, 303, 404, 505, 606, 707)
GATES = ("ring", "hold", "shift", "unequal")
RELIEFS = ("optimistic", "extragradient", "ema_fake", "k3p_pull", "row_damp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relief", required=True, choices=RELIEFS)
    parser.add_argument("--gates", nargs="+", default=list(GATES), choices=GATES)
    parser.add_argument("--offsets", nargs="+", type=int, default=list(OFFSETS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--root", type=Path, default=Path("/tmp/k3p-diag/relief"))
    args = parser.parse_args()
    pending = [(gate, offset) for gate in args.gates for offset in args.offsets]
    print(f"queued {args.relief} {len(pending)}", flush=True)
    running = []
    index = 0
    while index < len(pending) or running:
        while index < len(pending) and len(running) < args.jobs:
            gate, offset = pending[index]
            index += 1
            label = f"s{offset}"
            out = args.root / args.relief / f"{gate}-{label}"
            log = args.root / "logs" / f"{args.relief}-{gate}-{label}.log"
            if out.exists():
                raise SystemExit(f"output exists: {out}")
            log.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["K3P_RELIEF"] = args.relief
            env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [sys.executable, "-u", "-m", "benchmarks.toy100.det_init_screen",
                   "--gate", gate, "--init", "hid_q", "--seed-offset", str(offset),
                   "--output", str(out), "--log", str(log)]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            running.append((gate, label, proc, time.time()))
            print(f"START {args.relief} {gate} {label}", flush=True)
        still = []
        for gate, label, proc, t0 in running:
            if proc.poll() is None:
                still.append((gate, label, proc, t0))
                continue
            text = (proc.stdout.read() or "").strip().splitlines()
            last = text[-1] if text else ""
            print(f"DONE {args.relief} {gate} {label} rc={proc.returncode} "
                  f"dt={time.time() - t0:.1f}s {last}", flush=True)
        running = still
        if running:
            time.sleep(0.4)
    print(f"ALL_{args.relief.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()
