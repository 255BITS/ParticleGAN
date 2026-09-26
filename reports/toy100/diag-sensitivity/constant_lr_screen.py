"""One A/B: hid_q ring, hold, and stay with the cosine anneal turned off.

Constant LR is the pre-anneal rate. ``lr_anneal_start`` is 0 and both floors
are 1, so ``policy_multipliers`` is 1 at every step. The horizon cap stays.
Nothing else in the recipe changes. Default probes are not modified.

    python -u reports/toy100/diag-sensitivity/constant_lr_screen.py
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
K3P = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p"
OFFSETS = (0, 101, 202, 303, 404, 505, 606, 707)
GATES = ("ring", "hold", "shift")


def constant_config(source: Path, dest: Path) -> dict:
    config = json.loads(source.read_text())
    config["lr_anneal_start"] = 0.0
    config["lr_floor"] = 1.0
    config["network_lr_floor"] = 1.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(config, indent=2) + "\n")
    sys.path.insert(0, str(ROOT))
    from benchmarks.toy100.schedule import policy_multipliers
    for step in (0, 720, 721, 800, 1199, 4000):
        network, prior = policy_multipliers(
            step, 1200, 0.0, 1.0, config["network_lr_horizon_cap"],
            network_lr_floor=1.0,
        )
        if network != 1.0 or prior != 1.0:
            raise SystemExit(f"constant schedule is not 1 at step {step}: {network}, {prior}")
    return config


def command(gate: str, config: Path, output: Path) -> list[str]:
    common = ["--repo", str(ROOT), "--config", str(config), "--backend", "cpu",
              "--init", "hid_q", "--output", str(output)]
    if gate == "ring":
        return [sys.executable, "-u", str(K3P / "probe.py"), *common, "--task", "mode_hold"]
    script = K3P / ("hold.py" if gate == "hold" else "shift.py")
    return [sys.executable, "-u", str(script), *common, "--task", "mode_hold",
            "--network-floor", "1", "--prior-floor", "1", "--anneal-start", "0"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", nargs="+", default=list(GATES), choices=GATES)
    parser.add_argument("--offsets", nargs="+", type=int, default=list(OFFSETS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--root", type=Path, default=Path("/tmp/k3p-diag/constant-lr"))
    args = parser.parse_args()
    config_path = args.root / "config.json"
    config = constant_config(K3P / "config.json", config_path)
    print(json.dumps({
        "event": "constant_lr",
        "lr": config["lr"],
        "prior_lr": config["lr"] * config["prior_lr_mult"],
        "lr_anneal_start": config["lr_anneal_start"],
        "lr_floor": config["lr_floor"],
        "network_lr_floor": config["network_lr_floor"],
        "network_lr_horizon_cap": config["network_lr_horizon_cap"],
    }), flush=True)
    pending = [(gate, offset) for gate in args.gates for offset in args.offsets]
    print(f"queued constant_lr {len(pending)}", flush=True)
    running = []
    index = 0
    while index < len(pending) or running:
        while index < len(pending) and len(running) < args.jobs:
            gate, offset = pending[index]
            index += 1
            out = args.root / f"{gate}-s{offset}"
            log = args.root / "logs" / f"{gate}-s{offset}.log"
            if out.exists():
                raise SystemExit(f"output exists: {out}")
            log.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
            cmd = command(gate, config_path, out)
            if offset:
                env["K3P_SEED_OFFSET"] = str(offset)
                shim = ROOT / "benchmarks/toy100/det_init_seedshim.py"
                cmd = [sys.executable, "-u", str(shim), *cmd[2:]]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log.open("w"),
                                    stderr=subprocess.STDOUT)
            running.append((gate, offset, proc, time.time(), log))
            print(f"START constant_lr {gate} s{offset}", flush=True)
        still = []
        for gate, offset, proc, t0, log in running:
            if proc.poll() is None:
                still.append((gate, offset, proc, t0, log))
                continue
            print(f"DONE constant_lr {gate} s{offset} rc={proc.returncode} "
                  f"dt={time.time() - t0:.1f}s", flush=True)
        running = still
        if running:
            time.sleep(0.4)
    print("ALL_CONSTANT_LR_DONE", flush=True)


if __name__ == "__main__":
    main()
