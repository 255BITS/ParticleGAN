"""One K3P CPU gate under an optional deterministic init.

Prints a single JSON object on stdout. Training chatter stays on stderr
when ``--quiet-driver`` is set; the driver log is always the ``--log`` file.

    python -u -m benchmarks.toy100.det_init_screen --gate ring --init eye --log /tmp/k3p-det/logs/eye-ring.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
K3P = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p"
CONFIG = K3P / "config.json"
GATES = ("ring", "unequal", "stripes", "blobs", "hold", "shift", "grid100", "rotated100")


def _command(gate: str, init: str | None, output: Path) -> list[str]:
    probe = [sys.executable, "-u", str(K3P / "probe.py"), "--repo", str(ROOT),
             "--config", str(CONFIG), "--backend", "cpu", "--output", str(output)]
    if gate == "ring":
        cmd = probe + ["--task", "mode_hold"]
    elif gate == "unequal":
        cmd = probe + ["--task", "vector_unequal_mass"]
    elif gate == "stripes":
        cmd = probe + ["--task", "img_stripes2"]
    elif gate == "blobs":
        cmd = probe + ["--task", "img_blobs4"]
    elif gate == "hold":
        cmd = [sys.executable, "-u", str(K3P / "hold.py"), "--repo", str(ROOT),
               "--config", str(CONFIG), "--task", "mode_hold", "--backend", "cpu",
               "--network-floor", "0.01", "--prior-floor", "0.05", "--output", str(output)]
    elif gate == "shift":
        cmd = [sys.executable, "-u", str(K3P / "shift.py"), "--repo", str(ROOT),
               "--config", str(CONFIG), "--task", "mode_hold", "--backend", "cpu",
               "--network-floor", "0.01", "--prior-floor", "0.05", "--output", str(output)]
    elif gate in ("grid100", "rotated100"):
        cmd = [sys.executable, "-u", "-m", "benchmarks.toy100", "run",
               "--config", str(CONFIG), "--problem", gate, "--device", "cpu",
               "--no-render", "--output", str(output)]
    else:
        raise ValueError(gate)
    if init:
        cmd.extend(["--init", init])
    return cmd


def _summary(gate: str, output: Path, returncode: int) -> dict:
    row = {"gate": gate, "returncode": returncode}
    if gate in ("grid100", "rotated100"):
        coverage = output / f"gate-{gate}.json"
        accuracy = output / f"accuracy-gate-{gate}.json"
        if coverage.exists():
            row["coverage"] = json.loads(coverage.read_text()).get("status")
        if accuracy.exists():
            row["accuracy"] = json.loads(accuracy.read_text()).get("status")
        summary = output / gate / "summary.json"
        if summary.exists():
            row["train_status"] = json.loads(summary.read_text()).get("status")
        row["status"] = "PASS" if row.get("coverage") == "PASS" and row.get("accuracy") == "PASS" else "FAIL"
        if not coverage.exists() and not accuracy.exists():
            row["status"] = "ERROR"
        return row
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
    elif gate == "shift":
        hold = payload.get("continued_hold") or {}
        row["stay"] = f"{hold.get('passing_checks')}/{hold.get('checks')}"
        row["pass_all"] = hold.get("pass_all")
    else:
        live = (payload.get("result") or {}).get("live") or {}
        keep = ("modes", "hq", "min_mass_ratio", "sw1_normalized", "mass_tv",
                "component_min_eigen_ratio", "mean_rmse")
        row["live"] = {key: live.get(key) for key in keep if key in live}
        conv = (payload.get("result") or {}).get("convergence") or {}
        if "passing_suffix" in conv:
            row["passing_suffix"] = conv.get("passing_suffix")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, choices=GATES)
    parser.add_argument("--init", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--diag", action="store_true",
                        help="write a read-only per-step trace to <output>/diag.jsonl")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    if args.seed_offset:
        env["K3P_SEED_OFFSET"] = str(args.seed_offset)
    if args.diag:
        env["K3P_DIAG_TRAJ"] = str(args.output / "diag.jsonl")
    cmd = _command(args.gate, args.init, args.output)
    if args.seed_offset:
        shim = str(Path(__file__).with_name("det_init_seedshim.py"))
        cmd = [sys.executable, "-u", shim, *cmd[2:]]
    with args.log.open("w") as log:
        done = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    row = _summary(args.gate, args.output, done.returncode)
    row.update(init=args.init or "default", seed_offset=args.seed_offset, log=str(args.log))
    print(json.dumps(row, sort_keys=True), flush=True)
    raise SystemExit(0 if row.get("status") != "ERROR" else 1)


if __name__ == "__main__":
    main()
