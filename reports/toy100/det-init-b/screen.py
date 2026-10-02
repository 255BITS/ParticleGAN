"""Run one K3P CPU gate and print a one-line summary.

    python -u reports/toy100/det-init-b/screen.py --variant baseline --gate ring --output /tmp/out/ring
    python -u reports/toy100/det-init-b/screen.py --variant dct_frob --gate unequal --seed-offset 101 --output ...

variant ``baseline`` leaves PyTorch init in place. Other names are ``--init`` choices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
K3P = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p"
GATES = ("ring", "unequal", "hold", "stay", "grid100", "rotated100", "stripes", "blobs")


def command(variant, gate, output, seed_offset, init_only):
    init = [] if variant == "baseline" else ["--init", variant]
    offset = [] if not seed_offset else ["--seed-offset", str(seed_offset)]
    if gate == "ring":
        cmd = [sys.executable, "-u", str(K3P / "probe.py"), "--repo", str(ROOT), "--config", str(K3P / "config.json"),
               "--task", "mode_hold", "--backend", "cpu", "--output", str(output)]
    elif gate == "unequal":
        cmd = [sys.executable, "-u", str(K3P / "probe.py"), "--repo", str(ROOT), "--config", str(K3P / "config.json"),
               "--task", "vector_unequal_mass", "--backend", "cpu", "--output", str(output)]
    elif gate == "stripes":
        cmd = [sys.executable, "-u", str(K3P / "probe.py"), "--repo", str(ROOT), "--config", str(K3P / "config.json"),
               "--task", "img_stripes2", "--backend", "cpu", "--output", str(output)]
    elif gate == "blobs":
        cmd = [sys.executable, "-u", str(K3P / "probe.py"), "--repo", str(ROOT), "--config", str(K3P / "config.json"),
               "--task", "img_blobs4", "--backend", "cpu", "--output", str(output)]
    elif gate == "hold":
        cmd = [sys.executable, "-u", str(K3P / "hold.py"), "--repo", str(ROOT), "--config", str(K3P / "config.json"),
               "--task", "mode_hold", "--backend", "cpu", "--network-floor", "0.01", "--prior-floor", "0.05",
               "--output", str(output)]
    elif gate == "stay":
        cmd = [sys.executable, "-u", str(K3P / "shift.py"), "--repo", str(ROOT), "--config", str(K3P / "config.json"),
               "--task", "mode_hold", "--backend", "cpu", "--network-floor", "0.01", "--prior-floor", "0.05",
               "--output", str(output)]
    elif gate in ("grid100", "rotated100"):
        cmd = [sys.executable, "-u", str(K3P / "native100.py"), "--repo", str(ROOT), "--candidate", str(K3P),
               "--task", gate, "--backend", "cpu", "--output", str(output)]
    else:
        raise SystemExit(f"unknown gate {gate}")
    if init_only:
        if gate in ("hold", "stay", "grid100", "rotated100"):
            raise SystemExit("--init-only is only wired on probe gates")
        cmd.append("--init-only")
    return cmd + init + offset


def _dig(record, *path):
    value = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def summarize(output: Path) -> dict:
    path = output / "result.json"
    if not path.exists():
        return {"status": "MISSING"}
    record = json.loads(path.read_text())
    inits = []
    for optimizer in _dig(record, "proof", "initial_optimizers") or []:
        for row in optimizer:
            inits.append(row.get("sha256", ""))
    init_sha = hashlib.sha256("".join(inits).encode()).hexdigest() if inits else None
    live = _dig(record, "result", "live") or {}
    convergence = _dig(record, "verdict", "convergence") or _dig(record, "result", "convergence") or {}
    coverage = _dig(record, "coverage", "status")
    accuracy = _dig(record, "accuracy", "status")
    return {
        "status": record.get("status"),
        "seconds": None if record.get("seconds") is None else round(record["seconds"], 1),
        "modes": live.get("modes", _dig(record, "last_live", "modes")),
        "hq": live.get("hq", _dig(record, "last_live", "hq")),
        "suffix": convergence.get("passing_suffix"),
        "min_mass_ratio": live.get("min_mass_ratio"),
        "hold": _dig(record, "gate", "status"),
        "hold_checks": _dig(record, "gate", "hold_checks"),
        "stay": _dig(record, "continued_hold", "passing_checks"),
        "stay_checks": _dig(record, "continued_hold", "checks"),
        "coverage": coverage,
        "accuracy": accuracy,
        "init_sha256": init_sha,
        "error": (record.get("error") or "")[:240] or None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gate", required=True, choices=GATES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    env.pop("LD_PRELOAD", None)
    cmd = command(args.variant, args.gate, args.output, args.seed_offset, args.init_only)
    print(json.dumps({"event": "START", "variant": args.variant, "gate": args.gate,
                      "seed_offset": args.seed_offset, "init_only": args.init_only}), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT, env=env)
    row = summarize(args.output)
    row.update(event="SUMMARY", variant=args.variant, gate=args.gate, seed_offset=args.seed_offset, rc=rc)
    print(json.dumps(row), flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
