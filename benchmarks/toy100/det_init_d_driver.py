"""Family D CPU screen. One JSON line per finished job, easy to tail.

    python -u -m benchmarks.toy100.det_init_d_driver --phase baseline
    python -u -m benchmarks.toy100.det_init_d_driver --phase determinism
    python -u -m benchmarks.toy100.det_init_d_driver --phase screen
    python -u -m benchmarks.toy100.det_init_d_driver --phase priority --variants hq_pb_pq,mix_pb_pq

Logs: $K3P_DET_ROOT/screen.jsonl and $K3P_DET_ROOT/logs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SCREEN = ROOT / "benchmarks/toy100/det_init_screen.py"
OUT = Path(os.environ.get("K3P_DET_ROOT", "/tmp/k3p-det-d"))
JSONL = OUT / "screen.jsonl"
OFFSETS = (0, 101, 202, 303, 404, 505, 606, 707)
SCREEN_GATES = ("ring", "unequal")
PRIORITY = ("ring", "hold", "unequal", "stripes", "blobs", "grid100", "rotated100", "shift")


def _variants():
    sys.path.insert(0, str(ROOT))
    from particlegan.det_init import VARIANTS
    return VARIANTS


def _finished(path: Path) -> bool:
    result = path / "result.json"
    if result.exists():
        status = json.loads(result.read_text()).get("status")
        if status not in (None, "ERROR"):
            return True
    if (path / "gate-grid100.json").exists() or (path / "gate-rotated100.json").exists():
        return True
    return False


def _row(variant: str, offset: int, gate: str, proc: subprocess.CompletedProcess, seconds: float) -> dict:
    text = (proc.stdout or "").strip().splitlines()
    payload = {}
    if text:
        try:
            payload = json.loads(text[-1])
        except json.JSONDecodeError:
            payload = {"status": "ERROR", "raw": text[-1][:400]}
    payload.update(variant=variant, seed_offset=offset, gate=gate,
                   wall_seconds=round(seconds, 3), driver_rc=proc.returncode)
    return payload


def _launch(job):
    variant, offset, gate, init_only = job
    name = "baseline" if variant is None else variant
    folder = OUT / "out" / name / f"s{offset}" / ("init-" + gate if init_only else gate)
    log = OUT / "logs" / f"{name}-s{offset}-{gate}{'-init' if init_only else ''}.log"
    if _finished(folder) and not init_only:
        return {"variant": name, "seed_offset": offset, "gate": gate, "status": "SKIP"}
    if init_only and (folder / "initial-values.pt").exists() and (folder / "result.json").exists():
        return {"variant": name, "seed_offset": offset, "gate": gate, "status": "SKIP", "init_only": True}
    if folder.exists():
        subprocess.run(["rm", "-rf", str(folder)], check=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(SCREEN), "--gate", gate, "--output", str(folder),
           "--log", str(log), "--seed-offset", str(offset)]
    if variant:
        cmd.extend(["--init", variant])
    if init_only:
        cmd.append("--init-only")
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    row = _row(name, offset, gate, proc, time.perf_counter() - started)
    if init_only:
        row["init_only"] = True
    with JSONL.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def _jobs(phase: str, variants):
    if phase == "baseline":
        return [(None, offset, gate, False) for offset in OFFSETS for gate in SCREEN_GATES]
    if phase == "determinism":
        return [(name, offset, gate, True) for name in variants for gate in SCREEN_GATES for offset in (0, 101)]
    if phase == "screen":
        return [(name, offset, gate, False) for name in variants for offset in OFFSETS for gate in SCREEN_GATES]
    raise ValueError(phase)


def _tensor_sha(path: Path) -> str:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()

    def eat(value):
        if torch.is_tensor(value):
            raw = value.detach().cpu().contiguous()
            digest.update(str(tuple(raw.shape)).encode())
            digest.update(raw.numpy().tobytes())
        elif isinstance(value, (list, tuple)):
            for item in value:
                eat(item)

    eat(blob)
    return digest.hexdigest()


def _check_determinism(variants):
    report = {}
    for name in variants:
        per_gate = {}
        for gate in SCREEN_GATES:
            left = OUT / "out" / name / "s0" / f"init-{gate}" / "initial-values.pt"
            right = OUT / "out" / name / "s101" / f"init-{gate}" / "initial-values.pt"
            if not left.exists() or not right.exists():
                per_gate[gate] = {"match": False, "error": "missing checkpoint"}
                continue
            sha_left, sha_right = _tensor_sha(left), _tensor_sha(right)
            per_gate[gate] = {"match": sha_left == sha_right, "sha256": sha_left, "sha_s101": sha_right}
        report[name] = per_gate
    path = OUT / "determinism.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    rows = [gate for per_gate in report.values() for gate in per_gate.values()]
    print(json.dumps({"event": "determinism", "pass": sum(row["match"] for row in rows),
                      "n": len(rows), "path": str(path)}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("baseline", "determinism", "screen", "priority"))
    parser.add_argument("--variants", default=None, help="comma list; default is every family D flag")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("K3P_DET_JOBS", "3")))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    names = tuple(args.variants.split(",")) if args.variants else _variants()
    header = {"event": "build", "torch": torch.__version__, "torch_git": getattr(torch.version, "git_version", None),
              "cuda": torch.cuda.is_available(), "phase": args.phase, "variants": list(names)}
    print(json.dumps(header), flush=True)
    (OUT / "build.json").write_text(json.dumps(header, indent=2) + "\n")
    if args.phase == "priority":
        jobs = [(name, 0, gate, False) for name in names for gate in PRIORITY]
    else:
        jobs = _jobs(args.phase, names)
    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        list(pool.map(_launch, jobs))
    if args.phase == "determinism":
        report = _check_determinism(names)
        rows = [gate for per_gate in report.values() for gate in per_gate.values()]
        if not all(row["match"] for row in rows):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
