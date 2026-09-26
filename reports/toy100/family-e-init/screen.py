"""CPU screen for family E orthogonal inits.

Logs one JSON line per finished job to ``/tmp/family-e-init/screen.log``.
Tail that file. Plans:

  time        one K3P baseline ring run, for a wall-clock reading
  det         init-only determinism (offset 0 twice, and offset 101)
  calibrate   K3P random init, ring + unequal mass, 8 sample-seed offsets
  screen      every family-E variant, same two gates and 8 offsets
  prio        8 priority gates at offset 0 for the names passed on the command line

Usage: python -u screen.py PLAN [jobs] [variant ...]
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/workspace")
import torch

ROOT = Path("/workspace")
K3P = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p"
SHIM = Path(__file__).resolve().parent / "shim.py"
OUT = Path("/tmp/family-e-init")
LOG = OUT / "screen.log"
OFFSETS = (0, 101, 202, 303, 404, 505, 606, 707)
EARLY = ("mode_hold", "vector_unequal_mass")
PRIO = (
    ("probe", "mode_hold"),
    ("hold", "mode_hold"),
    ("probe", "vector_unequal_mass"),
    ("probe", "img_stripes2"),
    ("probe", "img_blobs4"),
    ("native", "grid100"),
    ("native", "rotated100"),
    ("shift", "mode_hold"),
)


def variants():
    from particlegan.family_e_init import FAMILIES, BIASES
    # Frobenius scale matches std bitwise on these Linear hosts, so it is not a second screen.
    return [f"{fam}_{bias}_pq" for fam in FAMILIES for bias in BIASES]


_LOCK = threading.Lock()


def _line(row):
    text = json.dumps(row)
    print(text, flush=True)
    with _LOCK:
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def _env(offset, dump):
    env = dict(os.environ)
    env.update({
        "K3P_SEED_OFFSET": str(offset),
        "K3P_INIT_DUMP": str(dump),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    env.pop("PYTHONPATH", None)
    return env


def _cmd(kind, task, output, init, init_only):
    if kind == "hold":
        script = K3P / "hold.py"
        extra = ["--config", str(K3P / "config.json"), "--backend", "cpu",
                 "--network-floor", "0.01", "--prior-floor", "0.05"]
    elif kind == "shift":
        script = K3P / "shift.py"
        extra = ["--config", str(K3P / "config.json"), "--backend", "cpu",
                 "--network-floor", "0.01", "--prior-floor", "0.05"]
    elif kind == "native":
        script = K3P / "native100.py"
        extra = ["--candidate", str(K3P)]
    else:
        script = K3P / "probe.py"
        extra = ["--config", str(K3P / "config.json"), "--backend", "cpu"]
    cmd = [sys.executable, "-u", str(SHIM), str(script), "--repo", str(ROOT), "--task", task,
           "--output", str(output), *extra]
    if init:
        cmd += ["--init", init]
    if init_only:
        cmd += ["--init-only"]
    if kind == "native":
        # native100 has no --config; the shim still wraps seeds.
        cmd = [c for c in cmd if c != "--config"]
    return cmd


def _tensor_hash(path: Path) -> str:
    if not path.exists():
        return ""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()

    def walk(value):
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                walk(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(blob)
    return digest.hexdigest()


def _metrics(payload):
    verdict = payload.get("verdict") or {}
    metrics = {row.get("metric"): row.get("value") for row in verdict.get("metrics") or []}
    conv = verdict.get("convergence") or {}
    return {
        "status": payload.get("status"),
        "modes": metrics.get("modes"),
        "hq": metrics.get("hq"),
        "mmr": metrics.get("min_mass_ratio"),
        "eigen": metrics.get("component_min_eigen_ratio"),
        "cov": metrics.get("component_covariance_error"),
        "suffix": conv.get("passing_suffix"),
        "seconds": None if payload.get("seconds") is None else round(payload["seconds"], 1),
    }


def _done(path: Path) -> bool:
    result = path / "result.json"
    if not result.exists():
        return False
    try:
        status = json.loads(result.read_text()).get("status")
    except json.JSONDecodeError:
        return False
    return status in {"PASS", "FAIL", "INCOMPLETE", "INITIALIZATION_CAPTURED"}


def _run(job):
    kind, task, init, offset, init_only, label = job
    folder = OUT / "out" / (init or "k3p") / label
    if _done(folder):
        _line({"event": "skip", "job": f"{init or 'k3p'}/{label}"})
        return
    if folder.exists():
        subprocess.run(["rm", "-rf", str(folder)], check=False)
    folder.parent.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "logs" / f"{init or 'k3p'}-{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        rc = subprocess.call(_cmd(kind, task, folder, init, init_only), cwd=ROOT,
                             stdout=handle, stderr=subprocess.STDOUT, env=_env(offset, folder))
    row = {"event": "done", "job": f"{init or 'k3p'}/{label}", "rc": rc,
           "seconds": round(time.time() - started, 1)}
    result = folder / "result.json"
    if result.exists():
        payload = json.loads(result.read_text())
        row.update(_metrics(payload))
    else:
        row["status"] = "MISSING"
    init_log = folder / "family-e-init.json"
    if init_log.exists():
        row["init_sha"] = json.loads(init_log.read_text())["all_params_sha256"][:12]
    _line(row)
    _leaderboard()


def _leaderboard():
    rows = []
    root = OUT / "out"
    if not root.exists():
        return
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    for name in names:
        cells = []
        passes = {task: [0, 0] for task in EARLY}
        for offset in OFFSETS:
            for task in EARLY:
                path = root / name / f"s{offset}-{task}" / "result.json"
                if not path.exists():
                    cells.append(".")
                    continue
                payload = json.loads(path.read_text())
                status = payload.get("status")
                passes[task][1] += 1
                passes[task][0] += status == "PASS"
                metrics = _metrics(payload)
                mark = "P" if status == "PASS" else "F"
                if task == "mode_hold" and metrics["modes"] is not None:
                    mark += str(metrics["modes"])
                cells.append(mark)
        if passes["mode_hold"][1] or passes["vector_unequal_mass"][1]:
            rows.append((passes["mode_hold"][0], passes["vector_unequal_mass"][0], name, passes, cells))
    rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
    lines = ["| variant | ring | unequal | cells |", "|---|---:|---:|---|"]
    for ring, unequal, name, passes, cells in rows:
        lines.append(f"| `{name}` | {ring}/{passes['mode_hold'][1]} | {unequal}/{passes['vector_unequal_mass'][1]} | {' '.join(cells)} |")
    text = "\n".join(lines) + "\n"
    with _LOCK:
        (OUT / "LEADERBOARD.md").write_text(text)


def _jobs(plan, selected):
    chosen = selected or variants()
    jobs = []
    if plan == "time":
        jobs.append(("probe", "mode_hold", None, 0, False, "s0-mode_hold"))
    elif plan == "det":
        for name in chosen:
            jobs.append(("probe", "mode_hold", name, 0, True, "init-s0"))
            jobs.append(("probe", "mode_hold", name, 0, True, "init-s0b"))
            jobs.append(("probe", "mode_hold", name, 101, True, "init-s101"))
    elif plan == "calibrate":
        for offset in OFFSETS:
            for task in EARLY:
                jobs.append(("probe", task, None, offset, False, f"s{offset}-{task}"))
    elif plan == "screen":
        for name in chosen:
            for offset in OFFSETS:
                for task in EARLY:
                    jobs.append(("probe", task, name, offset, False, f"s{offset}-{task}"))
    elif plan == "prio":
        for name in chosen:
            for kind, task in PRIO:
                jobs.append((kind, task, name, 0, False, f"prio-{kind}-{task}"))
    else:
        raise SystemExit(f"unknown plan {plan}")
    return jobs


def _check_det(selected):
    chosen = selected or variants()
    bad = []
    for name in chosen:
        root = OUT / "out" / name
        hashes = []
        files = []
        for label in ("init-s0", "init-s0b", "init-s101"):
            path = root / label / "family-e-init.json"
            values = root / label / "initial-values.pt"
            if not path.exists():
                bad.append((name, label, "missing"))
                continue
            hashes.append(json.loads(path.read_text())["all_params_sha256"])
            files.append(_tensor_hash(values))
        if len(hashes) == 3 and (len(set(hashes)) != 1 or len(set(files)) != 1):
            bad.append((name, "mismatch", hashes))
    report = {"event": "determinism", "checked": len(chosen), "bad": [
        {"name": item[0], "detail": item[1]} for item in bad]}
    _line(report)
    (OUT / "determinism.json").write_text(json.dumps(report, indent=1) + "\n")
    return not bad


def main():
    plan = sys.argv[1]
    jobs_n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    selected = [arg for arg in sys.argv[3 if len(sys.argv) > 2 and sys.argv[2].isdigit() else 2:] if not arg.isdigit()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    _line({"event": "start", "plan": plan, "jobs": jobs_n, "torch": torch.__version__,
           "torch_cuda": torch.version.cuda, "torch_git": getattr(torch.version, "git_version", None),
           "cuda": torch.cuda.is_available(), "tail": str(LOG)})
    todo = _jobs(plan, selected)
    with ThreadPoolExecutor(jobs_n) as pool:
        list(pool.map(_run, todo))
    if plan == "det":
        ok = _check_det(selected)
        if not ok:
            raise SystemExit(1)
    _leaderboard()
    _line({"event": "plan_done", "plan": plan})


if __name__ == "__main__":
    main()
