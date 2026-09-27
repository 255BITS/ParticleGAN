"""Run no-R1 arms (arms.json here) on the k3p-constant-suite tasks; one task per subprocess, a shared job pool.

The worker is the suite's own ``run_suite.worker`` (unchanged) with ``suite_adapter.install`` swapped for
``nr_adapter.install``; after it returns, the task's nr receipt is written to <out>/nr_receipt.json and merged into
<out>/result.json as ``nr_receipt``.

Usage:
  run_nr.py --arms nr_none_none,nr_symcap_hinge --device cuda:1 --jobs 6 --tasks screen
  run_nr.py --arms all --device cuda:0 --jobs 6 --tasks native-grid100,ring8-shift [--steps N --runs-dir runs_smoke]
--tasks: 'screen' (7 screen tasks), 'all' (26), or a comma list. CPU hosts (custom-loop toys) ignore --device.
Output: runs/<arm>/<task>/{result.json,nr_receipt.json}; logs/<arm>.log (one line per task);
        logs/<arm>/<task>.log (one line per eval). For a non-default --runs-dir the logs go to logs_<runs-dir name>/.
  tail -f logs/*.log            # task completions
  tail -f logs/<arm>/*.log      # per-eval lines
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent / "k3p-constant-suite"
ROOT = HERE.parents[1]
PY = Path("/home/martyn/dev/ParticleGAN/.venv/bin/python")
sys.path[:0] = [str(HERE), str(SUITE), str(ROOT)]

import run_suite as rs  # noqa: E402

SCREEN = ["native-grid100", "native-rotated100", "native-staggered100", "toy-img_bars4", "toy-two_pole",
          "ring8-shift", "ring8-multishift"]
ARMS = json.loads((HERE / "arms.json").read_text())["arms"]
if (HERE / "arms_e1.json").exists():
    ARMS.update(json.loads((HERE / "arms_e1.json").read_text())["arms"])


def worker(args):
    import nr_adapter
    rs_sa = sys.modules["suite_adapter"]
    rs_sa.install = nr_adapter.install
    rs.worker(args)
    out = Path(args.out)
    cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    rec = nr_adapter.receipt(cfg)
    (out / "nr_receipt.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    res_path = out / "result.json"
    if res_path.exists():
        res = json.loads(res_path.read_text())
        res["nr_receipt"] = rec
        res_path.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"# NR receipt spike={rec['spike']} settle={rec['settle']} r1={rec['r1_weight_applied']} "
          f"loss={rec['d_loss_calls']} oadam_steps={rec['oadam_steps']} anchor_active={rec['anchor_active']} "
          f"penalty_calls={rec['nr_penalty_calls']} stock_calls={rec['stock_k3p_penalty_calls']}", flush=True)


def resolve(args):
    arms = list(ARMS) if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}")
    tasks = (SCREEN if args.tasks == "screen" else list(rs.TASKS) if args.tasks == "all"
             else [t.strip() for t in args.tasks.split(",")])
    bad = [t for t in tasks if t not in rs.ORDER]
    if bad:
        raise SystemExit(f"unknown tasks {bad}; choose from {rs.TASKS}")
    tasks.sort(key=rs.ORDER.get)
    return arms, tasks


def launch(args):
    arms, tasks = resolve(args)
    runs_root = Path(args.runs_dir).resolve()
    log_root = HERE / "logs" if runs_root == (HERE / "runs").resolve() else HERE / f"logs_{runs_root.name}"
    locks, arm_logs, left = [], {}, {}
    for arm in arms:
        (runs_root / arm).mkdir(parents=True, exist_ok=True)
        lock = (runs_root / arm / ".launcher.lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another launcher is already running arm {arm} into {runs_root}")
        locks.append(lock)
        (log_root / arm).mkdir(parents=True, exist_ok=True)
        arm_logs[arm] = (log_root / f"{arm}.log").open("a", buffering=1)
        arm_logs[arm].write(f"# {time.strftime('%F %T')} arm={arm} device={args.device} tasks={len(tasks)} "
                            f"steps={args.steps}\n")
        left[arm] = len(tasks)
    gpu = args.device.split(":")[-1] if ":" in args.device else "0"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    env.pop("K3P_INIT", None)
    # task-major order: the long native/ring tasks of every arm start first
    pending = [(arm, task) for task in tasks for arm in arms]
    active = []

    def finish_arm(arm):
        left[arm] -= 1
        if left[arm] == 0:
            arm_logs[arm].write(f"# {time.strftime('%F %T')} arm={arm} finished\n")

    while pending or active:
        while pending and len(active) < args.jobs:
            arm, task = pending.pop(0)
            out = runs_root / arm / task
            if (out / "result.json").exists() and not args.overwrite:
                arm_logs[arm].write(f"skip {task} (result.json exists)\n")
                finish_arm(arm)
                continue
            if out.exists():
                subprocess.run(["rm", "-rf", str(out)], check=True)
            cmd = [str(PY), "-u", str(Path(__file__).resolve()), "--worker", "--arm", arm, "--task", task,
                   "--out", str(out)] + (["--steps", str(args.steps)] if args.steps else [])
            fh = (log_root / arm / f"{task}.log").open("w")
            active.append((arm, task, out, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                                            cwd=ROOT), fh, time.monotonic()))
        time.sleep(2)
        for item in list(active):
            arm, task, out, proc, fh, t0 = item
            if proc.poll() is None:
                continue
            fh.close()
            active.remove(item)
            r = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
            metric = r.get("metric") or {}
            arm_logs[arm].write(f"{time.strftime('%T')} {task:22s} {r.get('status', 'NO_RESULT'):5s} "
                                f"{metric.get('name')}={metric.get('value')} | {r.get('detail', '')[:150]} | "
                                f"{r.get('seconds', time.monotonic() - t0):.0f}s rc={proc.returncode}\n")
            finish_arm(arm)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", help="comma list or 'all'")
    ap.add_argument("--arm", help=argparse.SUPPRESS)
    ap.add_argument("--tasks", default="screen", help="'screen', 'all' or a comma list")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None, help="smoke only: shorten native/hold/ring budgets")
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--task", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args)
    else:
        if not args.arms:
            ap.error("--arms is required")
        launch(args)


if __name__ == "__main__":
    main()
