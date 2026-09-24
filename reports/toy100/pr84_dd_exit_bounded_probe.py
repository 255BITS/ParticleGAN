"""Fail-fast gates for DD-exit projection on the stall-reach bounded G step.

Order, seed 0, one CPU thread, neural host:
  1. Warm 1001-1200 versus the PR84 AVX2 pin (196/200, min modes 8). Regress kills.
  2. Cold trajectory, then cold ring. A keep needs the full 8-mode ring.
  3. Continued stay 1210-2400 only if the ring was acquired and the bounded
     step actually carried a negative high-D directional derivative.
     If clipped steps never exit, that is the mutual-exclusivity kill.

``ATEN_CPU_CAPABILITY`` selects the build. The primary pin is avx2.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

METHOD = "reachstall_ddexit"
PIN_WARM_PASSING = 196
PIN_WARM_MIN_MODES = 8


def emit(**row):
    print(json.dumps(row, default=float), flush=True)


def _dynamics(path):
    summary = json.loads(path.read_text())
    return summary


def _counts(recorder):
    receipt = recorder.receipt()
    keys = (
        "projected_updates", "negative_dd_updates", "clipped_exit_updates",
        "legacy_full_step_exit_updates", "curv_factor_min", "curv_factor_mean",
        "first_clipped_update", "first_clipped_curv_factor", "first_clipped_mean_dd",
        "g_curvature_bound", "d_curvature_bound", "outer_steps",
    )
    return {key: receipt.get(key) for key in keys}


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")


def run_warm(output):
    from reports.toy100.gan_followup_probe import warm
    import torch
    torch.set_num_threads(1)
    dest = output / "warm"
    if dest.exists():
        raise RuntimeError(f"{dest} already exists")
    warm(dest, METHOD)
    summary = _dynamics(dest / "summary.json")
    variant = json.loads((dest / "forks" / f"{METHOD}.json").read_text())
    receipt = variant.get("dynamics_receipt") or {}
    row = dict(
        identity_passing=summary["identity"]["local"]["passing_checks"],
        identity_min_modes=summary["identity"]["local"]["min_modes"],
        passing=summary[METHOD]["local"]["passing_checks"],
        checks=summary[METHOD]["local"]["checks"],
        min_modes=summary[METHOD]["local"]["min_modes"],
        min_hq=summary[METHOD]["local"]["min_hq"],
        final=summary[METHOD]["final"],
        projected_updates=receipt.get("projected_updates"),
        negative_dd_updates=receipt.get("negative_dd_updates"),
        clipped_exit_updates=receipt.get("clipped_exit_updates"),
        legacy_full_step_exit_updates=receipt.get("legacy_full_step_exit_updates"),
        curv_factor_min=receipt.get("curv_factor_min"),
        g_curvature_bound=receipt.get("g_curvature_bound"),
        d_curvature_bound=receipt.get("d_curvature_bound"),
    )
    emit(event="WARM_GATE", **row)
    return row


def run_cold(output, task):
    from reports.toy100.gan_followup_probe import cold
    import torch
    torch.set_num_threads(1)
    folder = {"trajectory": "cold-trajectory", "mode_hold": "cold-ring"}[task]
    dest = output / folder
    if dest.exists():
        raise RuntimeError(f"{dest} already exists")
    cold(dest, METHOD, [task])
    payload = json.loads((dest / f"{task}.json").read_text())
    live = (payload.get("result") or {}).get("live") or {}
    obs = (payload.get("result") or {}).get("observations") or []
    row = dict(
        task=task, passed=bool(payload["verdict"]["passed"]),
        status=payload["verdict"]["status"],
        live_modes=live.get("modes"), live_hq=live.get("hq"),
        tail=[{k: r.get(k) for k in ("step", "modes", "hq", "mse") if k in r} for r in obs[-5:]],
        **_counts_from_dynamics(payload.get("dynamics") or {}),
    )
    emit(event="COLD_GATE", **{k: v for k, v in row.items() if k != "tail"}, tail=row["tail"])
    return row


def _counts_from_dynamics(dynamics):
    keys = (
        "projected_updates", "negative_dd_updates", "clipped_exit_updates",
        "legacy_full_step_exit_updates", "curv_factor_min", "curv_factor_mean",
        "first_clipped_update", "first_clipped_curv_factor", "first_clipped_mean_dd",
        "g_curvature_bound", "d_curvature_bound",
    )
    return {key: dynamics.get(key) for key in keys}


def run_stay(output, steps):
    from reports.toy100.gan_followup_probe import stay
    import torch
    torch.set_num_threads(1)
    dest = output / "stay"
    if dest.exists():
        raise RuntimeError(f"{dest} already exists")
    stay(dest, METHOD, steps)
    payload = json.loads((dest / "stay.json").read_text())
    summary = payload["summary"]
    row = dict(
        checks=summary["checks"], passing=summary["passing"],
        min_modes=summary["min_modes"], min_hq=summary["min_hq"],
        final=summary["final"], failing_steps=summary["failing_steps"],
        **_counts_from_dynamics(summary.get("dynamics") or {}),
    )
    emit(event="STAY_GATE", **{k: v for k, v in row.items() if k != "failing_steps"},
         n_failing=len(row["failing_steps"]))
    return row


def run_avx512_ring(output):
    env = os.environ.copy()
    env["ATEN_CPU_CAPABILITY"] = "avx512"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    dest = output / "cold-ring-avx512"
    cmd = [sys.executable, "-m", "reports.toy100.pr84_dd_exit_bounded_probe",
           "--phase", "cold-ring", "--output", str(dest)]
    emit(event="AVX512_START", cmd=" ".join(cmd[-6:]))
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
    if proc.returncode != 0:
        emit(event="AVX512_FAIL", returncode=proc.returncode)
        return dict(passed=False, status="error", returncode=proc.returncode)
    decision = json.loads((dest / "decision.json").read_text())
    emit(event="AVX512_GATE", **decision.get("cold_ring", {}))
    return decision.get("cold_ring")


def _exclusive(rows):
    clipped = sum(int(row.get("clipped_exit_updates") or 0) for row in rows)
    negative = sum(int(row.get("negative_dd_updates") or 0) for row in rows)
    legacy = sum(int(row.get("legacy_full_step_exit_updates") or 0) for row in rows)
    return dict(clipped_exit_updates=clipped, negative_dd_updates=negative,
                legacy_full_step_exit_updates=legacy,
                mutually_exclusive=clipped == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "warm", "cold-trajectory", "cold-ring", "stay"),
                        default="all")
    parser.add_argument("--steps", type=int, default=2400)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args.output.mkdir(parents=True, exist_ok=True)
    decision = dict(
        mechanism="DD-exit shrink of the curvature-bounded stall-reach G step",
        purity="GAN dynamics only",
        host="neural", seed=0, method=METHOD,
        g_curvature_bound=0.25, d_curvature_bound=3.0, ramp="stall",
        pin="PR84 AVX2 warm 196/200 min modes 8; cold ring 8; stay 53/120 final 6",
        stall_reach="warm 200/200; cold ring 8; stay 97/120 final 8",
    )
    emit(event="DECLARED", phase=args.phase, **{k: decision[k] for k in (
        "mechanism", "seed", "method", "g_curvature_bound")})

    def finish(verdict, reason):
        decision["verdict"] = verdict
        decision["reason"] = reason
        _write(args.output / "decision.json", decision)
        emit(event="DONE", verdict=verdict, reason=reason)
        return 0

    if args.phase in ("all", "warm"):
        decision["warm"] = run_warm(args.output)
        warm = decision["warm"]
        if warm["identity_passing"] != 200 or warm["identity_min_modes"] != 8:
            return finish("ABORT", "warm identity is not the AVX2 8-mode pin; comparison is invalid")
        if (warm["passing"] < PIN_WARM_PASSING or warm["min_modes"] < PIN_WARM_MIN_MODES
                or warm["g_curvature_bound"] != 0.25 or warm["d_curvature_bound"] != 3.0):
            return finish("KILL", "warm regress versus PR84 pin, or a curvature bound moved")
        if args.phase == "warm":
            return finish("WARM_PASS", "warm did not regress; later gates not requested")

    if args.phase in ("all", "cold-trajectory"):
        decision["cold_trajectory"] = run_cold(args.output, "trajectory")
        if not decision["cold_trajectory"]["passed"]:
            return finish("KILL", "cold trajectory regressed")
        if args.phase == "cold-trajectory":
            return finish("TRAJ_PASS", "cold trajectory passed; ring not requested")

    if args.phase in ("all", "cold-ring"):
        decision["cold_ring"] = run_cold(args.output, "mode_hold")
        ring = decision["cold_ring"]
        if not ring["passed"] or ring.get("live_modes") != 8:
            if args.phase == "all":
                decision["cold_ring_avx512"] = run_avx512_ring(args.output)
            return finish("KILL", "cold ring did not acquire the full 8-mode target")
        if args.phase == "cold-ring":
            return finish("RING_PASS", "cold ring acquired 8 modes")

    if args.phase == "all":
        counted = [decision[name] for name in ("warm", "cold_trajectory", "cold_ring")]
        decision["exclusivity"] = _exclusive(counted)
        emit(event="EXCLUSIVITY", **decision["exclusivity"])
        if decision["exclusivity"]["mutually_exclusive"]:
            return finish(
                "KILL",
                "bounded steps never had a negative high-D directional derivative; "
                "DD exit and the curvature bound are still mutually exclusive")
        decision["cold_ring_avx512"] = run_avx512_ring(args.output)

    if args.phase in ("all", "stay"):
        decision["stay"] = run_stay(args.output, args.steps)
        final = decision["stay"]["final"]
        final_modes = None if not final else final[1]
        final_hq = None if not final else final[2]
        if final_modes != 8:
            return finish("DEMOTE", "continued run does not end on the full ring")
        if final_hq is None or final_hq < .9:
            return finish("DEMOTE", "continued run ends on 8 modes below the HQ bar")
        return finish("KEEP", "full ring acquired and the continued run ends on it")

    return finish(decision.get("verdict", "SCORED"), decision.get("reason", "phase complete"))


if __name__ == "__main__":
    sys.exit(main())
