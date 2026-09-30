"""Run one ARM over the K3P constant-LR suite (or a --tasks subset); one task per subprocess.

Suite (26 tasks):
  native-{grid100,rotated100,staggered100}  benchmarks.toy100, 7000 updates, seed 1234: coverage AND accuracy gate
  toy-<19 transfer hosts>                    benchmarks.transfer_suite frozen hosts, own budgets/seeds: sustained live gate
  hold-mode_hold                             mode_hold host extended to 7500: first 200-check confirmation, then a
                                             1200-update hold and a 300-update extension (gap-fill protocol)
  shift-mode_hold                            mode_hold host, (1,0) shift at 2400, 3600 updates: deadline recovery
  ring8-shift                                simple-critic ring protocol (20k particles, batch 2048), (1,0) at 2400, 4600
  ring8-multishift                           same, +(1,0) at 2400, -(1,0) at 4600, +(1,0) at 6800, 9000 updates
Init: tasks use their develop entry point's init. native/toy/hold/shift: constructor init (develop makes no
particlegan.init call; --init/K3P_INIT unset). ring: particlegan.init.deterministic_orthogonal_ (G seed 0,
D seed 1, R2 prior), as develop's GANTrainer examples do.

Usage:
  run_suite.py --arm k3p_const_ams --device cuda:0 [--jobs 4] [--tasks ring8-shift,toy-two_pole] [--steps N]
Output: runs/<arm>/<task>/result.json; logs/<arm>/<task>.log (one line per eval); logs/<arm>.log (one line per task).
  tail -f logs/<arm>.log          # task completions
  tail -f logs/<arm>/*.log        # per-eval lines
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / ".venv/bin/python"
BASE_CONFIG = ROOT / "configs/toy100/constraints_simple_regularization.json"
GATE_SRC = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py"
NATIVE = ("grid100", "rotated100", "staggered100")
TRANSFER = ("two_pole", "trajectory", "residual_student", "unipolar", "ae_gan_hold", "cover_leftover",
            "unused_token_hold", "mid_scale_identity", "mode_hold", "vector_two_broad", "vector_unequal_mass",
            "vector_unequal_width", "vector_anisotropic", "vector_overlap", "vector_spiral", "img_stripes2",
            "img_bars4", "img_blobs4", "img_intensity2")
LEGACY = set(TRANSFER[:9])
# Longest first (measured K3P gap-fill seconds where known).
TASKS = ([f"native-{p}" for p in NATIVE] + ["hold-mode_hold", "ring8-multishift", "shift-mode_hold", "ring8-shift"]
         + [f"toy-{t}" for t in sorted(TRANSFER, key=lambda t: TRANSFER.index(t))])
ORDER = {t: i for i, t in enumerate(TASKS)}
CONSTRUCTOR = "constructor (develop entry point makes no particlegan.init call; --init and K3P_INIT unset)"
RING_INIT = "particlegan.init.deterministic_orthogonal_(G, seed=0), (D, seed=1), (recipe.make_prior())"


def group_of(task):
    return task.split("-", 1)[0]


# ================================================================== worker
def worker(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ.get("K3P_INIT"):
        raise SystemExit("unset K3P_INIT")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(1, str(HERE))
    import particlegan
    assert Path(particlegan.__file__).resolve().is_relative_to(ROOT), particlegan.__file__
    import suite_adapter as sa
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task, group = args.task, group_of(args.task)
    name = task.split("-", 1)[1]
    base = json.loads(BASE_CONFIG.read_text())
    sa.ARM = sa.ARMS[args.arm]
    cfg = sa.arm_config(base)
    cfg_path = out / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    route = ("ring" if group == "ring8" else "legacy" if group in ("hold", "shift") or name in LEGACY
             else "gantrainer")
    log = lambda line: print(line, flush=True)  # noqa: E731
    receipt = sa.install(args.arm, route=route, config=cfg, log=log)
    result = dict(arm=args.arm, task=task, group=group, route=route, formulation=sa.ARM["formulation"],
                  steps_override=args.steps, status="ERROR", passed=False, metric=None, detail="")
    started = time.monotonic()
    log(f"# arm={args.arm} task={task} route={route} particlegan={particlegan.__file__}")
    try:
        runner = dict(native=run_native, toy=run_transfer, hold=run_hold, shift=run_shift, ring8=run_ring)[group]
        result.update(runner(name, cfg, cfg_path, out, args, sa, log))
    except Exception:
        result["error"] = traceback.format_exc()
        result["detail"] = result["error"].strip().splitlines()[-1][:300]
        log(result["error"])
    result["seconds"] = round(time.monotonic() - started, 1)
    rec = sa.finish_receipt()
    result["init"] = dict(function=RING_INIT if route == "ring" and sa.ARM.get("init") != "constructor" else CONSTRUCTOR,
                          init_sha256=rec.get("init_sha256"), init_tensors=rec.get("init_tensors"))
    result["receipts"] = {k: rec.get(k) for k in ("lr_constant", "amsgrad_all", "noise", "k3p_legacy", "simple_calls",
                                                   "lr", "particlegan_file", "simple_d_betas",
                                                   "activity", "final_sha256", "penalty_variant")}
    result["receipts"]["recipe"] = (rec["gantrainer_recipes"][0] if rec["gantrainer_recipes"] else None)
    result["receipts"]["declared_noise"] = {k: cfg[k] for k in ("input_noise_std", "output_noise_std")}
    result["receipts"]["declared_lr"] = {k: cfg.get(k) for k in ("lr_floor", "network_lr_floor", "d_lr_mult",
                                                                 "reg_coeff", "prior_lr_mult")}
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str, allow_nan=False) + "\n")
    log(f"# DONE {task} status={result['status']} metric={result['metric']} {result['seconds']:.0f}s")


# The 9 custom-loop transfer hosts run on CPU (cuda:1 harness fix): under the CUDA device policy on develop
# ae_gan_hold/unused_token_hold crash (CPU-only torch.set_rng_state with a CUDA generator state) and
# mid_scale_identity refuses ("CPU only"); CPU for all 9 keeps them on one device, as the gap-fill runs did.
CPU_HOSTS = set(TRANSFER[:9])


def _device(name=None):
    from benchmarks.toy100.device import apply_device_policy
    apply_device_policy("cpu" if name in CPU_HOSTS else "cuda")


def run_native(problem, cfg, cfg_path, out, args, sa, log):
    from benchmarks.toy100 import __main__ as toy_main
    ns = argparse.Namespace(command="run", config=cfg_path, output=out / "bench", problem=problem, steps=args.steps,
                            device="cuda", no_render=True, require_accuracy=True, init=None)
    code = toy_main._run(ns)
    bench = out / "bench"
    rd = lambda p: json.loads(p.read_text()) if p.exists() else {}  # noqa: E731
    pick = lambda stem: bench / f"{stem}-{problem}.json" if (bench / f"{stem}-{problem}.json").exists() else bench / f"{stem}.json"  # noqa: E731
    gate, acc, s = rd(pick("gate")), rd(pick("accuracy-gate")), rd(bench / problem / "summary.json")
    final = s.get("final", {}).get("live", {})
    passed = gate.get("status") == "PASS" and acc.get("status") == "PASS"
    first = (s.get("first_full_coverage_step") or {}).get("live")
    return dict(status="PASS" if passed else "FAIL", passed=passed, exit_code=code,
                metric=dict(name="final modes/HQ", value=f"{final.get('modes')}/{final.get('hq', float('nan')):.3f}"),
                detail=f"coverage {gate.get('status')} accuracy {acc.get('status')} first100 {first} "
                       f"massTV {final.get('mass_tv')}",
                raw=dict(final=final, first_full_coverage=first, holdout=s.get("holdout"),
                         stable_pass=(s.get("stable_pass_step") or {}).get("live")))


def run_transfer(name, cfg, cfg_path, out, args, sa, log):
    _device(name)
    from benchmarks.locked_shared import observation
    original = observation.Recorder.record

    def record(self, step, measure):
        n = len(self.curve)
        original(self, step, measure)
        if len(self.curve) == n:
            return
        row = self.curve[-1]
        log(f"eval {step:5d} " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in row.items() if k not in ("step", "seconds")))
    observation.Recorder.record = record
    from benchmarks.transfer_suite import toy100_compatibility as tc
    records = tc.run(cfg_path, out / "bench", tasks=(name,))
    row = records[0]
    verdict = row["verdict"]
    conv = verdict.get("convergence", {})
    metrics = {m["metric"]: m["value"] for m in verdict.get("metrics", [])}
    import glob
    import gzip
    nr = {}
    for f in glob.glob(str(out / "bench" / "episodes" / "*.json.gz")):
        nr = json.load(gzip.open(f)).get("noise_receipt") or {}
    sa.RECEIPT["noise"]["host_policy"] = {k: nr.get(k) for k in (
        "input_std", "input_nonzero_steps", "input_sigma_first", "output_nonzero_steps", "output_scale_initial")}
    return dict(status=verdict["status"], passed=bool(verdict.get("passed")),
                metric=dict(name="passing suffix", value=f"{conv.get('passing_suffix')}/{conv.get('observations')}"),
                detail=f"confirmed {conv.get('confirmed_step')} " + " ".join(
                    f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()),
                raw=dict(verdict=verdict, ema=row.get("ema_verdict", {}).get("status"), noise_applied=row.get("noise_applied")))


def _gate_class():
    import importlib.util
    spec = importlib.util.spec_from_file_location("convergence_gate", GATE_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConvergenceGate


class GateDone(Exception):
    pass


def run_hold(name, cfg, cfg_path, out, args, sa, log):
    _device()
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    from benchmarks.transfer_suite.protocol import required_tasks
    from benchmarks.toy100 import continuous_probe as cp
    steps, post_window = args.steps or 7500, 300
    gate = _gate_class()()
    recipe, noise, _ = declared_recipe(cfg)
    spec = {**next(r for r in required_tasks() if r["name"] == "mode_hold"), "steps": steps}
    dense, state = [], {"stop_after": None}

    def on_event(event):
        if event.get("event") != "checkpoint":
            return
        step, point = event["step"], dict(step=event["step"], modes=event["modes"], hq=event["hq"])
        if step <= gate.start_step:
            if step % 50 == 0:
                log(f"eval {step:5d} acq m={point['modes']} hq={point['hq']:.3f}")
            return
        dense.append(point)
        if not gate.done:
            gate.observe(point)
            if gate.done:
                state["stop_after"] = step + post_window
        if step % 50 == 0:
            log(f"eval {step:5d} {gate.status} m={point['modes']} hq={point['hq']:.3f} hold={gate.hold_checks} "
                f"streak={gate.streak}")
        if state["stop_after"] is not None and step >= state["stop_after"]:
            raise GateDone()
    stopped = "budget"
    try:
        cp._run_extended(spec, recipe, noise, cfg, noise_horizon=1200, diagnostic_every=50,
                         dense_after=gate.start_step, dense_until=steps, shift_step=None, shift=(1.0, 0.0),
                         freeze_after_shift=False, log=on_event)
    except GateDone:
        stopped = "gate_done_plus_post_window"
    summ = gate.summary()
    conv = summ["converged_step"]
    ext = [p for p in dense if conv and p["step"] > conv + summ["hold_budget"]] if summ["status"] == "PASS" else []
    ext_pass = sum(p["modes"] == 8 and p["hq"] >= .9 for p in ext)
    held = [p for p in dense if conv and conv < p["step"] <= conv + summ["hold_checks"]]
    passed = summ["status"] == "PASS" and len(ext) == post_window and ext_pass == post_window
    return dict(status="PASS" if passed else "FAIL", passed=passed,
                metric=dict(name="hold+ext", value=f"{summ['hold_checks']}/1200+{ext_pass}/{len(ext)}"),
                detail=f"gate {summ['status']} converged {conv} first_fail {summ['first_hold_failure']} "
                       f"minHQ(hold) {min((p['hq'] for p in held), default=None)} stopped {stopped}",
                raw=dict(gate=summ, extension=dict(checks=len(ext), passing=ext_pass,
                                                   min_hq=min((p["hq"] for p in ext), default=None))))


def run_shift(name, cfg, cfg_path, out, args, sa, log):
    _device()
    from benchmarks.toy100 import continuous_probe as cp

    def on_event(event):
        if event.get("event") == "shift" or (event.get("event") == "checkpoint" and event["step"] % 50 == 0):
            log(f"eval {event['step']:5d} {event['event']} m={event['modes']} hq={event['hq']:.3f}")
    res = cp.run_probe(cfg, mode="scheduled", steps=3600, noise_horizon=1200, diagnostic_every=10,
                       shift_step=2400, shift=(1.0, 0.0), log=on_event)
    rec, hold = res.get("shift_recovery") or {}, res.get("continued_hold") or {}
    dw = rec.get("deadline_window") or {}
    passed = bool(rec.get("deadline_pass"))
    return dict(status="PASS" if passed else "FAIL", passed=passed,
                metric=dict(name="deadline window", value=f"{dw.get('passing_checks')}/{dw.get('checks')}"),
                detail=f"delay {rec.get('delay_updates')} pre-shift hold {hold.get('passing_checks')}/{hold.get('checks')}",
                raw=dict(shift_recovery=rec, continued_hold=hold, final=res.get("final"), ema=res.get("ema")))


# ------------------------------------------------------------------ ring tasks
RING_SHIFTS = {"shift": [(2400, (1.0, 0.0))],
               "multishift": [(2400, (1.0, 0.0)), (4600, (-1.0, 0.0)), (6800, (1.0, 0.0))]}
RING_STEPS = {"shift": 4600, "multishift": 9000}
# Grid search (GRID.md) added lr, betas, prior_betas, reg_kappa: no earlier arm sets them, so earlier ring results
# are unchanged. batch_size stays the ring protocol's 2048.
RECIPE_KEYS = ("input_noise_std", "output_noise_std", "lr_floor", "network_lr_floor", "reg_coeff", "d_lr_mult",
               "prior_lr_mult", "lr", "betas", "prior_betas", "reg_kappa")


def ring_score(points, shifts, end, prehold=(1210, 2400)):
    """Per-segment ring metrics. Pass = 8 modes & HQ >= .90 at an observation (every 10 updates)."""
    pts = sorted(points, key=lambda p: p["step"])
    ok = {p["step"]: p["pass"] for p in pts}
    first_shift = shifts[0]
    pre = [p for p in pts if prehold[0] <= p["step"] <= prehold[1]]
    first_acq = next((p["step"] for p in pts if p["pass"] and p["step"] <= first_shift), None)
    seg0 = [p["pass"] for p in pts if first_acq is not None and first_acq <= p["step"] <= first_shift]
    dep0 = sum(a and not b for a, b in zip(seg0, seg0[1:]))
    pre_fail = sum(not p["pass"] for p in pre)
    segs, total_fail, departures = [], pre_fail, dep0
    bounds = list(shifts) + [end]
    for k, start in enumerate(shifts):
        stop = bounds[k + 1]
        window = [p for p in pts if start < p["step"] <= stop]
        arrival = next((p["step"] for p in window if p["pass"]), None)
        post = [p["pass"] for p in window if arrival is not None and p["step"] >= arrival]
        fails = sum(not g for g in post)
        dep = sum(a and not b for a, b in zip(post, post[1:]))
        segs.append(dict(shift_at=start, arrival=None if arrival is None else arrival - start,
                         post_pass=len(post) - fails, post_n=len(post), post_fails=fails, departures=dep,
                         final_hq=window[-1]["hq"] if window else None))
        total_fail += fails
        departures += dep
    arrived = all(s["arrival"] is not None for s in segs)
    return dict(prehold=f"{len(pre) - pre_fail}/{len(pre)}", prehold_pass=len(pre) - pre_fail, first_acq=first_acq,
                segments=segs, fails_outside_transit=total_fail, departures=departures, arrived_all=arrived,
                final_hq=pts[-1]["hq"] if pts else None)


def run_ring(name, cfg, cfg_path, out, args, sa, log):
    import torch
    from particlegan import GANTrainer, get_recipe, init
    from particlegan.training import input_noise_std, output_noise_std
    from benchmarks.locked_shared import mode_hold
    from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
    steps = args.steps or RING_STEPS[name]
    shifts = [(s, d) for s, d in RING_SHIFTS[name] if s < steps]
    overrides = {k: v for k, v in sa.ARM["config"].items() if k in RECIPE_KEYS}
    recipe = get_recipe(total_steps=steps, **overrides, **sa.k3p_overlay())
    device = "cuda"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    G = SimpleMLPGenerator(recipe.z_dim, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2)
    D = SimpleMLPDiscriminator(2, mode_hold.HIDDEN, mode_hold.N_HIDDEN, mode_hold.FOURIER)
    old_init = sa.ARM.get("init") == "constructor"  # calibration arm: no particlegan.init call
    if not old_init:
        init.deterministic_orthogonal_(G, seed=0)
        init.deterministic_orthogonal_(D, seed=1)
    G, D = G.to(device), D.to(device)
    prior = recipe.make_prior()
    prior = (prior if old_init else init.deterministic_orthogonal_(prior)).to(device)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0, optimizer_options={"foreach": False, "fused": False})
    stream = torch.Generator(device=device).manual_seed(0)
    means = mode_hold.ring_means().to(device)
    points, t0 = [], time.monotonic()
    shift_at = dict(shifts)
    log("# step modes hq pass | Ld Lg pen s | lr G/D/prior | noise in/out | sec")
    for step in range(1, steps + 1):
        idx = torch.randint(0, 8, (recipe.batch_size,), device=device, generator=stream)
        real = means[idx] + mode_hold.SIGMA * torch.randn(recipe.batch_size, 2, device=device, generator=stream)
        observe = step % 10 == 0 or step == steps
        stats = trainer.step(real, collect_stats=observe)
        if observe:
            fake = trainer.sample(mode_hold.EVAL_N, generator=torch.Generator(device=device).manual_seed(9))
            d = mode_hold.diversity(fake, means)
            p = dict(step=step, modes=d["modes"], hq=d["hq"], passed=d["modes"] == 8 and .9 <= d["hq"] <= 1.0)
            p["pass"] = p.pop("passed")
            ps = stats.get("penalty_stats") or {}
            p.update({k: ps[k] for k in ("real_rms_mean", "real_rms_max", "fake_rms_max") if k in ps})
            points.append(p)
            lrs = [trainer.opt_g.param_groups[0]["lr"], trainer.opt_d.param_groups[0]["lr"],
                   trainer.opt_g.param_groups[-1]["lr"]]
            vals = [float(stats[k]) for k in ("loss_d", "loss_g", "penalty")]
            if not all(map(lambda v: v == v and abs(v) != float("inf"), vals)):
                raise RuntimeError(f"nonfinite losses at {step}: {vals}")
            log(f"{step:5d} m={p['modes']} hq={p['hq']:.3f} {'PASS' if p['pass'] else 'fail'} | "
                f"Ld={vals[0]:+.4f} Lg={vals[1]:+.4f} pen={vals[2]:.3g} s={ps.get('s', float('nan')):.3g} "
                f"gR={ps.get('real_rms_mean', float('nan')):.3g}/{ps.get('real_rms_max', float('nan')):.3g} | "
                f"lr {'/'.join(f'{v:.3g}' for v in lrs)} | noise {input_noise_std(recipe, step - 1):.3g}/"
                f"{output_noise_std(recipe, step - 1):.3g} | {time.monotonic() - t0:.0f}s")
        if step in shift_at:
            means.add_(means.new_tensor(shift_at[step]))
            log(f"# SHIFT {shift_at[step]} after update {step}")
    score = ring_score(points, [s for s, _ in shifts], steps) if shifts else None
    passed = bool(score and score["arrived_all"] and score["fails_outside_transit"] == 0)
    arr = "/".join("none" if s["arrival"] is None else f"+{s['arrival']}" for s in (score or {}).get("segments", []))
    return dict(status="PASS" if passed else "FAIL", passed=passed,
                metric=dict(name="fails outside transit", value=None if score is None else score["fails_outside_transit"]),
                detail=(f"prehold {score['prehold']} arrival {arr} departures {score['departures']} "
                        f"final HQ {score['final_hq']:.3f}") if score else "no shift in budget",
                raw=dict(score=score, recipe=recipe.to_dict(), points=points))


# ================================================================== launcher
def launch(args):
    tasks = TASKS if not args.tasks else [t.strip() for t in args.tasks.split(",")]
    bad = [t for t in tasks if t not in ORDER]
    if bad:
        raise SystemExit(f"unknown tasks {bad}; choose from {TASKS}")
    if args.arm not in json.loads((HERE / "arms.json").read_text())["arms"]:
        raise SystemExit(f"unknown arm {args.arm}")
    tasks.sort(key=ORDER.get)
    runs = Path(args.runs_dir).resolve() / args.arm  # workers run with cwd=ROOT
    default_runs = Path(args.runs_dir).resolve() == (HERE / "runs").resolve()
    log_root = (Path(args.logs_dir).resolve() if args.logs_dir else HERE / "logs" if default_runs
                else HERE / "logs" / Path(args.runs_dir).resolve().name)
    runs.mkdir(parents=True, exist_ok=True)
    import fcntl  # one launcher per arm: a second launcher would rm -rf the first one's live task dirs
    lock = (runs / ".launcher.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another launcher is already running arm {args.arm} into {runs}")
    logs = log_root / args.arm
    logs.mkdir(parents=True, exist_ok=True)
    arm_log = (log_root / f"{args.arm}.log").open("a", buffering=1)
    gpu = args.device.split(":")[-1] if ":" in args.device else "0"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    env.pop("K3P_INIT", None)
    pending, active = list(tasks), []
    arm_log.write(f"# {time.strftime('%F %T')} arm={args.arm} device={args.device} tasks={len(tasks)} steps={args.steps}\n")
    while pending or active:
        while pending and len(active) < args.jobs:
            task = pending.pop(0)
            out = runs / task
            if (out / "result.json").exists() and not args.overwrite:
                arm_log.write(f"skip {task} (result.json exists)\n")
                continue
            if out.exists():
                subprocess.run(["rm", "-rf", str(out)], check=True)
            cmd = [str(PY), "-u", str(Path(__file__).resolve()), "--worker", "--arm", args.arm, "--task", task,
                   "--out", str(out)] + (["--steps", str(args.steps)] if args.steps else [])
            fh = (logs / f"{task}.log").open("w")
            active.append((task, out, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT),
                           fh, time.monotonic()))
        time.sleep(2)
        for item in list(active):
            task, out, proc, fh, t0 = item
            if proc.poll() is None:
                continue
            fh.close()
            active.remove(item)
            r = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
            metric = r.get("metric") or {}
            arm_log.write(f"{time.strftime('%T')} {task:26s} {r.get('status', 'NO_RESULT'):5s} "
                          f"{metric.get('name')}={metric.get('value')} | {r.get('detail', '')[:160]} | "
                          f"{r.get('seconds', time.monotonic() - t0):.0f}s rc={proc.returncode}\n")
    arm_log.write(f"# {time.strftime('%F %T')} arm={args.arm} finished\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tasks", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None, help="smoke only: shorten native/hold/ring budgets")
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    ap.add_argument("--logs-dir", default=None, help="default: logs/ (or logs/<runs-dir name>)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--task", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    worker(args) if args.worker else launch(args)


if __name__ == "__main__":
    main()
