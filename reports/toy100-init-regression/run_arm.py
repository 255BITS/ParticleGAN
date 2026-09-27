"""One ablation arm x one problem of the toy100 init-regression study.

The benchmark (``benchmarks.toy100``: config, data, affine G, particles, D,
evaluation, coverage + accuracy gates) runs unchanged through its own ``_run``
entry point, on the shipped config ``configs/toy100/constraints_simple_regularization.json``
(seed 1234, 7000 updates; RpGAN-logistic + b_cap(1,k=1), Adam (0,.999), input
noise .5->0, output noise .029, cosine LR decay with network-LR horizon cap).

Only D's construction-time weights change between arms. The benchmark's
``_init_linear`` (xavier_uniform + zero bias, which also fixes the RNG stream)
still runs first; the arm's transform is applied right after it, before the
trainer (optimizers, EMA critic) is built, with ``recipe.initialization=None``
so the recipe does not touch D again. Arm ``recipe`` instead leaves the
transform empty and sets ``initialization='batch_feature_zero'`` (the public
path, identical tensors to arm ``new``).

Every arm except ``recipe_fixed`` disables this branch's library fix
(``initialization._host_initialized``) so it reproduces develop's direct API.

A read-only probe (own RNG streams, D frozen) records D(real), D(fake) and
input-gradient norms every 50 updates to diag.jsonl; one line per 250 updates
goes to stdout (tail -f logs/<arm>-<problem>.log). result.json per run.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

import particlegan  # noqa: E402
from particlegan import initialization as I  # noqa: E402
from particlegan import qr_bz_pq_init as Q  # noqa: E402
import benchmarks.toy100.train as toy_train  # noqa: E402
from benchmarks.toy100 import __main__ as toy_main  # noqa: E402
from benchmarks.toy100.gate import evaluate_suite  # noqa: E402
from benchmarks.toy100.accuracy_gate import evaluate_suite as evaluate_accuracy  # noqa: E402

CONFIG = ROOT / "configs/toy100/constraints_simple_regularization.json"
PROBE_EVERY, PRINT_EVERY, PROBE_N = 50, 250, 2048


def _linears(D):
    return [m for m in D.modules() if isinstance(m, nn.Linear)]


def _xavier_rms(w):
    return math.sqrt(2.0 / (w.shape[0] + w.shape[1]))


def _default_rms(w):  # nn.Linear default U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    return 1.0 / math.sqrt(3.0 * w.shape[1])


def _qr(D, rms_fn, key=1):
    """QR weights keyed exactly like initialize_(D, key=1), at a chosen per-layer RMS."""
    params = list(D.parameters())
    for m in _linears(D):
        w = m.weight
        idx = next(i for i, p in enumerate(params) if p is w)
        rows, cols = w.shape
        v = Q.semi_orthogonal(Q._key(key, idx, tuple(w.shape)), rows, cols, variant="qr")
        w.copy_((v * rms_fn(w) * math.sqrt(max(rows, cols))).to(w))


def _restore(D, saved, which):
    for i, m in enumerate(_linears(D)):
        if i in which:
            m.weight.copy_(saved[i])


def _saved(D):
    return [m.weight.detach().clone() for m in _linears(D)]


@torch.no_grad()
def t_new(D):
    I._initialize(D, key=1)


@torch.no_grad()
def t_qr_xavier(D):
    _qr(D, _xavier_rms)


@torch.no_grad()
def t_rand_default(D):
    for m in _linears(D):
        m.weight.mul_(_default_rms(m.weight) / _xavier_rms(m.weight))


@torch.no_grad()
def t_new_hidden_old_readout(D):
    s = _saved(D)
    t_new(D)
    _restore(D, s, {len(s) - 1})


@torch.no_grad()
def t_old_hidden_new_readout(D):
    s = _saved(D)
    t_new(D)
    _restore(D, s, set(range(len(s) - 1)))


@torch.no_grad()
def t_qr_realized(D):
    _qr(D, lambda w: float(w.double().pow(2).mean().sqrt()))


def r2_prior(trainer):
    z = trainer.prior.z
    with torch.no_grad():
        z.copy_(Q.qmc_draw(0, z.shape, ("uniform", -5.0, 5.0), rows_as_points=True).to(z))


ARMS = {
    "old": dict(t=None, desc="old init: xavier_uniform random, zero bias (legacy pin kept)"),
    "recipe": dict(t=None, recipe_init=True, desc="public path on develop: recipe.initialization='batch_feature_zero'"),
    "new": dict(t=t_new, desc="initialize_(D,key=1): QR at torch-default Linear RMS (= recipe path)"),
    "qr_xavier": dict(t=t_qr_xavier, desc="QR (same keys) at the host's xavier RMS per layer"),
    "rand_default": dict(t=t_rand_default, desc="old xavier draw rescaled to torch-default RMS per layer"),
    "new_hid_old_ro": dict(t=t_new_hidden_old_readout, desc="new L0-L2 + old xavier readout"),
    "old_hid_new_ro": dict(t=t_old_hidden_new_readout, desc="old xavier L0-L2 + new QR readout"),
    "new_r2prior": dict(t=t_new, prior=r2_prior, desc="new D + R2 re-spaced prior (the hook's prior)"),
    "qr_xavier_r2prior": dict(t=t_qr_xavier, prior=r2_prior, desc="QR at xavier RMS + R2 prior (~ registry hook)"),
    "hook": dict(t=None, hook=True, desc="develop's registry hook --init batch_feature_zero (xavier-RMS QR + R2 prior)"),
    "fix_scale": dict(t=t_qr_realized,
                      desc="rejected fix: QR at the host's realized per-layer RMS (scale-honoring direct API)"),
    "recipe_fixed": dict(t=None, recipe_init=True, lib_fix=True,
                         desc="THE FIX: public path; initialize_ keeps host-initialized (non-default-scale) weights"),
}


class Probe:
    def __init__(self, trainer, out):
        self.t, self.file, self.t0 = trainer, out.open("w", buffering=1), time.monotonic()
        dev = trainer.device
        self.zgen = torch.Generator(device=dev).manual_seed(777)
        self.ugen = torch.Generator(device=dev).manual_seed(778)

    def __call__(self, real, stats):
        t = self.t
        step = t.completed_steps
        if step % PROBE_EVERY and step != 1:
            return
        D = t.D.model if hasattr(t.D, "model") else t.D
        flags = [p.requires_grad for p in D.parameters()]
        D.requires_grad_(False)
        try:
            # fork: G's OutputNoise draws from the global CUDA RNG; the probe must not shift training noise
            with torch.no_grad(), torch.random.fork_rng(devices=[real.device]):
                z, _ = t.prior.sample(PROBE_N, generator=self.zgen)
                fake = t.G(z)
                real = real[:PROBE_N]
                fake = fake[:len(real)]
                u = torch.rand(len(real), device=real.device, generator=self.ugen)
                path = fake + u[:, None] * (real - fake)
            x = torch.cat([real, fake, path]).detach().requires_grad_(True)
            with torch.enable_grad():
                out = D(x)
                g = torch.autograd.grad(out.sum(), x)[0].norm(dim=1)
        finally:
            for p, f in zip(D.parameters(), flags):
                p.requires_grad_(f)
        n = len(real)
        row = dict(step=step, dr=float(out[:n].mean()), df=float(out[n:2 * n].mean()),
                   dabs=float(out[:n].abs().max()), g_real=float(g[:n].mean()), g_fake=float(g[n:2 * n].mean()),
                   g_path=float(g[2 * n:].mean()), g_max=float(g.max()),
                   loss_d=float(stats["loss_d"]), loss_g=float(stats["loss_g"]),
                   lr_d=t.opt_d.param_groups[0]["lr"])
        self.file.write(json.dumps(row) + "\n")
        if step % PRINT_EVERY == 0 or step == 1:
            print(f"PROBE {step:5d} Dr={row['dr']:+.3f} Df={row['df']:+.3f} g r={row['g_real']:.2f} "
                  f"p={row['g_path']:.2f} max={row['g_max']:.2f} Ld={row['loss_d']:+.3f} Lg={row['loss_g']:+.3f} "
                  f"lr_d={row['lr_d']:.2g} {time.monotonic() - self.t0:.0f}s", flush=True)
        self.gmax = max(getattr(self, "gmax", 0.0), row["g_max"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--problem", required=True, choices=("grid100", "rotated100", "staggered100"))
    ap.add_argument("--steps", type=int, default=None, help="smoke only")
    ap.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    args = ap.parse_args()
    spec = ARMS[args.arm]
    assert Path(particlegan.__file__).resolve().is_relative_to(ROOT), particlegan.__file__
    if os.environ.get("K3P_INIT"):
        raise SystemExit("unset K3P_INIT")
    run = args.runs_dir / args.arm / args.problem
    if run.exists():
        raise SystemExit(f"{run} exists; remove it to rerun")
    run.mkdir(parents=True)
    print(f"ARM {args.arm} {args.problem} | {spec['desc']} | particlegan={particlegan.__file__}", flush=True)

    original_init, original_make = toy_train._init_linear, toy_train.make_trainer
    probes, init_stats = {}, {}

    def init_linear(module):
        original_init(module)  # xavier + zero bias: keeps the RNG stream of every arm identical
        if spec["t"] is not None and type(module).__name__ == "SimpleMLPDiscriminator":
            spec["t"](module)

    def make_trainer(config, recipe):
        if not spec.get("hook"):
            recipe = dataclasses.replace(recipe, initialization="batch_feature_zero" if spec.get("recipe_init") else None)
        trainer = original_make(config, recipe)
        if "prior" in spec:
            spec["prior"](trainer)
        D = trainer.D.model if hasattr(trainer.D, "model") else trainer.D
        init_stats.update({f"L{i}": dict(rms=float(m.weight.pow(2).mean().sqrt()),
                                         sn=float(torch.linalg.matrix_norm(m.weight.double(), 2)))
                           for i, m in enumerate(_linears(D))})
        ema = trainer.opt_d.ema_critic
        if ema is not None:  # EMA critic must start from the same weights as D
            ema_model = ema.model if hasattr(ema, "model") else ema
            try:
                same = all(torch.equal(a, b) for a, b in zip(D.state_dict().values(),
                                                              getattr(ema_model, "state_dict")().values()))
            except Exception:  # noqa: BLE001
                same = None
            init_stats["ema_matches_D"] = same
        print("INIT " + " ".join(f"{k}:rms={v['rms']:.4f},sn={v['sn']:.3f}" if isinstance(v, dict) else f"{k}={v}"
                                 for k, v in init_stats.items()), flush=True)
        probe = Probe(trainer, run / "diag.jsonl")
        probes["p"] = probe
        inner = trainer.step

        def step(real, **kw):
            stats = inner(real, **kw)
            probe(real, stats)
            return stats
        trainer.step = step
        return trainer

    if not spec.get("lib_fix") and hasattr(I, "_host_initialized"):
        I._host_initialized = lambda *a, **k: False  # every other arm reproduces develop's initialize_ (c720645e)
    toy_train._init_linear, toy_train.make_trainer = init_linear, make_trainer
    ns = argparse.Namespace(command="run", config=CONFIG, output=run / "bench", problem=args.problem,
                            steps=args.steps, device="cuda", no_render=True, require_accuracy=True,
                            init="batch_feature_zero" if spec.get("hook") else None)
    t0 = time.monotonic()
    code = toy_main._run(ns)
    cov = evaluate_suite(run / "bench", problem=args.problem)["status"]
    acc = evaluate_accuracy(run / "bench", problem=args.problem)["status"]
    s = json.loads((run / "bench" / args.problem / "summary.json").read_text())
    final = s.get("final", {}).get("live", {})
    evals = []
    for line in (run / "bench" / args.problem / "events.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("event") == "eval" and r.get("model") == "live":
            evals.append((r["step"], r["metrics"].get("modes"), r["metrics"].get("hq")))
    first100 = next((st for st, m, _ in evals if m == 100), None)
    result = dict(arm=args.arm, problem=args.problem, desc=spec["desc"], exit_code=code,
                  gate="PASS" if cov == "PASS" and acc == "PASS" else "FAIL", coverage=cov, accuracy=acc,
                  modes=final.get("modes"), hq=final.get("hq"), final_passed=final.get("passed"),
                  first_100_modes_step=first100, max_grad=getattr(probes.get("p"), "gmax", None),
                  modes_at={st: m for st, m, _ in evals if st in (500, 750, 1000, 1500, 3000)},
                  init=init_stats, wall_seconds=time.monotonic() - t0)
    (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE {args.arm} {args.problem} gate={result['gate']} modes={result['modes']} hq={result['hq']} "
          f"max_grad={result['max_grad']:.2f} first100={first100} {result['wall_seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
