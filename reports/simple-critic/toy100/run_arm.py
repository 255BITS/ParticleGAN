"""Run one arm of the simple-critic transfer study on the 100-Gaussian gate.

The benchmark harness (``benchmarks.toy100``: config resolution, data, affine G,
particles, D, evaluation schedule, coverage gate and accuracy gate) is used
unchanged through its own ``run`` entry point. This adapter only:

* writes the arm's JSON config: the default
  ``configs/toy100/constraints_simple_regularization.json`` with instance noise
  off (input and output noise 0) and constant LRs (``lr_floor`` and
  ``network_lr_floor`` 1), except for the ``ref_stock`` control;
* for simple-critic arms, swaps the critic update inside the ordinary
  ``GANTrainer.step``: the recipe's D loss becomes the simple critic loss from
  ``../worker.py`` (``SimpleCriticLoss``, same code as the ring study) and the
  critic optimizer is rebuilt with no EMA critic, no guard and Adam betas
  (0, beta2). The generator/prior update, LR policy, EMA and data streams are
  the trainer's own;
* uses the package's default initialization for every arm (refs included): the benchmark
  resolves its archived configs through ``benchmarks.legacy`` whose recipe pins
  ``initialization=None`` (old random init); the adapter replaces that pin with
  ``initialization="batch_feature_zero"`` before the trainer is built, so
  ``GANTrainer`` -> ``recipe.make_optimizers`` initializes D (key 1) and G (key 0) and
  syncs the EMA critic, exactly as the public path. The benchmark's ``--init`` registry
  hook is not used (``K3P_INIT`` must be unset). Kept as drawn, by design of the public
  path: the ``affine_square_v1`` generator (identity weight, zero bias = constants) and its
  caller-supplied particle table (the model policy's uniform(-5, 5) square draw;
  ``make_prior``'s R2 draw is overwritten by that policy and supplied priors are never
  re-initialized). ``result.json["init_receipt"]`` records all of this per problem;
* probes D every ``PROBE_EVERY`` updates (no effect on training: its own
  latent/u streams, D frozen): max |D(real)| and input-gradient norms at real,
  fake and interpolate points.

One line per probe/eval in logs/<arm>.log (tail -f). result.json per arm.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(HERE.parent))

import torch  # noqa: E402

import particlegan  # noqa: E402
from particlegan import initialization as _pg_init  # noqa: E402
import benchmarks.toy100.train as toy_train  # noqa: E402
from benchmarks.toy100 import __main__ as toy_main  # noqa: E402
from worker import SimpleCriticLoss, grad_norm  # noqa: E402  (ring-study critic, unchanged)
from round5_worker import round5_loss  # noqa: E402  (round-5 terms: cap ends, rate, pair-center)
import init_receipt  # noqa: E402

INITIALIZATION = "batch_feature_zero"  # the package default (#194); overrides the legacy recipe's None pin

BASE_CONFIG = ROOT / "configs/toy100/constraints_simple_regularization.json"
PROBLEMS = ("grid100", "rotated100", "staggered100")
PROBE_EVERY = 50
PROBE_N = 2048

SIMPLE = dict(loss="wgan", real="r1", lam_real=1.0, lam_drift=None, path="secant", lam_path=10.0,
              path_target=0.5, path_u="0.1,0.9", cap="all", lam_cap=10.0, cap_target=1.0,
              lam_center=0.0, lazy_k=1)
ARMS = {
    "ref_stock": dict(kind="ref", matched=False,
                      desc="REF: benchmark default as shipped (RpGAN-logistic + b_cap(1,k=1), Adam b2 .999, "
                           "input noise .5->0, output noise .029, cosine LR decay)"),
    "ref_matched": dict(kind="ref", matched=True,
                        desc="REF: benchmark default critic (RpGAN-logistic + b_cap(1,k=1), Adam b2 .999), "
                             "noise off, constant LRs"),
    "sec_nodamp": dict(kind="simple", d_beta2=0.9, latent_damping=0.0, crit=SIMPLE,
                       desc="wgan + r1(1) + path-secant(10,t=.5) + cap-all(10,c=1) [Db2=.9, A2=0]"),
    "secant_r1_b2": dict(kind="simple", d_beta2=0.9, latent_damping=0.5, crit=SIMPLE,
                         desc="wgan + r1(1) + path-secant(10,t=.5) + cap-all(10,c=1) [Db2=.9, A2=.5]"),
    "sec_nodamp_lazy4": dict(kind="simple", d_beta2=0.9, latent_damping=0.0, crit={**SIMPLE, "lazy_k": 4},
                             desc="wgan + r1(1) + path-secant(10,t=.5) + cap-all(10,c=1) [Db2=.9, A2=0, lazy_k=4]"),
}
# Round 5: ring B_cap3 ports (cap c=3, critic LR x0.5 via the config's d_lr_mult) plus one change each.
CAP3 = {**SIMPLE, "cap_target": 3.0}
R5 = dict(kind="simple", d_beta2=0.9, latent_damping=0.0, cfg={"d_lr_mult": 0.5})
ARMS.update({
    "c3_r1w": dict(R5, crit={**CAP3, "lam_real": 0.1},
                   desc="wgan + r1(0.1) + path-secant(10,t=.5) + cap-all(10,c=3) [Db2=.9, A2=0, critic LR x.5]"),
    "c3_capinterp": dict(R5, crit={**CAP3, "cap": "interp"},
                         desc="wgan + r1(1) + path-secant(10,t=.5) + cap-interp(10,c=3) [Db2=.9, A2=0, critic LR x.5]"),
    "rp_center": dict(R5, crit={**CAP3, "loss": "rplogistic"}, r5={"lam_pair_center": 1.0},
                      desc="rplogistic + r1(1) + path-secant(10,t=.5) + cap-all(10,c=3) + pair-center(1) "
                           "[Db2=.9, A2=0, critic LR x.5]"),
})


def arm_config(name: str) -> dict:
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg["name"] = f"simple-critic/{name}"
    if ARMS[name].get("matched", True):
        cfg.update(input_noise_std=0.0, output_noise_std=0.0, lr_floor=1.0, network_lr_floor=1.0)
    cfg.update(ARMS[name].get("cfg", {}))
    return cfg


class _Crit:
    """Stands in for the trainer's (loss, penalty) pair so GANTrainer.step runs the simple critic."""

    def __init__(self, trainer, spec, log, r5=None):
        cls = SimpleCriticLoss if r5 is None else round5_loss(
            SimpleCriticLoss, SimpleNamespace(**{"lam_rate": 0.0, "lam_pair_center": 0.0, **r5}))
        self.trainer, self.loss_fn = trainer, cls(SimpleNamespace(**spec))
        self.base_loss = trainer.loss
        self.critic_steps, self.collect_stats, self.last_stats, self.last_terms = 0, False, {}, {}
        self.log = log

    # trainer.loss replacement: d_loss is zero (the whole critic loss lives in the "penalty")
    def d_loss(self, real_logits, fake_logits):
        return real_logits.new_zeros(())

    def g_loss(self, fake_logits, real_logits):
        return self.base_loss.g_loss(fake_logits, real_logits)  # RpGAN, unchanged

    def __call__(self, critic, real, fake):  # trainer.penalty replacement
        self.critic_steps += 1
        u = torch.rand(len(real), device=real.device, generator=self.trainer.penalty_generator)
        total, terms = self.loss_fn(critic, real, fake, u, self.critic_steps)
        self.last_terms = terms
        return total


def convert_to_simple(trainer, arm):
    spec = ARMS[arm]
    D = trainer.D
    # Critic optimizer: recipe factory, no EMA critic and no guard -> Adam update + read-only bookkeeping.
    recipe = trainer.recipe
    opt_d = recipe.make_critic_optimizer(D, ema_critic=None, betas=(recipe.betas[0], spec["d_beta2"]),
                                         **trainer.optimizer_options)
    assert opt_d.guard is None and opt_d.ema_critic is None, "critic optimizer must carry no controller"
    trainer.opt_d = opt_d
    trainer.initial_lrs[1] = [g["lr"] for g in opt_d.param_groups]
    crit = _Crit(trainer, dict(spec["crit"]), None, spec.get("r5", {} if "cfg" in spec else None))
    trainer.loss, trainer.penalty = crit, crit
    return crit


class Probe:
    def __init__(self, trainer, problem, out, log, crit):
        self.t, self.problem, self.log, self.crit = trainer, problem, log, crit
        dev = trainer.device
        self.zgen = torch.Generator(device=dev).manual_seed(777)
        self.ugen = torch.Generator(device=dev).manual_seed(778)
        self.rows, self.file = [], out.open("w", buffering=1)
        self.t0 = time.monotonic()

    def __call__(self, real, stats):
        t = self.t
        step = t.completed_steps
        if step % PROBE_EVERY and step != t.recipe.total_steps and step != 1:
            return
        D = t.D
        flags = [p.requires_grad for p in D.parameters()]
        D.requires_grad_(False)
        try:
            with torch.no_grad():
                z, _ = t.prior.sample(PROBE_N, generator=self.zgen)
                fake = t.G(z)
                real = real[:PROBE_N]
                fake = fake[:len(real)]
                u = torch.rand(len(real), device=real.device, generator=self.ugen)
                path = fake + u[:, None] * (real - fake)
            x = torch.cat([real, fake, path]).detach().requires_grad_(True)
            with torch.enable_grad():
                out = D(x)
                g = torch.autograd.grad(out.sum(), x)[0]
        finally:
            for p, f in zip(D.parameters(), flags):
                p.requires_grad_(f)
        n = len(real)
        norm = grad_norm(g)
        dr, df = out[:n].detach(), out[n:2 * n].detach()
        row = dict(step=step, dr_mean=float(dr.mean()), dr_absmax=float(dr.abs().max()),
                   df_mean=float(df.mean()), g_real=float(norm[:n].mean()), g_fake=float(norm[n:2 * n].mean()),
                   g_path=float(norm[2 * n:].mean()), g_max=float(norm.max()),
                   loss_d=float(stats["loss_d"]), loss_g=float(stats["loss_g"]),
                   lr_d=t.opt_d.param_groups[0]["lr"], lr_g=t.opt_g.param_groups[0]["lr"],
                   lr_prior=t.opt_g.param_groups[1]["lr"])
        if self.crit is not None:
            row.update({k: float(v) for k, v in self.crit.last_terms.items()})
        self.rows.append(row)
        self.file.write(json.dumps(row) + "\n")
        terms = ""
        if self.crit is not None:
            terms = " " + " ".join(f"{k}={float(v):.3g}" for k, v in self.crit.last_terms.items()
                                   if k == "base" or float(v) != 0)
        print(f"PROBE {self.problem} {step:5d} Dr={row['dr_mean']:+.3f} |Dr|max={row['dr_absmax']:.3f} "
              f"Df={row['df_mean']:+.3f} g r={row['g_real']:.3f} f={row['g_fake']:.3f} p={row['g_path']:.3f} "
              f"max={row['g_max']:.3f} Ld={row['loss_d']:+.4f}{terms} Lg={row['loss_g']:+.4f} "
              f"lr_d={row['lr_d']:.5g} {time.monotonic() - self.t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--steps", type=int, default=None, help="smoke tests only: override the config's steps")
    ap.add_argument("--runs-dir", type=Path, default=HERE / "runs", help="parent of the arm's output dir")
    ap.add_argument("--init", choices=("recipe", "none", "hook"), default="recipe",
                    help="controls only: recipe = package default via the recipe path (the study setting); "
                         "none = keep the legacy None pin (old random init replay); "
                         "hook = develop's toy100 mechanism, registry use_init('batch_feature_zero') (also re-spaces the prior)")
    args = ap.parse_args()
    arm, spec = args.arm, ARMS[args.arm]
    assert Path(particlegan.__file__).resolve().is_relative_to(ROOT), particlegan.__file__
    if os.environ.get("K3P_INIT"):
        raise SystemExit("unset K3P_INIT: the adapter applies the package init through the recipe path")
    runs = args.runs_dir / arm
    if runs.exists():
        raise SystemExit(f"{runs} exists; remove it to rerun")
    (HERE / "configs").mkdir(exist_ok=True)
    cfg_path = HERE / "configs" / f"{arm}.json"
    cfg_path.write_text(json.dumps(arm_config(arm), indent=2, sort_keys=True) + "\n")
    diag = runs / "diag"
    diag.mkdir(parents=True)
    out = runs / "bench"
    print(f"ARM {arm} | {spec['desc']} | particlegan={particlegan.__file__} | torch={torch.__version__}",
          flush=True)

    probes, receipts = {}, {}
    original_make = toy_train.make_trainer

    def make_trainer(config, recipe):
        if spec["kind"] == "simple" and spec["latent_damping"] != recipe.latent_damping_max_rate:
            recipe = dataclasses.replace(recipe, latent_damping_max_rate=spec["latent_damping"])
        pinned = recipe.initialization
        if args.init == "recipe":
            recipe = dataclasses.replace(recipe, initialization=INITIALIZATION)
            assert _pg_init._external_init is None, "an --init registry hook is installed"
        trainer = original_make(config, recipe)
        crit = convert_to_simple(trainer, arm) if spec["kind"] == "simple" else None
        problem = config["problem"]
        if args.init == "hook":  # reference builds would advance the hook's construction order; record live hashes only
            receipts[config["problem"]] = {"initialization": "hook", "external_init_hook": _pg_init._external_init,
                                           "param_sha256": {k: init_receipt.param_sha256(m) for k, m in
                                                            (("G", trainer.G), ("D", trainer.D), ("prior", trainer.prior))}}
            print(f"INIT {config['problem']} init=hook hook={_pg_init._external_init} sha "
                  + " ".join(f"{k}={v[:12]}" for k, v in receipts[config['problem']]["param_sha256"].items()), flush=True)
        else:
            # make_trainer forks the CPU/device RNG, so these reference builds leave the run untouched.
            ref = {k: original_make(config, r) for k, r in (("old", init_receipt.old_recipe(recipe)),
                                                            ("public", recipe))}
            receipts[problem] = {**init_receipt.receipt(
                recipe, {"G": trainer.G, "D": trainer.D, "prior": trainer.prior},
                old={"G": ref["old"].G, "D": ref["old"].D, "prior": ref["old"].prior},
                public={"G": ref["public"].G, "D": ref["public"].D, "prior": ref["public"].prior},
                ema_D=trainer.opt_d.ema_critic,
                applied_via="GANTrainer -> recipe.make_optimizers (benchmark make_trainer, legacy None pin replaced)"),
                "legacy_pin_replaced": pinned}
            print(f"INIT {problem} " + init_receipt.summary_line(receipts[problem])[2:].strip(), flush=True)
            del ref
        probe = Probe(trainer, problem, diag / f"{problem}.jsonl", None, crit)
        probes[problem] = probe
        inner = trainer.step

        def step(real, **kw):
            stats = inner(real, **kw)
            probe(real, stats)
            return stats
        trainer.step = step
        print(f"TRAINER {problem} opt_d={type(trainer.opt_d).__name__} betas={trainer.opt_d.param_groups[0]['betas']} "
              f"ema_critic={trainer.opt_d.ema_critic is not None} A2={trainer.recipe.latent_damping_max_rate} "
              f"noise_in={config['input_noise_std']} noise_out={config['output_noise_std']} "
              f"lr_floor={config['lr_floor']} net_floor={config.get('network_lr_floor')}", flush=True)
        return trainer

    toy_train.make_trainer = make_trainer
    ns = argparse.Namespace(command="run", config=cfg_path, output=out, problem=None, steps=args.steps,
                            device="cuda", no_render=True, require_accuracy=True,
                            init=INITIALIZATION if args.init == "hook" else None)
    started = time.monotonic()
    code = toy_main._run(ns)

    # ---- aggregate
    result = {"arm": arm, "formulation": spec["desc"], "init_mode": args.init, "exit_code": code,
              "wall_seconds": time.monotonic() - started, "init_receipt": receipts, "problems": {}}
    gate = json.loads((out / "gate.json").read_text()) if (out / "gate.json").exists() else {}
    acc = json.loads((out / "accuracy-gate.json").read_text()) if (out / "accuracy-gate.json").exists() else {}
    result["gate_status"], result["accuracy_status"] = gate.get("status"), acc.get("status")
    for problem in PROBLEMS:
        s_path = out / problem / "summary.json"
        s = json.loads(s_path.read_text()) if s_path.exists() else {}
        evals = []
        ev_path = out / problem / "events.jsonl"
        if ev_path.exists():
            for line in ev_path.read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "eval" and r.get("model") == "live" and r.get("step", 0) > 0:
                    m = r["metrics"]
                    evals.append((r["step"], bool(m.get("passed")), m.get("modes"), m.get("hq")))
        rows = probes[problem].rows if problem in probes else []
        gmax = [r["g_max"] for r in rows]
        final = s.get("final", {}).get("live", {})
        result["problems"][problem] = {
            "status": s.get("status"),
            "first_full_coverage": (s.get("first_full_coverage_step") or {}).get("live"),
            "first_pass": (s.get("first_pass_step") or {}).get("live"),
            "stable_pass": (s.get("stable_pass_step") or {}).get("live"),
            "final": {k: final.get(k) for k in ("modes", "hq", "mass_tv", "min_cov_eig_ratio", "max_cov_eig_ratio",
                                                 "min_radial_median_ratio", "max_radial_median_ratio", "passed")},
            "holdout": s.get("holdout"),
            "evals_passed": sum(p for _, p, _, _ in evals), "evals_total": len(evals),
            "evals_passed_after_1000": sum(p for st, p, _, _ in evals if st >= 1000),
            "evals_after_1000": sum(1 for st, _, _, _ in evals if st >= 1000),
            "max_abs_D_real": max((r["dr_absmax"] for r in rows), default=None),
            "max_grad_norm": max(gmax, default=None),
            "median_grad_norm_max": statistics.median(gmax) if gmax else None,
            "probes_gmax_gt2": sum(g > 2 for g in gmax), "probes": len(gmax),
            "train_seconds": s.get("train_seconds"),
            "error": s.get("error"),
        }
    (runs / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE {arm} exit={code} gate={result['gate_status']} accuracy={result['accuracy_status']} "
          f"{result['wall_seconds']:.0f}s", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
