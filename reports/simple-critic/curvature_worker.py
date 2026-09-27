"""Curvature arms on the simple-critic ring-8 shift protocol (wraps lr_grid.py + worker.py read-only).

Hypothesis: with R1 at the reals, D's peak there is a flat plateau (zero slope, no curvature
floor), so there is no restoring stiffness at equilibrium. Ideal D ~ Huber-rounded negative
distance to the nearest real: -kappa d^2/2 for d < 1/kappa, -d + 1/(2 kappa) beyond.

kappa (fixed once from the data, no sweep): 1/kappa = 2 * SIGMA = 0.14 (ring mode std 0.07 per
axis), so the rounded cap spans the mode (P(|x - mu| < 2 sigma) = 86% in 2D); kappa = 7.142857.

New critic terms (added to worker.SimpleCriticLoss's terms; all off = worker's loss bit-for-bit):
  --lam-margin  peak margin: reals r, offsets delta (uniform direction, |delta| ~ U(0, 1/kappa)):
                lam * E relu(kappa|delta|^2/2 - (D(r) - D(r+delta)))^2
  --nnpair      cap/path interpolation points on segments fake -> its nearest real (in batch)
                instead of random real/fake pairs
  --lam-huber   Huber profile, no input gradients: fakes f with nearest real r, and r+delta:
                lam * E (D(r) - D(x) - huber_kappa(|x - r|))^2
                huber_kappa(d) = kappa d^2/2 (d < 1/kappa) else d - 1/(2 kappa)
Offsets come from a dedicated generator (seed + 5), so the worker's u / latent / data streams are
unchanged.

Observation-only diagnostic (own generator, seed + 13, every observation):
  curv       mean over probe reals of (2 D(r) - D(r+e) - D(r-e)) / |e|^2, |e| = 0.01, random
             direction (= 2 mean(D(r) - D(r +- e))/|e|^2; > 0 means a peaked/concave D)
  curv_wide  same at |e| = 1/kappa
  gf_med     median input-grad norm of D at the eval fakes
Added to each result.json point, metrics.jsonl and the log line.

Usage:
  curvature_worker.py --lr-c 0.5 --lr-g 1 --arm A [--lam-margin 10] [--nnpair] [--lam-huber 10] <worker flags>
  curvature_worker.py --k3p-constant --out-root DIR     (observation-only rerun of k3p_worker --arm constant)
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
import torch.nn.functional as F

SIGMA = 0.07
KAPPA = 1.0 / (2 * SIGMA)
RHO = 1.0 / KAPPA
EPS = 0.01
SEED = 0
LATEST = {}


def huber(d):
    return torch.where(d < RHO, 0.5 * KAPPA * d * d, d - 0.5 * RHO)


def offsets(n, device, gen):
    direc = torch.randn(n, 2, device=device, generator=gen)
    direc = direc / direc.norm(dim=1, keepdim=True).clamp_min(1e-12)
    radius = RHO * torch.rand(n, device=device, generator=gen)
    return direc * radius[:, None]


# ---------------------------------------------------------------- diagnostic
@torch.no_grad()
def curvature(D, real):
    gen = torch.Generator(device=real.device).manual_seed(SEED + 13)
    direc = torch.randn(len(real), 2, device=real.device, generator=gen)
    direc = direc / direc.norm(dim=1, keepdim=True).clamp_min(1e-12)
    out = {}
    for key, eps in (("curv", EPS), ("curv_wide", RHO)):
        e = eps * direc
        d0, dp, dm = (D(x).reshape(-1) for x in (real, real + e, real - e))
        out[key] = float((2 * d0 - dp - dm).mean() / eps ** 2)
    return out


def fake_grad_median(D, fake):
    x = fake.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        g = torch.autograd.grad(D(x).sum(), x)[0]
    return float(g.flatten(1).norm(dim=1).median())


def wrap_probe(original):
    def probe_plus(D, real, fake, u):
        out = original(D, real, fake, u)  # restores D's train/requires_grad flags on exit
        was = D.training
        flags = [p.requires_grad for p in D.parameters()]
        D.eval()
        D.requires_grad_(False)
        try:
            extra = curvature(D, real)
            extra["gf_med"] = fake_grad_median(D, fake)
        finally:
            for p, f in zip(D.parameters(), flags):
                p.requires_grad_(f)
            D.train(was)
        LATEST.clear()
        LATEST.update(extra)
        out.update(extra)
        return out
    return probe_plus


class LogFile:
    """Line-buffered log wrapper: appends the curvature fields to every observation line."""

    def __init__(self, f):
        self.f = f

    def write(self, s):
        if s[:5].strip().isdigit() and LATEST:
            s = s.rstrip("\n") + (f" | curv={LATEST['curv']:+.3g} wide={LATEST['curv_wide']:+.3g} "
                                  f"gfmed={LATEST['gf_med']:.3f}\n")
        return self.f.write(s)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.close()

    def __getattr__(self, name):
        return getattr(self.f, name)


class LogPath(type(Path())):
    def open(self, mode="r", *a, **kw):
        f = super().open(mode, *a, **kw)
        return LogFile(f) if self.suffix == ".log" and "w" in mode else f


def merge_curvature(out_dir):
    """Copy the curvature fields from metrics.jsonl into result.json points + medians."""
    res_path = out_dir / "result.json"
    res = json.loads(res_path.read_text())
    rows = {r["step"]: r for r in map(json.loads, (out_dir / "metrics.jsonl").read_text().splitlines())}
    for p in res["points"]:
        for k in ("curv", "curv_wide", "gf_med"):
            p[k] = rows[p["step"]][k]
    med = lambda k: float(torch.tensor([p[k] for p in res["points"]]).median())
    res["curvature"] = {"kappa": KAPPA, "eps": EPS, "median_curv": med("curv"),
                        "median_curv_wide": med("curv_wide"), "median_gf": med("gf_med")}
    res_path.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    return res


# ---------------------------------------------------------------- simple-critic arms
def simple_main(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lam-margin", type=float, default=0.0)
    pre.add_argument("--lam-huber", type=float, default=0.0)
    pre.add_argument("--nnpair", action="store_true")
    extra, rest = pre.parse_known_args(argv)

    import worker
    import lr_grid

    base_loss_cls, describe, run = worker.SimpleCriticLoss, worker.describe, worker.run

    class CurvatureCriticLoss(base_loss_cls):
        """worker.SimpleCriticLoss (non-lazy branch copied verbatim) + margin / nnpair / huber."""

        def __init__(self, args):
            super().__init__(args)
            assert args.lazy_k == 1 and args.path_u is not None
            self.gen = None

        def _offsets(self, n, device):
            if self.gen is None:
                self.gen = torch.Generator(device=device).manual_seed(SEED + 5)
            return offsets(n, device, self.gen)

        def __call__(self, D, real, fake, u, step=1):
            a, n = self.a, len(real)
            parts = [real, fake]
            u = self.u_lo + (self.u_hi - self.u_lo) * u
            nn_idx = None
            if extra.nnpair or extra.lam_huber:
                with torch.no_grad():
                    nn_idx = torch.cdist(fake.detach(), real.detach()).argmin(1)
            if self.need_path:
                target = real[nn_idx] if extra.nnpair else real
                parts.append(fake + u[:, None] * (target - fake))
            x = torch.cat(parts).detach()
            need_grad = self.need_real_grad or self.need_fake_grad or self.need_path
            if need_grad:
                x.requires_grad_(True)
            out = D(x)
            dr, df = out[:n], out[n:2 * n]
            terms = {"base": self.base(dr, df)}
            zero = out.new_zeros(())
            if need_grad:
                g = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
                sq = g.pow(2).flatten(1).sum(1)
                norm = torch.sqrt(sq + 1e-12)
                npath = norm[2 * n:] if self.need_path else None
            lam_drift = a.lam_real if a.lam_drift is None else a.lam_drift
            terms["drift"] = lam_drift * dr.pow(2).mean() if "drift" in self.real_terms else zero
            terms["r1"] = a.lam_real * sq[:n].mean() if "r1" in self.real_terms else zero
            if a.path == "lower":
                terms["path"] = a.lam_path * F.relu(a.path_target - npath).pow(2).mean()
            elif a.path == "two_sided":
                terms["path"] = a.lam_path * (npath - a.path_target).pow(2).mean()
            elif a.path == "secant":
                with torch.no_grad():
                    nn = torch.cdist(fake.detach(), real.detach()).argmin(1)
                    dist = (real[nn] - fake).detach().norm(dim=1)
                terms["path"] = a.lam_path * F.relu(a.path_target * dist - (dr[nn] - df)).pow(2).mean()
            else:
                terms["path"] = zero
            if a.cap == "interp":
                terms["cap"] = a.lam_cap * F.relu(npath - a.cap_target).pow(2).mean()
            elif a.cap == "all":
                terms["cap"] = a.lam_cap * F.relu(norm - a.cap_target).pow(2).mean()
            else:
                terms["cap"] = zero
            if a.lam_center:
                terms["center"] = a.lam_center * dr.mean().pow(2)
            if extra.lam_margin or extra.lam_huber:
                delta = self._offsets(n, real.device)
                shifted = (real + delta).detach()
                d_r, d_s = dr.reshape(-1), D(shifted).reshape(-1)
                rad2 = delta.pow(2).sum(1)
                if extra.lam_margin:
                    terms["margin"] = extra.lam_margin * F.relu(
                        0.5 * KAPPA * rad2 - (d_r - d_s)).pow(2).mean()
                if extra.lam_huber:
                    d_f = df.reshape(-1)
                    dist_f = (real[nn_idx] - fake).detach().norm(dim=1)
                    res_f = d_r[nn_idx] - d_f - huber(dist_f)
                    res_s = d_r - d_s - huber(rad2.sqrt())
                    terms["huber"] = extra.lam_huber * torch.cat([res_f, res_s]).pow(2).mean()
            total = sum(terms.values())
            return total, {k: v.detach() for k, v in terms.items()}

    def describe_c(args):
        s = describe(args)
        add = []
        if extra.lam_margin:
            add.append(f"margin({extra.lam_margin:g},κ={KAPPA:.3g})")
        if extra.lam_huber:
            add.append(f"huber({extra.lam_huber:g},κ={KAPPA:.3g})")
        if extra.nnpair:
            add.append("nnpair")
        if not add:
            return s
        head, sep, tail = s.partition(" [")
        return head + " + " + " + ".join(add) + sep + tail

    def run_c(args):
        args.lam_margin, args.lam_huber, args.nnpair = extra.lam_margin, extra.lam_huber, extra.nnpair
        args.log = LogPath(args.log or HERE / "logs" / f"{args.arm}.log")
        result = run(args)
        merge_curvature(Path(args.output or HERE / "runs" / args.arm))
        return result

    worker.SimpleCriticLoss, worker.describe, worker.run = CurvatureCriticLoss, describe_c, run_c
    worker.probe = wrap_probe(worker.probe)
    sys.argv = [sys.argv[0], *rest]
    lr_grid.main()


# ---------------------------------------------------------------- k3p rerun (observation only)
def k3p_main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--k3p-constant", action="store_true")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--steps", type=int, default=4600)
    p.add_argument("--device", default="cuda:1")
    a = p.parse_args(argv)
    import k3p_worker
    k3p_worker.HERE = LogPath(a.out_root.resolve())
    k3p_worker.probe = wrap_probe(k3p_worker.probe)
    k3p_worker.run(argparse.Namespace(arm="constant", steps=a.steps, device=a.device))
    merge_curvature(a.out_root.resolve() / "runs" / "k3p_constant")


if __name__ == "__main__":
    argv = sys.argv[1:]
    (k3p_main if "--k3p-constant" in argv else simple_main)(argv)
