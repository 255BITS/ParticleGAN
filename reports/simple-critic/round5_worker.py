"""Round-5 arms on the simple-critic ring-8 shift protocol (wraps lr_grid.py + worker.py read-only,
reuses curvature_worker.py's observation-only curvature/grad diagnostics).

Base = B_cap3: wgan + R1(1) + secant(10, t .5, u in [.1,.9]) + cap-all(10, c=3), Dβ2 .9, A2 0,
critic LR x.5 (critic .002125, G .00425, prior .0085). With every new flag off the loss is
worker.SimpleCriticLoss's non-lazy branch verbatim, so B_cap3 is reproduced bit-for-bit.

New flags:
  --cap ends     cap only at the real + fake points (no path/interp points are built)
                 (--cap interp, already in worker.py, caps only the u in [.1,.9] interpolates)
  --lam-rate L   temporal rate penalty. At critic step k the critic has parameters θ_k (the result of
                 step k-1). A frozen copy holds θ_{k-1}, the parameters from before the previous
                 critic step. On the current batch (the same real r and fake f the step trains on):
                     L * mean_i[(D_θk(r_i) - D_θk-1(r_i))^2 + (D_θk(f_i) - D_θk-1(f_i))^2]
                 Gradient flows only through D_θk; D_θk-1 is evaluated under no_grad. After the loss
                 is built the copy is overwritten with θ_k, ready for step k+1. At step 1 the copy
                 equals θ_1, so the term is 0. It is a proximal penalty on D's output change per step.
  --lam-pair-center L  centering of the paired logits: L * (mean_i (D(r_i) + D(f_i)) / 2)^2.
                 Used with --loss rplogistic, worker.py's RpGAN critic base:
                     softplus(-(D(r_i) - D(f_i))) averaged over the pairs i (= softplus(D(f) - D(r))),
                 which is particlegan.gan_loss.RpGAN.d_loss. The generator loss stays the public
                 trainer's (worker --g-loss rpgan, the default for every arm, B_cap3 included):
                     RpGAN.g_loss = mean_i softplus(-(D(f'_i) - D(r_i))) on fresh fakes f'.
                 RpGAN only sees differences, so D's level is free; the centering pins it.
Diagnostics (curvature_worker, observation only): curv / curv_wide / gf_med on every observation,
in the log line, metrics.jsonl and result.json (points + "curvature" medians).

Usage: round5_worker.py --lr-c 0.5 --lr-g 1 --arm A [--lam-rate 10] [--lam-pair-center 1] <worker flags>
       (--cap ends is accepted in place of worker's --cap choices)
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
import torch.nn.functional as F


class Round5Terms:
    """New-term state shared by the ring worker and the toy100 adapter."""

    def __init__(self, lam_rate=0.0, lam_pair_center=0.0, cap_ends=False):
        self.lam_rate, self.lam_pair_center, self.cap_ends = lam_rate, lam_pair_center, cap_ends
        self.prev = None

    def add(self, terms, D, x_rf, dr, df):
        """x_rf: detached real+fake batch; dr, df: current D outputs on it (with grad)."""
        if self.lam_pair_center:
            terms["pcenter"] = self.lam_pair_center * (0.5 * (dr + df)).mean().pow(2)
        if self.lam_rate:
            if self.prev is None:
                self.prev = copy.deepcopy(D)
                self.prev.requires_grad_(False)
            with torch.no_grad():
                was = self.prev.training
                self.prev.train(D.training)
                old = self.prev(x_rf)
                self.prev.train(was)
            cur = torch.cat([dr, df])
            n = len(dr)
            diff2 = (cur - old).pow(2).reshape(2, n, -1).sum(0)  # per pair: (Δr)^2 + (Δf)^2
            terms["rate"] = self.lam_rate * diff2.mean()
            with torch.no_grad():  # θ_k becomes "before the previous step" at step k+1
                for p, q in zip(self.prev.parameters(), D.parameters()):
                    p.copy_(q)
                for p, q in zip(self.prev.buffers(), D.buffers()):
                    p.copy_(q)


def round5_loss(base_cls, extra):
    """worker.SimpleCriticLoss (non-lazy branch, copied verbatim) + Round5Terms."""

    class Round5CriticLoss(base_cls):
        def __init__(self, args):
            cap_ends = args.cap == "ends"
            if cap_ends:
                args.cap = "none"  # worker's flags: real/fake grads but no path points
            super().__init__(args)
            if cap_ends:
                args.cap = "ends"
                self.need_real_grad = self.need_fake_grad = True
            assert args.lazy_k == 1 and args.path_u is not None
            self.r5 = Round5Terms(extra.lam_rate, extra.lam_pair_center, cap_ends)

        def __call__(self, D, real, fake, u, step=1):
            a, n = self.a, len(real)
            parts = [real, fake]
            u = self.u_lo + (self.u_hi - self.u_lo) * u
            if self.need_path:
                parts.append(fake + u[:, None] * (real - fake))
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
            elif a.cap == "ends":
                terms["cap"] = a.lam_cap * F.relu(norm[:2 * n] - a.cap_target).pow(2).mean()
            else:
                terms["cap"] = zero
            if a.lam_center:
                terms["center"] = a.lam_center * dr.mean().pow(2)
            self.r5.add(terms, D, x[:2 * n].detach(), dr, df)
            total = sum(terms.values())
            return total, {k: v.detach() for k, v in terms.items()}

    return Round5CriticLoss


def main(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lam-rate", type=float, default=0.0)
    pre.add_argument("--lam-pair-center", type=float, default=0.0)
    extra, rest = pre.parse_known_args(argv)
    cap_ends = False
    if "--cap" in rest and rest[rest.index("--cap") + 1] == "ends":
        rest[rest.index("--cap") + 1] = "none"  # parsed by worker.main as none, restored in run_r5
        cap_ends = True

    import curvature_worker as cw
    import lr_grid
    import worker

    describe, run = worker.describe, worker.run
    worker.SimpleCriticLoss = round5_loss(worker.SimpleCriticLoss, extra)

    def describe_r5(args):
        s = describe(args)
        add = []
        if extra.lam_pair_center:
            add.append(f"pair-center({extra.lam_pair_center:g})")
        if extra.lam_rate:
            add.append(f"rate({extra.lam_rate:g})")
        if not add:
            return s
        head, sep, tail = s.partition(" [")
        return head + " + " + " + ".join(add) + sep + tail

    def run_r5(args):
        if cap_ends:
            args.cap = "ends"
        args.lam_rate, args.lam_pair_center = extra.lam_rate, extra.lam_pair_center
        args.log = cw.LogPath(args.log or HERE / "logs" / f"{args.arm}.log")
        result = run(args)
        cw.merge_curvature(Path(args.output or HERE / "runs" / args.arm))
        return result

    worker.describe, worker.run = describe_r5, run_r5
    worker.probe = cw.wrap_probe(worker.probe)
    sys.argv = [sys.argv[0], *rest]
    lr_grid.main()


if __name__ == "__main__":
    main(sys.argv[1:])
