"""
Gradient regularizers for GAN discriminators.

The repo baseline (examples/100gaussians.py) uses R1+R2, i.e. zero-centered
penalties on ||grad_x D(x)|| at the reals and the fakes. That drives D toward
flatness: the unique global minimum of the penalty alone is a constant D.

This module collects a family of *re-centered* alternatives so they can be
swapped in a controlled experiment. The interesting one is the "eikonal"
penalty (n - 1)^2, borrowed from implicit-surface fitting, which pins the slope
of D at 1 instead of 0 and therefore *forbids* a flat discriminator while still
bounding its steepness.

All arms share the same coefficient units: every data-point arm is written as

    penalty = (coeff / 2) * ( E_real[phi(n_r)] + E_fake[phi(n_f)] )

so that `a_r1r2` with coeff=0.02 reproduces `r1_gamma=0.02` in the example
script exactly.

Two orthogonal knobs sit on top of that family, both off by default:

  * `norm` picks *which* norm of grad_x D(x) is penalized. Penalizing the L1
    norm of the gradient corresponds to enforcing an L-infinity margin around
    the decision boundary, and penalizing the L-infinity norm corresponds to an
    L1 margin -- the norm on the gradient and the norm on the margin are dual
    (cf. arXiv:1910.06922, "Gradient penalty from a maximum margin
    perspective"). The default L2/L2 pairing is just the self-dual case.
  * `target_anneal` moves the penalty *center* c over training. As c -> 0 the
    re-centered arms continuously morph into a zero-centered R1/R2-like
    penalty: b_cap's cap slides down to relu(n)^2 = n^2 and c_eikonal's
    (n - 1)^2 becomes (n - 0)^2 = ||g||^2. So an annealed run starts out
    forbidding a flat D and ends up with the baseline's flatness pressure,
    without ever switching arms mid-run.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class GradRegularizer:
    """
    Discriminator gradient penalty with a selectable centering scheme.

    Args:
        arm (str): one of ARMS.
            - 'a_r1r2':    phi(n) = n^2          (baseline R1+R2, zero-centered)
            - 'b_cap':     phi(n) = relu(n - kappa)^2   (one-sided cap, free below kappa)
            - 'c_eikonal': phi(n) = (n - 1)^2    (two-sided, slope pinned at 1)
            - 'd_asym':    phi(n) = relu(n - 1)^2 + 0.25 * relu(1 - n)^2
                           (eikonal, but 4x cheaper to be too flat than too steep)
            - 'e_interp':  (n_i - 1)^2 on real/fake interpolates (WGAN-GP geometry,
                           centered at 1 rather than at 0)
            - 'f_none':    no penalty at all.
        coeff (float): penalty strength (`r1_gamma` in the example script).
        kappa (float): the cap for 'b_cap'; ignored by the other arms.
        lazy_k (int): apply the penalty only every k-th step and multiply the
            coefficient by k ("lazy regularization", StyleGAN2). k=1 means
            every step.
        norm (str): which norm of grad_x D(x) the arms b_cap / c_eikonal /
            d_asym / e_interp penalize.
            - 'l2':   n = sqrt(sum_i g_i^2 + 1e-12)   (default, self-dual)
            - 'l1':   n = sum_i |g_i|
            - 'linf': n = max_i |g_i|
            l1 and linf need no epsilon: there is no sqrt to differentiate at
            g = 0. Because the norm on the gradient is dual to the norm in
            which the margin is measured, penalizing ||g||_1 enforces an
            L-infinity margin and penalizing ||g||_inf enforces an L1 margin
            (arXiv:1910.06922). `a_r1r2` is defined as the squared-L2 R1/R2
            form and keeps that path regardless of this setting, so asking it
            for any other norm is a ValueError rather than a silent no-op.
        target_anneal (str): schedule for the penalty *center* c, i.e. the
            kappa in b_cap and the literal 1.0 in c_eikonal / d_asym /
            e_interp. With c0 = kappa for b_cap and 1.0 for the others:
            - 'none':    c = c0 at every step (the historical behavior).
            - 'linear':  c(step) = c0 * max(0, 1 - step / total_steps).
            - 'delayed': c = c0 while step < 0.6 * total_steps, then linear
                         from c0 down to 0 at total_steps.
            As c -> 0 the re-centered arms degrade continuously into a
            zero-centered R1/R2-like penalty (relu(n)^2 = n^2 for b_cap,
            (n - 0)^2 for c_eikonal), so an anneal is a smooth handover from
            "D may not be flat" to "D should be flat". `a_r1r2` has no center
            and is unaffected.
        total_steps (int): run length the schedule is expressed against.
            Required (> 0) whenever `target_anneal` is not 'none'.
    """

    ARMS = ("a_r1r2", "b_cap", "c_eikonal", "d_asym", "e_interp", "f_none")
    NORMS = ("l2", "l1", "linf")
    ANNEALS = ("none", "linear", "delayed")

    # Fraction of the run that 'delayed' holds the center at c0 before the ramp.
    DELAY_FRAC = 0.6

    def __init__(
        self,
        arm: str,
        coeff: float,
        kappa: float = 1.0,
        lazy_k: int = 1,
        norm: str = "l2",
        target_anneal: str = "none",
        total_steps: int = 0,
    ) -> None:
        if arm not in self.ARMS:
            raise ValueError(f"Unknown grad regularizer arm: {arm} (expected one of {self.ARMS})")

        self.arm = arm
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)
        self.norm = str(norm)
        self.target_anneal = str(target_anneal)
        self.total_steps = int(total_steps)

        if self.lazy_k < 1:
            raise ValueError(f"lazy_k must be >= 1, got {lazy_k}")

        if self.norm not in self.NORMS:
            raise ValueError(f"Unknown grad norm: {norm} (expected one of {self.NORMS})")
        if self.arm == "a_r1r2" and self.norm != "l2":
            raise ValueError(
                "arm 'a_r1r2' is the squared-L2 R1/R2 penalty and supports only "
                f"norm='l2', got norm={norm!r}"
            )

        if self.target_anneal not in self.ANNEALS:
            raise ValueError(
                f"Unknown target_anneal: {target_anneal} (expected one of {self.ANNEALS})"
            )
        if self.target_anneal != "none" and self.total_steps <= 0:
            raise ValueError(
                f"target_anneal={target_anneal!r} needs total_steps > 0, got {total_steps}"
            )

    # -------------------------
    #  Public API
    # -------------------------

    def penalty(
        self,
        D: torch.nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the penalty term to add to the discriminator loss.

        Returns:
            (penalty_loss, stats)
            penalty_loss: scalar tensor attached to D's graph, or a detached
                zero on the right device/dtype when the arm is 'f_none' or the
                lazy schedule skips this step.
            stats: {'applied': bool, 'pen': float} plus 'center' (the penalty
                center actually used) on the steps where it is applied.
        """
        skip = self.arm == "f_none" or (self.lazy_k > 1 and step % self.lazy_k != 0)
        if skip:
            zero = torch.zeros((), device=x_real.device, dtype=x_real.dtype)
            return zero, {"applied": False, "pen": 0.0}

        # Lazy regularization: fewer applications, proportionally bigger hits,
        # so the time-averaged pressure on D is unchanged.
        coeff_eff = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff

        center = self.center(step)

        if self.arm == "e_interp":
            pen = coeff_eff * self._interp_term(D, x_real, x_fake, center)
        else:
            n_r = self._grad_norm(D, x_real, squared=(self.arm == "a_r1r2"))
            n_f = self._grad_norm(D, x_fake, squared=(self.arm == "a_r1r2"))
            phi = self._phi
            pen = (coeff_eff / 2.0) * (phi(n_r, center).mean() + phi(n_f, center).mean())

        return pen, {"applied": True, "pen": float(pen.detach()), "center": float(center)}

    def center(self, step: int) -> float:
        """
        The penalty center c in force at `step` (see `target_anneal`).

        c0 is `kappa` for b_cap and 1.0 for c_eikonal / d_asym / e_interp.
        `a_r1r2` is zero-centered by construction and always reports 0.0.
        """
        if self.arm == "a_r1r2":
            return 0.0
        c0 = self.kappa if self.arm == "b_cap" else 1.0

        if self.target_anneal == "none":
            return c0
        if self.target_anneal == "linear":
            return c0 * max(0.0, 1.0 - step / self.total_steps)
        if self.target_anneal == "delayed":
            hold = self.DELAY_FRAC * self.total_steps
            if step < hold:
                return c0
            frac = (step - hold) / max(1e-12, self.total_steps - hold)
            return c0 * max(0.0, 1.0 - frac)
        raise ValueError(f"Unknown target_anneal: {self.target_anneal}")

    # -------------------------
    #  Penalty kernels
    # -------------------------

    def _phi(self, n: torch.Tensor, center: float) -> torch.Tensor:
        """Per-sample penalty on the gradient norm (or its square, for a_r1r2)."""
        if self.arm == "a_r1r2":
            # `n` is already ||g||^2 here: no sqrt, no epsilon, so this is
            # bit-for-bit the inline R1/R2 formula in examples/100gaussians.py.
            return n
        if self.arm == "b_cap":
            return F.relu(n - center).pow(2)
        if self.arm == "c_eikonal":
            return (n - center).pow(2)
        if self.arm == "d_asym":
            # Steepness costs full price; flatness costs a quarter. Keeps the
            # "no flat D" property while letting D relax between the modes.
            return F.relu(n - center).pow(2) + 0.25 * F.relu(center - n).pow(2)
        raise ValueError(f"Unknown grad regularizer arm: {self.arm}")

    def _interp_term(
        self,
        D: torch.nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        center: float,
    ) -> torch.Tensor:
        """E[(||grad D(x_i)|| - c)^2] on per-sample real/fake interpolates."""
        eps = torch.rand(
            x_real.shape[0], 1, device=x_real.device, dtype=x_real.dtype
        )
        x_i = eps * x_real.detach() + (1.0 - eps) * x_fake.detach()
        n_i = self._grad_norm(D, x_i, squared=False)
        return (n_i - center).pow(2).mean()

    def _grad_norm(
        self,
        D: torch.nn.Module,
        x: torch.Tensor,
        squared: bool = False,
    ) -> torch.Tensor:
        """
        ||grad_x D(x)||, per sample, in the norm selected by `self.norm`, with
        create_graph=True so the penalty is differentiable w.r.t. D's params.

        `squared` returns the squared L2 norm ||g||^2 without the sqrt/epsilon
        (the R1/R2 form) and ignores `self.norm`, which the constructor has
        already pinned to 'l2' for the one arm that asks for it.
        """
        x = x.detach().clone().requires_grad_(True)
        logits = D(x)
        g = torch.autograd.grad(logits.sum(), x, create_graph=True)[0]
        if squared:
            return g.pow(2).flatten(1).sum(dim=1)
        if self.norm == "l1":
            # No epsilon: |.| is already differentiable a.e. and there is no
            # sqrt whose derivative blows up at g = 0.
            return g.abs().flatten(1).sum(dim=1)
        if self.norm == "linf":
            return g.abs().flatten(1).max(dim=1).values
        sq = g.pow(2).flatten(1).sum(dim=1)
        # Epsilon keeps the sqrt differentiable at g = 0.
        return torch.sqrt(sq + 1e-12)


def grad_norm_stats(
    D: torch.nn.Module,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
) -> Dict[str, float]:
    """
    Measurement-only summary of D's gradient-norm field (no create_graph, so
    nothing here is backpropagated).

    Reports the median and the 10/90 percentiles of ||grad_x D(x)|| at the
    reals (nr), at the fakes (nf), and at per-sample uniform interpolates
    x_i = eps * x_real + (1 - eps) * x_fake with eps ~ U(0, 1) (ni).

    Safe to call inside a `torch.no_grad()` eval block: it re-enables grad
    locally and works on detached clones.
    """
    with torch.enable_grad():
        eps = torch.rand(
            x_real.shape[0], 1, device=x_real.device, dtype=x_real.dtype
        )
        x_interp = eps * x_real.detach() + (1.0 - eps) * x_fake.detach()

        norms = {}
        for key, x in (("r", x_real), ("f", x_fake), ("i", x_interp)):
            xd = x.detach().clone().requires_grad_(True)
            logits = D(xd)
            g = torch.autograd.grad(logits.sum(), xd, create_graph=False)[0]
            norms[key] = torch.sqrt(g.pow(2).flatten(1).sum(dim=1) + 1e-12).detach()

    stats = {}
    for key, n in norms.items():
        q = torch.quantile(
            n.float(), torch.tensor([0.1, 0.5, 0.9], device=n.device)
        )
        stats[f"q10_n{key}"] = float(q[0])
        stats[f"med_n{key}"] = float(q[1])
        stats[f"q90_n{key}"] = float(q[2])
    return stats
