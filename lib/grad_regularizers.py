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
    """

    ARMS = ("a_r1r2", "b_cap", "c_eikonal", "d_asym", "e_interp", "f_none")

    def __init__(
        self,
        arm: str,
        coeff: float,
        kappa: float = 1.0,
        lazy_k: int = 1,
    ) -> None:
        if arm not in self.ARMS:
            raise ValueError(f"Unknown grad regularizer arm: {arm} (expected one of {self.ARMS})")

        self.arm = arm
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)

        if self.lazy_k < 1:
            raise ValueError(f"lazy_k must be >= 1, got {lazy_k}")

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
            stats: {'applied': bool, 'pen': float}
        """
        skip = self.arm == "f_none" or (self.lazy_k > 1 and step % self.lazy_k != 0)
        if skip:
            zero = torch.zeros((), device=x_real.device, dtype=x_real.dtype)
            return zero, {"applied": False, "pen": 0.0}

        # Lazy regularization: fewer applications, proportionally bigger hits,
        # so the time-averaged pressure on D is unchanged.
        coeff_eff = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff

        if self.arm == "e_interp":
            pen = coeff_eff * self._interp_term(D, x_real, x_fake)
        else:
            n_r = self._grad_norm(D, x_real, squared=(self.arm == "a_r1r2"))
            n_f = self._grad_norm(D, x_fake, squared=(self.arm == "a_r1r2"))
            phi = self._phi
            pen = (coeff_eff / 2.0) * (phi(n_r).mean() + phi(n_f).mean())

        return pen, {"applied": True, "pen": float(pen.detach())}

    # -------------------------
    #  Penalty kernels
    # -------------------------

    def _phi(self, n: torch.Tensor) -> torch.Tensor:
        """Per-sample penalty on the gradient norm (or its square, for a_r1r2)."""
        if self.arm == "a_r1r2":
            # `n` is already ||g||^2 here: no sqrt, no epsilon, so this is
            # bit-for-bit the inline R1/R2 formula in examples/100gaussians.py.
            return n
        if self.arm == "b_cap":
            return F.relu(n - self.kappa).pow(2)
        if self.arm == "c_eikonal":
            return (n - 1.0).pow(2)
        if self.arm == "d_asym":
            # Steepness costs full price; flatness costs a quarter. Keeps the
            # "no flat D" property while letting D relax between the modes.
            return F.relu(n - 1.0).pow(2) + 0.25 * F.relu(1.0 - n).pow(2)
        raise ValueError(f"Unknown grad regularizer arm: {self.arm}")

    def _interp_term(
        self,
        D: torch.nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
    ) -> torch.Tensor:
        """E[(||grad D(x_i)|| - 1)^2] on per-sample real/fake interpolates."""
        eps = torch.rand(
            x_real.shape[0], 1, device=x_real.device, dtype=x_real.dtype
        )
        x_i = eps * x_real.detach() + (1.0 - eps) * x_fake.detach()
        n_i = self._grad_norm(D, x_i, squared=False)
        return (n_i - 1.0).pow(2).mean()

    @staticmethod
    def _grad_norm(
        D: torch.nn.Module,
        x: torch.Tensor,
        squared: bool = False,
    ) -> torch.Tensor:
        """
        ||grad_x D(x)|| per sample, with create_graph=True so the penalty is
        differentiable w.r.t. D's parameters.

        `squared` returns ||g||^2 without the sqrt/epsilon (the R1/R2 form).
        """
        x = x.detach().clone().requires_grad_(True)
        logits = D(x)
        g = torch.autograd.grad(logits.sum(), x, create_graph=True)[0]
        sq = g.pow(2).flatten(1).sum(dim=1)
        if squared:
            return sq
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
