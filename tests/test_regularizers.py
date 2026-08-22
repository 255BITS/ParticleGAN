"""
Correctness tests for lib/grad_regularizers.py.

Everything runs in float64 on CPU so that central finite differences are a
meaningful reference. Run with:

    .venv/bin/python tests/test_regularizers.py
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.grad_regularizers import GradRegularizer, grad_norm_stats  # noqa: E402


SEED = 1234
BATCH = 8
IN_DIM = 2


class TinyFourierD(nn.Module):
    """
    Float64 stand-in for SimpleMLPDiscriminator in examples/100gaussians.py:
    same sin/cos features at frequencies pi * 2^i, just smaller and with tanh
    instead of LeakyReLU so the finite-difference check sees no kinks.
    """

    def __init__(self, in_dim: int = IN_DIM, hidden_dim: int = 16, n_hidden: int = 2, fourier: int = 2) -> None:
        super().__init__()
        self.fourier = fourier
        dim = in_dim + (2 * fourier * in_dim if fourier > 0 else 0)
        if fourier > 0:
            freqs = torch.pi * (2.0 ** torch.arange(fourier, dtype=torch.float64))
            self.register_buffer("freqs", freqs)
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.Tanh())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.fourier > 0:
            xf = x.unsqueeze(-1) * self.freqs  # (B, in_dim, K)
            h = torch.cat([h, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)
        return self.net(h).squeeze(-1)


def make_setup(seed: int = SEED):
    """Fresh D + (x_real, x_fake) batch, all float64 on CPU."""
    torch.manual_seed(seed)
    D = TinyFourierD().double()
    x_real = torch.randn(BATCH, IN_DIM, dtype=torch.float64)
    x_fake = torch.randn(BATCH, IN_DIM, dtype=torch.float64) + 0.5
    return D, x_real, x_fake


def penalty_value(reg, D, x_real, x_fake, step=0, seed=0):
    """Deterministic penalty (re-seeds so e_interp's eps is reproducible)."""
    torch.manual_seed(seed)
    pen, stats = reg.penalty(D, x_real, x_fake, step)
    return pen, stats


# -------------------------
#  1) Finite-difference gradient check
# -------------------------

def test_finite_differences():
    print("\n[1] finite-difference check (h=1e-6, float64)")
    h = 1e-6
    worst_overall = 0.0

    for arm in GradRegularizer.ARMS:
        if arm == "f_none":
            continue
        D, x_real, x_fake = make_setup()
        reg = GradRegularizer(arm=arm, coeff=0.02, kappa=0.5)

        pen, _ = penalty_value(reg, D, x_real, x_fake)
        D.zero_grad(set_to_none=True)
        pen.backward()
        # The final bias shifts D by a constant, so it has no effect on
        # grad_x D and autograd hands back None for it.
        analytic = {
            name: (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
            for name, p in D.named_parameters()
        }

        worst = 0.0
        for name, p in D.named_parameters():
            flat = p.data.view(-1)
            for i in range(flat.numel()):
                orig = flat[i].item()

                flat[i] = orig + h
                p_plus, _ = penalty_value(reg, D, x_real, x_fake)
                flat[i] = orig - h
                p_minus, _ = penalty_value(reg, D, x_real, x_fake)
                flat[i] = orig

                fd = (p_plus.item() - p_minus.item()) / (2.0 * h)
                ana = analytic[name].view(-1)[i].item()
                denom = max(abs(ana), abs(fd), 1e-8)  # abs floor for tiny grads
                rel = abs(fd - ana) / denom
                worst = max(worst, rel)

        worst_overall = max(worst_overall, worst)
        print(f"    {arm:10s} pen={pen.item():.8e}  max rel err = {worst:.3e}")
        assert worst < 1e-4, f"{arm}: finite-difference mismatch, max rel err {worst}"

    print(f"    OK: all arms differentiate correctly (worst {worst_overall:.3e} < 1e-4)")


# -------------------------
#  2) Baseline equivalence
# -------------------------

def test_r1r2_equivalence():
    print("\n[2] a_r1r2 == inline R1+R2 from examples/100gaussians.py")
    D, x_real, x_fake = make_setup()
    r1_gamma = 0.02

    # Inline repo formula.
    xr = x_real.detach().clone().requires_grad_(True)
    xf = x_fake.detach().clone().requires_grad_(True)
    grad_real = torch.autograd.grad(D(xr).sum(), xr, create_graph=True)[0]
    grad_fake = torch.autograd.grad(D(xf).sum(), xf, create_graph=True)[0]
    r1 = grad_real.pow(2).sum(dim=1).mean()
    r2 = grad_fake.pow(2).sum(dim=1).mean()
    ref = (r1_gamma / 2.0) * (r1 + r2)

    reg = GradRegularizer(arm="a_r1r2", coeff=r1_gamma)
    pen, stats = penalty_value(reg, D, x_real, x_fake)

    diff = abs(pen.item() - ref.item())
    print(f"    ref={ref.item():.12e}  arm={pen.item():.12e}  |diff|={diff:.3e}")
    assert diff < 1e-10, f"a_r1r2 does not reproduce the baseline: {diff}"
    assert stats["applied"] is True
    print("    OK: coeff=0.02 reproduces r1_gamma=0.02 exactly")


# -------------------------
#  3) b_cap is free below kappa
# -------------------------

def test_b_cap_threshold():
    print("\n[3] b_cap: inactive below kappa, active above")
    kappa = 1.0
    reg = GradRegularizer(arm="b_cap", coeff=0.02, kappa=kappa)

    D, x_real, x_fake = make_setup()
    with torch.no_grad():
        D.net[-1].weight.mul_(1e-3)
        D.net[-1].bias.mul_(1e-3)
    stats_small = grad_norm_stats(D, x_real, x_fake)
    pen_small, _ = penalty_value(reg, D, x_real, x_fake)
    print(f"    scaled down: med||g||_real={stats_small['med_nr']:.3e}  pen={pen_small.item():.3e}")
    assert stats_small["med_nr"] < kappa
    assert pen_small.item() < 1e-12, f"expected ~0 penalty below kappa, got {pen_small.item()}"

    with torch.no_grad():
        D.net[-1].weight.mul_(1e6)
        D.net[-1].bias.mul_(1e6)
    stats_big = grad_norm_stats(D, x_real, x_fake)
    pen_big, _ = penalty_value(reg, D, x_real, x_fake)
    print(f"    scaled up:   med||g||_real={stats_big['med_nr']:.3e}  pen={pen_big.item():.3e}")
    assert stats_big["med_nr"] > kappa
    assert pen_big.item() > 0.0, "expected positive penalty above kappa"
    print("    OK: one-sided cap only bites above kappa")


# -------------------------
#  4) The eikonal penalty forbids a flat D
# -------------------------

def test_eikonal_forbids_flat_D():
    print("\n[4] c_eikonal on a constant (flat) discriminator")
    coeff = 0.02
    D, x_real, x_fake = make_setup()
    with torch.no_grad():
        D.net[-1].weight.zero_()
        D.net[-1].bias.zero_()

    reg = GradRegularizer(arm="c_eikonal", coeff=coeff)
    pen, _ = penalty_value(reg, D, x_real, x_fake)

    # Flat D => n = 0 at reals and fakes => phi = (0-1)^2 = 1 on both sides
    # => penalty = (coeff/2) * (1 + 1) = coeff.  (The 1e-12 inside the sqrt
    # shifts n to 1e-6, hence the 1e-5 tolerance.)
    print(f"    pen={pen.item():.12f}  coeff={coeff}")
    assert abs(pen.item() - coeff) < 1e-5 * coeff + 1e-10, f"expected {coeff}, got {pen.item()}"

    # The zero-centered baseline is perfectly happy with the same flat D.
    pen_r1r2, _ = penalty_value(GradRegularizer("a_r1r2", coeff), D, x_real, x_fake)
    assert pen_r1r2.item() < 1e-20
    print(f"    a_r1r2 on the same flat D: pen={pen_r1r2.item():.3e}")
    print("    OK: the eikonal penalty FORBIDS a flat discriminator "
          "(it costs a full `coeff`), while R1+R2 rewards it with exactly zero.")


# -------------------------
#  5) Lazy regularization
# -------------------------

def test_lazy():
    print("\n[5] lazy_k semantics")
    lazy_k = 16
    D, x_real, x_fake = make_setup()

    lazy = GradRegularizer(arm="c_eikonal", coeff=0.02, lazy_k=lazy_k)
    eager = GradRegularizer(arm="c_eikonal", coeff=0.02, lazy_k=1)

    pen_skip, stats_skip = penalty_value(lazy, D, x_real, x_fake, step=1)
    assert stats_skip["applied"] is False
    assert pen_skip.item() == 0.0
    assert pen_skip.requires_grad is False
    assert pen_skip.device == x_real.device and pen_skip.dtype == x_real.dtype
    print(f"    step=1  applied={stats_skip['applied']} pen={pen_skip.item():.3e}")

    for step in (0, lazy_k):
        pen_on, stats_on = penalty_value(lazy, D, x_real, x_fake, step=step)
        pen_ref, _ = penalty_value(eager, D, x_real, x_fake, step=step)
        assert stats_on["applied"] is True
        assert pen_on.item() > 0.0
        ratio = pen_on.item() / pen_ref.item()
        print(f"    step={step:<3d} applied=True pen={pen_on.item():.8e} "
              f"= {ratio:.10f}x the every-step value")
        assert abs(ratio - lazy_k) < 1e-9, f"expected {lazy_k}x, got {ratio}"

    # f_none is always off.
    pen_none, stats_none = penalty_value(GradRegularizer("f_none", 0.02), D, x_real, x_fake, step=0)
    assert stats_none["applied"] is False and pen_none.item() == 0.0
    print("    OK: skipped steps return a detached zero; applied steps scale by lazy_k")


# -------------------------
#  6) grad_norm_stats under no_grad
# -------------------------

def test_grad_norm_stats():
    print("\n[6] grad_norm_stats inside torch.no_grad()")
    D, x_real, x_fake = make_setup()
    keys = {
        "med_nr", "med_nf", "med_ni",
        "q10_nr", "q90_nr", "q10_nf", "q90_nf", "q10_ni", "q90_ni",
    }
    D.eval()
    with torch.no_grad():
        stats = grad_norm_stats(D, x_real, x_fake)
    assert set(stats.keys()) == keys, f"key mismatch: {set(stats.keys()) ^ keys}"
    for k, v in stats.items():
        assert isinstance(v, float) and math.isfinite(v), f"{k} = {v}"
    for tag in ("r", "f", "i"):
        assert stats[f"q10_n{tag}"] <= stats[f"med_n{tag}"] <= stats[f"q90_n{tag}"]
    print("    " + "  ".join(f"{k}={stats[k]:.4f}" for k in sorted(keys)))
    assert x_real.grad is None and x_fake.grad is None
    print("    OK: finite floats, ordered quantiles, inputs untouched")


# -------------------------
#  7) Bad arm name
# -------------------------

def test_bad_arm():
    print("\n[7] arm validation")
    try:
        GradRegularizer(arm="z_nope", coeff=0.02)
    except ValueError as e:
        print(f"    OK: ValueError({e})")
    else:
        raise AssertionError("expected ValueError for an unknown arm")


if __name__ == "__main__":
    test_finite_differences()
    test_r1r2_equivalence()
    test_b_cap_threshold()
    test_eikonal_forbids_flat_D()
    test_lazy()
    test_grad_norm_stats()
    test_bad_arm()
    print("\nAll grad-regularizer tests passed.")
