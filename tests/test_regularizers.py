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


# -------------------------
#  8) Dual-norm variants differentiate correctly
# -------------------------

def test_finite_differences_dual_norms():
    print("\n[8] finite-difference check for the dual norms (h=1e-6, float64)")
    h = 1e-6

    # linf's max() is only differentiable where the argmax is unique. A random
    # D on random inputs makes ties a measure-zero event, but the FD probe also
    # has to stay clear of them, so we assert the margin is comfortably wider
    # than h before trusting the comparison.
    # kappa is chosen per norm so the cap is actually active: ||g||_inf on this
    # fixture is ~0.3, well under the 0.5 the L2/L1 cases use.
    for arm, norm, kappa, tol in (
        ("b_cap", "l1", 0.5, 1e-4),
        ("b_cap", "linf", 0.1, 1e-3),
        ("c_eikonal", "l1", 0.5, 1e-4),
    ):
        D, x_real, x_fake = make_setup()
        reg = GradRegularizer(arm=arm, coeff=0.02, kappa=kappa, norm=norm)

        if norm == "linf":
            gaps = []
            for x in (x_real, x_fake):
                xd = x.detach().clone().requires_grad_(True)
                g = torch.autograd.grad(D(xd).sum(), xd)[0].abs()
                top2 = g.flatten(1).sort(dim=1, descending=True).values
                gaps.append(float((top2[:, 0] - top2[:, 1]).min()))
            print(f"    linf argmax margin: {min(gaps):.3e} (>> h={h:.0e})")
            assert min(gaps) > 1e-3, "argmax too close to a tie for a clean FD check"

        pen, _ = penalty_value(reg, D, x_real, x_fake)
        assert pen.item() > 0.0, f"{arm}/{norm}: need an active penalty to test"
        D.zero_grad(set_to_none=True)
        pen.backward()
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
                denom = max(abs(ana), abs(fd), 1e-8)
                worst = max(worst, abs(fd - ana) / denom)

        print(f"    {arm:10s} norm={norm:5s} pen={pen.item():.8e}  max rel err = {worst:.3e}")
        assert worst < tol, f"{arm}/{norm}: finite-difference mismatch, max rel err {worst}"

    # The three norms genuinely differ: ||g||_inf <= ||g||_2 <= ||g||_1 in 2D.
    D, x_real, x_fake = make_setup()
    norms = {}
    for norm in GradRegularizer.NORMS:
        reg = GradRegularizer(arm="c_eikonal", coeff=0.02, norm=norm)
        norms[norm] = float(reg._grad_norm(D, x_real).mean())
    print("    mean ||g||: " + "  ".join(f"{k}={v:.4f}" for k, v in norms.items()))
    assert norms["linf"] <= norms["l2"] <= norms["l1"] + 1e-12
    print("    OK: l1 and linf gradients are exact and properly ordered")


# -------------------------
#  9) Annealed penalty center
# -------------------------

def test_target_anneal_schedule():
    print("\n[9] target_anneal schedules the penalty center")
    total = 1000
    coeff = 0.02
    kappa = 0.5

    lin = GradRegularizer("c_eikonal", coeff, target_anneal="linear", total_steps=total)
    assert lin.center(0) == 1.0
    assert abs(lin.center(500) - 0.5) < 1e-12
    assert lin.center(total) == 0.0
    print(f"    linear c_eikonal: c(0)={lin.center(0)} c(500)={lin.center(500)} "
          f"c(1000)={lin.center(total)}")

    # c0 = kappa for b_cap, not 1.0.
    lin_cap = GradRegularizer(
        "b_cap", coeff, kappa=kappa, target_anneal="linear", total_steps=total
    )
    assert lin_cap.center(0) == kappa
    assert abs(lin_cap.center(500) - kappa / 2.0) < 1e-12
    assert lin_cap.center(total) == 0.0
    print(f"    linear b_cap:     c(0)={lin_cap.center(0)} c(500)={lin_cap.center(500)} "
          f"c(1000)={lin_cap.center(total)}")

    # At the end of the anneal the eikonal penalty *is* the zero-centered one.
    D, x_real, x_fake = make_setup()
    pen_end, stats_end = penalty_value(lin, D, x_real, x_fake, step=total)
    assert stats_end["center"] == 0.0
    plain = GradRegularizer("c_eikonal", coeff)
    n_r = plain._grad_norm(D, x_real)
    n_f = plain._grad_norm(D, x_fake)
    ref_zero = (coeff / 2.0) * (n_r.pow(2).mean() + n_f.pow(2).mean())
    print(f"    c_eikonal @ step=1000: {pen_end.item():.12e}  "
          f"mean((n-0)^2) form: {ref_zero.item():.12e}")
    assert abs(pen_end.item() - ref_zero.item()) < 1e-14
    # ...and therefore agrees with a_r1r2 up to the 1e-12 inside the sqrt.
    pen_r1r2, _ = penalty_value(GradRegularizer("a_r1r2", coeff), D, x_real, x_fake)
    assert abs(pen_end.item() - pen_r1r2.item()) < 1e-10
    print(f"    a_r1r2 on the same D:  {pen_r1r2.item():.12e}  -> the arms have merged")

    # 'delayed' holds, then ramps.
    dly = GradRegularizer("c_eikonal", coeff, target_anneal="delayed", total_steps=total)
    assert dly.center(0) == 1.0
    assert dly.center(599) == 1.0, "delayed must hold c0 right up to 0.6 * total_steps"
    assert dly.center(600) == 1.0
    prev = dly.center(600)
    for step in range(601, total + 1):
        c = dly.center(step)
        assert c < prev, f"delayed center not decreasing at step {step}"
        prev = c
    assert dly.center(total) == 0.0
    assert abs(dly.center(800) - 0.5) < 1e-12
    print(f"    delayed: c(599)={dly.center(599)} c(600)={dly.center(600)} "
          f"c(800)={dly.center(800)} c(1000)={dly.center(total)}")

    # A schedule is a no-op for a_r1r2, which has no center to move.
    r1 = GradRegularizer("a_r1r2", coeff, target_anneal="linear", total_steps=total)
    assert r1.center(0) == 0.0 and r1.center(total) == 0.0
    p0, _ = penalty_value(r1, D, x_real, x_fake, step=0)
    p1, _ = penalty_value(r1, D, x_real, x_fake, step=total)
    assert p0.item() == p1.item()
    print("    OK: centers follow the schedule; a_r1r2 is unaffected")


# -------------------------
#  10) Defaults are bit-identical to the pre-change module
# -------------------------

# Penalties from the committed (pre-dual-norm, pre-anneal) module, on
# make_setup()'s fixture with coeff=0.02, kappa=0.5, step=0, seed=0.
LEGACY_PENALTIES = {
    "a_r1r2": 0.0021669679779030312,
    "b_cap": 6.346602607057905e-09,
    "c_eikonal": 0.009918168926157385,
    "d_asym": 0.0024795422315393462,
    "e_interp": 0.00847055087350632,
    "f_none": 0.0,
}


def test_defaults_match_legacy():
    print("\n[10] default kwargs reproduce the pre-change module exactly")
    for arm, expected in LEGACY_PENALTIES.items():
        D, x_real, x_fake = make_setup()
        reg = GradRegularizer(arm=arm, coeff=0.02, kappa=0.5)
        assert reg.norm == "l2" and reg.target_anneal == "none"
        pen, stats = penalty_value(reg, D, x_real, x_fake)
        print(f"    {arm:10s} legacy={expected!r:26s} now={pen.item()!r}")
        assert pen.item() == expected, f"{arm}: {pen.item()!r} != legacy {expected!r}"
        if arm != "f_none":
            # The centers the legacy code hardcoded.
            assert stats["center"] == (0.0 if arm == "a_r1r2" else (0.5 if arm == "b_cap" else 1.0))
    print("    OK: every arm is bit-identical under the new defaults")


# -------------------------
#  11) New argument validation
# -------------------------

def test_new_arg_validation():
    print("\n[11] norm / target_anneal validation")
    for norm in ("l1", "linf"):
        try:
            GradRegularizer(arm="a_r1r2", coeff=0.02, norm=norm)
        except ValueError as e:
            print(f"    OK: a_r1r2 + norm={norm} -> ValueError({e})")
        else:
            raise AssertionError(f"expected ValueError for a_r1r2 with norm={norm}")

    try:
        GradRegularizer(arm="b_cap", coeff=0.02, norm="l3")
    except ValueError as e:
        print(f"    OK: unknown norm -> ValueError({e})")
    else:
        raise AssertionError("expected ValueError for an unknown norm")

    for anneal in ("linear", "delayed"):
        try:
            GradRegularizer(arm="c_eikonal", coeff=0.02, target_anneal=anneal)
        except ValueError as e:
            print(f"    OK: target_anneal={anneal} without total_steps -> ValueError({e})")
        else:
            raise AssertionError(f"expected ValueError for {anneal} without total_steps")

    try:
        GradRegularizer(arm="c_eikonal", coeff=0.02, target_anneal="cosine", total_steps=10)
    except ValueError as e:
        print(f"    OK: unknown target_anneal -> ValueError({e})")
    else:
        raise AssertionError("expected ValueError for an unknown target_anneal")

    # a_r1r2 with the (default) l2 norm stays legal.
    GradRegularizer(arm="a_r1r2", coeff=0.02, norm="l2")
    print("    OK: a_r1r2 + norm='l2' still constructs")


if __name__ == "__main__":
    test_finite_differences()
    test_r1r2_equivalence()
    test_b_cap_threshold()
    test_eikonal_forbids_flat_D()
    test_lazy()
    test_grad_norm_stats()
    test_bad_arm()
    test_finite_differences_dual_norms()
    test_target_anneal_schedule()
    test_defaults_match_legacy()
    test_new_arg_validation()
    print("\nAll grad-regularizer tests passed.")
