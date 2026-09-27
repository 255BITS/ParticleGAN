"""Known-value tests for the no-R1 terms and settling mechanisms (nr_adapter.py).

Run: /home/martyn/dev/ParticleGAN/.venv/bin/python -m pytest -q reports/k3p-no-r1/test_nr_terms.py
"""
import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nr_adapter as nr  # noqa: E402
from particlegan import get_recipe  # noqa: E402
from particlegan.grad_regularizers import GradientPenalty  # noqa: E402
from particlegan.k3p import CriticAnchor, CriticSpikeGuard  # noqa: E402

D_IN = 4


class Linear(nn.Module):
    """D(x) = w . x (+ b): input gradient w everywhere, RMS slope ||w|| / sqrt(d)."""

    def __init__(self, w, b=0.0):
        super().__init__()
        self.w = nn.Parameter(torch.as_tensor(w, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(float(b), dtype=torch.float64))

    def forward(self, x):
        return x.flatten(1) @ self.w + self.b


def linear_with_rms_slope(slope, d=D_IN):
    return Linear(torch.full((d,), float(slope), dtype=torch.float64))  # ||w||/sqrt(d) = slope


def batch(n=6, d=D_IN, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(n, d, generator=g, dtype=torch.float64)


@pytest.fixture
def nr_state(monkeypatch):
    def set_arm(spike, settle):
        monkeypatch.setitem(nr.NR, "spike", spike)
        monkeypatch.setitem(nr.NR, "settle", settle)
        monkeypatch.setitem(nr.NR, "kernels", [])
        monkeypatch.setitem(nr.NR, "sums", {})
        monkeypatch.setitem(nr.NR, "critic_opts", [])
        for key in ("oadam_by_opt", "adam_by_opt", "guard_reads"):
            monkeypatch.setitem(nr.NR, key, {})
        for key in ("r1_evals", "stock_calls", "pairsec_rows", "pairsec_no_neighbor", "oadam_steps"):
            monkeypatch.setitem(nr.NR, key, 0)
    set_arm("none", "none")
    return set_arm


# ------------------------------------------------------------------ spike controls
@pytest.mark.parametrize("slope,expected", [(0.5, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 4.0)])
def test_symcap_zero_below_slope_one_positive_above(slope, expected):
    D, x = linear_with_rms_slope(slope), batch()
    val = nr.spike_term("symcap", D, x, batch(seed=1), kappa=1.0)
    assert float(val) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("slope,expected", [(0.5, 0.0), (2.0, 1.0)])
def test_pathcap_linear_known_values(slope, expected):
    D, r, f = linear_with_rms_slope(slope), batch(), batch(seed=1)
    u = torch.rand(len(r), generator=torch.Generator().manual_seed(3))
    assert float(nr.spike_term("pathcap", D, r, f, 1.0, u)) == pytest.approx(expected, abs=1e-6)


def test_pathcap_evaluates_on_the_segment():
    """Quadratic D = a/2 ||x||^2 has grad a x: the cap must see ||a x_hat|| with x_hat = r + u (f - r)."""
    a = 3.0

    class Quad(nn.Module):
        def forward(self, x):
            return 0.5 * a * x.flatten(1).pow(2).sum(1)
    r, f = batch(), batch(seed=1)
    u = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 0.1], dtype=torch.float64)
    x_hat = r + u[:, None] * (f - r)
    expected = ((a * x_hat.norm(dim=1) / D_IN ** 0.5 - 1).relu() ** 2).mean()
    assert float(nr.spike_term("pathcap", Quad(), r, f, 1.0, u)) == pytest.approx(float(expected), rel=1e-6)
    assert torch.allclose(nr.path_points(r, f, u), x_hat)
    # u = 0 is the real, u = 1 the fake
    assert torch.equal(nr.path_points(r, f, torch.zeros(6, dtype=torch.float64)), r)
    assert torch.allclose(nr.path_points(r, f, torch.ones(6, dtype=torch.float64)), f)


def test_pairsec_equals_analytic_secant_for_linear_D():
    w = torch.tensor([2.0, -1.0, 0.5, 3.0], dtype=torch.float64)
    D, x = Linear(w, b=0.7), batch(n=8)
    slopes = nr.secant_slopes(D, x)
    dist = torch.cdist(x, x)
    dist.fill_diagonal_(float("inf"))
    j = dist.argmin(1)
    diff = x - x[j]
    analytic = (diff @ w).abs() / diff.norm(dim=1)
    assert torch.allclose(slopes, analytic, rtol=1e-10)
    kappa = 1.0
    expected = ((analytic / D_IN ** 0.5 - kappa).relu() ** 2).mean()
    assert float(nr.spike_term("pairsec", D, x, x, kappa)) == pytest.approx(float(expected), rel=1e-10)


def test_pairsec_parallel_pair_gives_gradient_norm_and_no_double_backprop():
    w = torch.tensor([3.0, 4.0, 0.0, 0.0], dtype=torch.float64)  # ||w|| = 5, RMS 2.5
    D = Linear(w)
    x = torch.stack([torch.zeros(4, dtype=torch.float64), 0.1 * w / 5.0]).double()  # the pair's difference is along w
    assert torch.allclose(nr.secant_slopes(D, x), torch.full((2,), 5.0, dtype=torch.float64))
    val = nr.spike_term("pairsec", D, x, x, 1.0)
    assert float(val) == pytest.approx((2.5 - 1.0) ** 2)
    # the term is a function of D's outputs only: it backpropagates without create_graph input gradients
    val.backward()
    assert D.w.grad is not None and torch.isfinite(D.w.grad).all()


def test_pairsec_all_duplicate_reals_is_inert_and_counted(nr_state):
    """toy-unused_token_hold feeds 8 identical reals: no distinct neighbour, so the term is 0 by definition (not a
    1e12-amplified logit round-off) and every row is counted as dropped."""
    x = batch(n=1).expand(8, D_IN).clone()

    class Jitter(nn.Module):  # identical inputs, logits differing by round-off
        def forward(self, x):
            return x.sum(1) * 5.0 + 1e-7 * torch.arange(len(x), dtype=x.dtype)
    val = nr.spike_term("pairsec", Jitter(), x, x, 1.0)
    assert float(val) == 0.0
    assert nr.NR["pairsec_no_neighbor"] == 8 and nr.NR["pairsec_rows"] == 8


def test_pairsec_skips_duplicates_to_the_nearest_distinct_real(nr_state):
    w = torch.tensor([2.0, -1.0, 0.5, 3.0], dtype=torch.float64)
    D, base = Linear(w), batch(n=5)
    x = torch.cat([base, base[:2]])  # rows 5, 6 duplicate rows 0, 1
    slopes = nr.secant_slopes(D, x)
    assert len(slopes) == 7 and nr.NR["pairsec_no_neighbor"] == 0
    dist = torch.cdist(x, x)
    dist[dist <= 1e-6] = float("inf")
    j = dist.argmin(1)
    diff = x - x[j]
    assert torch.allclose(slopes, (diff @ w).abs() / diff.norm(dim=1), rtol=1e-10)
    # one real plus its duplicate and nothing else: dropped, not divided by ~0
    y = torch.cat([base[:1], base[:1], base[1:2] + 100.0])
    assert len(nr.secant_slopes(D, y)) == 3  # each duplicate's nearest distinct real is the far row
    z = torch.cat([base[:1], base[:1]])
    assert float(nr.pairsec(D, z, 1.0)) == 0.0 and nr.NR["pairsec_no_neighbor"] == 2


def test_pairsec_below_slope_one_is_zero():
    assert float(nr.spike_term("pairsec", linear_with_rms_slope(0.9), batch(), batch(), 1.0)) == 0.0


def test_dvalcap_zero_within_one_of_mean_positive_outside():
    ident = lambda x: x[:, 0]  # noqa: E731  logits = first coordinate
    x = torch.zeros(4, D_IN, dtype=torch.float64)
    x[:, 0] = torch.tensor([10.0, 10.5, 9.2, 10.3])  # mean 10.0, all within 1; the offset (shift invariance) is ignored
    assert float(nr.dvalcap(ident, x)) == 0.0
    x[:, 0] = torch.tensor([0.0, 0.0, 0.0, 4.0])  # mean 1: |dev| = 1,1,1,3 -> relu(dev-1)^2 = 0,0,0,4
    assert float(nr.spike_term("dvalcap", ident, x, x, 1.0)) == pytest.approx(1.0)


def test_spike_none_is_exact_zero():
    assert float(nr.spike_term("none", linear_with_rms_slope(5.0), batch(), batch(), 1.0)) == 0.0


# ------------------------------------------------------------------ hinge
def test_hinge_sign_and_value():
    r, f = torch.tensor([2.0, 0.0, 0.5], requires_grad=True), torch.tensor([0.0, 0.0, 0.0])
    loss = nr.hinge_d(r, f)
    assert float(loss) == pytest.approx((0.0 + 1.0 + 0.5) / 3)
    loss.backward()
    # raising a real logit (inside the margin) lowers the loss; past the margin it does nothing
    assert r.grad.tolist() == pytest.approx([0.0, -1 / 3, -1 / 3])


def test_hinge_reaches_both_loss_classes(nr_state):
    from benchmarks.legacy.gan_loss import GANLoss as LegacyLoss
    from particlegan.gan_loss import GANLoss as PkgLoss
    r, f = torch.tensor([0.3, 2.0]), torch.tensor([0.0, 0.0])
    nr_state("none", "hinge")
    legacy = nr._legacy_d_loss_factory(LegacyLoss.d_loss)
    for fn, obj in ((nr._pkg_d_loss, PkgLoss()), (legacy, LegacyLoss("logistic", "rp"))):
        assert float(fn(obj, r, f)) == pytest.approx(0.35)
        assert float(obj.g_loss(f, r)) == pytest.approx(float(torch.nn.functional.softplus(-(f - r)).mean()))
    nr_state("none", "none")  # non-hinge arms keep RpGAN softplus exactly
    assert torch.equal(nr._pkg_d_loss(PkgLoss(), r, f), torch.nn.functional.softplus(-(r - f)).mean())
    assert torch.equal(legacy(LegacyLoss("logistic", "rp"), r, f), torch.nn.functional.softplus(-(r - f)).mean())


# ------------------------------------------------------------------ R1 is gone in every arm
@pytest.mark.parametrize("arm", sorted(nr.NR_ARMS))
def test_r1_is_zero_in_every_nr_arm(arm, nr_state):
    spec = nr.NR_ARMS[arm]
    assert "penalty" not in spec and spec["nr"]["spike"] in nr.SPIKES and spec["nr"]["settle"] in nr.SETTLES
    nr_state(spec["nr"]["spike"], spec["nr"]["settle"])
    # Real gradient nonzero (RMS slope .5 < cap), fakes below the cap, logits within 1 of their mean:
    # every nr term is exactly 0, while K3P's R1 is .25 * c/2.
    D = linear_with_rms_slope(0.5)
    r, f = batch(scale=0.1), batch(seed=1, scale=0.1)
    anchor = CriticAnchor(D, copy.deepcopy(D)) if spec["nr"]["settle"] == "anchor" else None
    kernel = GradientPenalty(coeff=0.3, kappa=1.0, lr_floor=0.0, anchor=anchor,
                             anchor_weight=spec["recipe"].get("reg_anchor_weight", 0.0))
    stock, _ = kernel._k3p_penalty(D, r, f, 1, 0.3, False, None)
    assert float(stock) == pytest.approx(0.15 * 0.25)
    kernel = GradientPenalty(coeff=0.3, kappa=1.0, lr_floor=0.0, anchor=anchor,
                             anchor_weight=spec["recipe"].get("reg_anchor_weight", 0.0))
    nr._PATH_GEN.clear()
    pen, stats = nr.nr_penalty(kernel, D, r, f, 1, 0.3, True, None)
    assert float(pen) == 0.0 and stats["spike"] == 0.0 and stats["fake_cap"] == 0.0


def test_arms_file_is_the_full_factorial_on_gs2():
    assert len(nr.NR_ARMS) == 20
    names = {f"nr_{s}_{t}" for s in nr.SPIKES for t in nr.SETTLES}
    assert set(nr.NR_ARMS) == names
    base = nr.sa.ARMS["gs2_c03_lr2_d05"]
    released = get_recipe()
    for name, spec in nr.NR_ARMS.items():
        assert spec["config"] == base["config"]
        want = dict(base["recipe"])
        if spec["nr"]["settle"] == "anchor":
            want["reg_anchor_weight"] = released.reg_anchor_weight
            assert released.reg_anchor_weight > 0
        assert spec["recipe"] == want
        assert spec["nr"]["guard_buffer"] == "max_exp_avg_sq"  # every arm is AMSGrad: one guard buffer for all


# ------------------------------------------------------------------ anchor at constant LR
def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))


def _constant_lr_loop(penalty_fn, steps=30, anchor_weight=1.0):
    recipe = get_recipe(lr_floor=1.0, network_lr_floor=1.0, reg_anchor_weight=anchor_weight, amsgrad=True,
                        reg_coeff=0.3)
    D = _mlp()
    opt = recipe.make_critic_optimizer(D, ema_critic=copy.deepcopy(D))
    pen = recipe.make_critic_penalty(opt, collect_stats=True)
    g = torch.Generator().manual_seed(1)
    stats = []
    for _ in range(steps):
        r = torch.randn(64, 2, generator=g)
        f = torch.randn(64, 2, generator=g) + 1.0
        loss = torch.nn.functional.softplus(-(D(r) - D(f))).mean() + pen(D, r, f)
        opt.zero_grad()
        loss.backward()
        opt.step()
        stats.append(pen.last_stats)
    return opt, pen, stats


def test_anchor_is_active_at_constant_lr(nr_state, monkeypatch):
    # stock K3P at constant LR: s == 1 forever, the anchor never starts (the gating we bypass)
    opt, pen, stats = _constant_lr_loop(None)
    assert all(s["s"] == 1.0 for s in stats) and not opt.record.anchor_started
    ema0 = [p.clone() for p in opt.ema_critic.parameters()]
    # nr anchor arm
    nr_state("none", "anchor")
    monkeypatch.setattr(GradientPenalty, "_k3p_penalty", nr.nr_penalty)
    opt, pen, stats = _constant_lr_loop(None)
    assert all(s["s"] == 1.0 for s in stats)
    assert opt.record.anchor_started
    assert stats[0]["prox"] == 0.0 and all(s["prox"] > 0 for s in stats[2:])
    # the EMA moved (record_step advances it after every critic step) and lags the live critic
    live = list(opt.critic.parameters())
    ema = list(opt.ema_critic.parameters())
    assert any(not torch.equal(e, e0) for e, e0 in zip(ema, ema0))
    assert any(not torch.equal(e, p) for e, p in zip(ema, live))
    rec = nr.receipt({})
    assert rec["anchor_active"] and rec["r1_weight_applied"] == 0.0 and rec["penalty_patched"]


def test_r1_probe_counts_squared_norms_and_stock_penalty_is_a_sentinel(nr_state, monkeypatch):
    from benchmarks.legacy import grad_regularizers as legacy_gr
    monkeypatch.setattr(GradientPenalty, "_grad_norm", GradientPenalty.__dict__["_grad_norm"])
    monkeypatch.setattr(legacy_gr.GradientPenalty, "_grad_norm", legacy_gr.GradientPenalty._grad_norm)
    monkeypatch.setattr(GradientPenalty, "_k3p_penalty", GradientPenalty._k3p_penalty)
    monkeypatch.setattr(nr.sa, "_orig_k3p_penalty", nr.sa._orig_k3p_penalty)
    stock = GradientPenalty._k3p_penalty
    nr.install_r1_probe()
    D, r, f = linear_with_rms_slope(0.5), batch(scale=0.1), batch(seed=1, scale=0.1)
    kernel = GradientPenalty(coeff=0.3, kappa=1.0, lr_floor=0.0)
    for spike in nr.SPIKES:  # every nr term: no squared norm requested
        nr_state(spike, "none")
        nr._PATH_GEN.clear()
        nr.nr_penalty(kernel, D, r, f, 1, 0.3, False, None)
        assert nr.NR["r1_evals"] == 0
    stock(kernel, D, r, f, 1, 0.3, False, None)  # phase-A R1 is caught by the probe
    assert nr.NR["r1_evals"] == 1 and nr.receipt({})["r1_weight_applied"] == 1
    nr.sa._orig_k3p_penalty = nr._stock_penalty_sentinel
    with pytest.raises(RuntimeError):
        nr.sa._counted_k3p_penalty(kernel, D, r, f, 1, 0.3, False, None)
    assert nr.NR["stock_calls"] == 1 and nr.receipt({})["stock_k3p_penalty_calls"] == 1


# ------------------------------------------------------------------ oadam critic update + guard buffer
def test_oadam_dispatch_matches_optimistic_adam_amsgrad(nr_state, monkeypatch):
    nr_state("none", "oadam")
    monkeypatch.setattr(torch.optim.Adam, "step", torch.optim.Adam.step)  # restored after the test
    nr.install_oadam()
    recipe = get_recipe(lr_floor=1.0, network_lr_floor=1.0, reg_anchor_weight=0.0, amsgrad=True, d_guard_ratio=0.0,
                        lr=0.01, d_lr_mult=0.5, betas=(0.0, 0.999))
    D = _mlp()
    D_ref = copy.deepcopy(D)
    opt = recipe.make_critic_optimizer(D)
    nr.NR["critic_opts"].append(opt)
    ref = nr.OptimisticAdam(D_ref.parameters(), lr=0.005, betas=(0.0, 0.999), amsgrad=True)
    other = torch.optim.Adam(_mlp().parameters(), lr=0.01)  # not the critic: stock Adam update
    g = torch.Generator().manual_seed(2)
    for _ in range(12):
        x = torch.randn(32, 2, generator=g)
        for model, o in ((D, opt), (D_ref, ref)):
            o.zero_grad()
            model(x).square().mean().backward()
            o.step()
    for p, q in zip(D.parameters(), D_ref.parameters()):
        assert torch.equal(p, q)
    st = next(iter(opt.state.values()))
    assert {"prev_step", "max_exp_avg_sq"} <= set(st)
    other.zero_grad()
    list(other.param_groups[0]["params"])[0].sum().backward()
    other.step()
    assert "prev_step" not in next(iter(other.state.values()))
    # per-optimizer counts: the critic got every update from OAdam, the other Adam only stock steps
    assert nr.NR["oadam_by_opt"] == {id(opt): 12} and nr.NR["adam_by_opt"][id(other)] == 1
    assert id(opt) not in nr.NR["adam_by_opt"]
    assert nr._update_rule(12, 0, 12) == "lib.oadam.OptimisticAdam.step"
    assert nr._update_rule(0, 12, 12) == "torch.optim.Adam.step"
    assert nr._update_rule(11, 1, 12).startswith("unverified")


def test_dispatch_counts_stock_steps_in_non_oadam_arms(nr_state, monkeypatch):
    nr_state("none", "none")
    monkeypatch.setattr(torch.optim.Adam, "step", torch.optim.Adam.step)
    nr.install_oadam()
    D = _mlp()
    opt = torch.optim.Adam(D.parameters(), lr=0.01, amsgrad=True)
    nr.NR["critic_opts"].append(opt)
    for _ in range(3):
        opt.zero_grad()
        D(torch.ones(4, 2)).sum().backward()
        opt.step()
    assert nr.NR["adam_by_opt"] == {id(opt): 3} and nr.NR["oadam_by_opt"] == {}
    assert "prev_step" not in next(iter(opt.state.values()))


def test_guard_reads_max_buffer(nr_state):
    p = nn.Parameter(torch.zeros(10))
    opt = torch.optim.Adam([p], lr=0.1, betas=(0.0, 0.999), amsgrad=True)
    opt.state[p] = {"step": torch.tensor(500.0), "exp_avg": torch.zeros(10),
                    "exp_avg_sq": torch.full((10,), 1e-4), "max_exp_avg_sq": torch.full((10,), 1.0)}
    p.grad = torch.full((10,), 0.2)  # ratio vs exp_avg_sq ~ 20 (> 5, clipped); vs max ~ .2 (kept)
    guard = CriticSpikeGuard(5.0, 200)
    assert int(nr.guard_apply_max(guard, opt)) == 0 and torch.equal(p.grad, torch.full((10,), 0.2))
    assert int(CriticSpikeGuard.apply_(guard, opt)) == 1  # the stock guard reads exp_avg_sq
    # OAdam state (int step) works too
    opt.state[p]["step"] = 500
    p.grad = torch.full((10,), 20.0)
    assert int(nr.guard_apply_max(guard, opt)) == 1
    assert nr.NR["guard_reads"] == {id(opt): {"max_exp_avg_sq": 2}}  # the key actually read is tallied


# ------------------------------------------------------------------ leaderboard receipt gate
def _good_receipt(spike, settle):
    rule = "lib.oadam.OptimisticAdam.step" if settle == "oadam" else "torch.optim.Adam.step"
    return dict(spike=spike, settle=settle, penalty_patched=True, nr_penalty_calls=10, r1_evals=0,
                stock_k3p_penalty_calls=0, d_loss_calls={"rp_hinge" if settle == "hinge" else "rp_softplus": 10},
                oadam_steps=10 if settle == "oadam" else 0, oadam_steps_non_critic=0,
                critic_optimizers=[dict(update_rule=rule)], guard_reads={"max_exp_avg_sq": 40},
                anchor_active=settle == "anchor", pathcap_pair_mismatch=0, lr=dict(lr_constant=True))


@pytest.mark.parametrize("arm", sorted(nr.NR_ARMS))
def test_receipt_gate_accepts_engaged_arms(arm):
    import summarize_nr as sm
    spec = nr.NR_ARMS[arm]["nr"]
    assert sm.receipt_ok(arm, {"nr_receipt": _good_receipt(spec["spike"], spec["settle"])}) == []


@pytest.mark.parametrize("change", [
    dict(nr_penalty_calls=0), dict(r1_evals=3), dict(stock_k3p_penalty_calls=1), dict(oadam_steps=0),
    dict(d_loss_calls={"rp_softplus": 10}), dict(anchor_active=False), dict(pathcap_pair_mismatch=2),
    dict(lr=dict(lr_constant=False)), dict(nonfinite_terms=True), dict(guard_reads={"exp_avg_sq": 4}),
    dict(critic_optimizers=[dict(update_rule="torch.optim.Adam.step")]), dict(settle="none")])
def test_receipt_gate_rejects_unengaged_overrides(change):
    import summarize_nr as sm
    rec = _good_receipt("pathcap", "oadam") | change
    if "anchor_active" in change:
        rec = _good_receipt("pathcap", "anchor") | change
        assert sm.receipt_ok("nr_pathcap_anchor", {"nr_receipt": rec})
        return
    if "d_loss_calls" in change:
        rec = _good_receipt("pathcap", "hinge") | change
        assert sm.receipt_ok("nr_pathcap_hinge", {"nr_receipt": rec})
        return
    assert sm.receipt_ok("nr_pathcap_oadam", {"nr_receipt": rec})
    assert sm.receipt_ok("nr_pathcap_oadam", {}) == ["no nr_receipt"]
