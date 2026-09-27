"""K3P package components: the R1 + fake-cap critic penalty, guard and A2 (CPU, float64)."""
import copy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3p_scenarios as sc  # noqa: E402

from particlegan.k3p import CriticSpikeGuard, LatentRowDamping  # noqa: E402
from particlegan.grad_regularizers import CriticStepRecord, GradientPenalty  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

DRIVER = r'''
import sys, types
from pathlib import Path
repo, out = Path(sys.argv[1]), sys.argv[2]
src = repo / {sources!r}
sys.path[:0] = [str(repo), str(repo / "tests"), str(src)]
# The frozen mechanism patches the multi-arm penalty it was recorded against;
# that penalty now lives only in the benchmarks' pinned copy.
import benchmarks.legacy.grad_regularizers as legacy_penalty
sys.modules["particlegan.grad_regularizers"] = legacy_penalty
import torch
torch.set_num_threads(1)
text = "\n".join(l for l in (src / "response.py").read_text().splitlines() if "device.type=='cuda'" not in l)
response = types.ModuleType("response"); response.__file__ = str(src / "response.py")
exec(compile(text, response.__file__, "exec"), response.__dict__)
sys.modules["response"] = response
import mechanism, latent
mechanism.GUARD_MIN_STEPS = 3
import k3p_scenarios
torch.save(k3p_scenarios.frozen_all(mechanism, latent, response), out)
'''


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("frozen_k3p")
    driver = tmp / "driver.py"
    driver.write_text(DRIVER.format(sources=sc.FROZEN_SOURCES))
    out = tmp / "frozen.pt"
    subprocess.run([sys.executable, str(driver), str(REPO), str(out)], check=True, cwd=tmp)
    return torch.load(out, weights_only=False)


# ---------------------------------------------------------------- reference formula
# The shipped penalty (gs2_c03_lr2_d05 from the k3p-constant-suite), written out
# term by term: c/2 * [mean ||g_r||^2 / d + mean relu(||g_f|| / sqrt(d) - kappa)^2].
def _ref_penalty(D, xr, xf, coeff, kappa):
    d = xr[0].numel()
    x = xr.detach().clone().requires_grad_(True)
    g_r = torch.autograd.grad(D(x).sum(), x, create_graph=True)[0]
    y = xf.detach().clone().requires_grad_(True)
    g_f = torch.autograd.grad(D(y).sum(), y, create_graph=True)[0]
    r1 = g_r.pow(2).flatten(1).sum(1).mean() / d
    rms_f = torch.sqrt(g_f.pow(2).flatten(1).sum(1) + 1e-12) / d ** 0.5
    return coeff / 2.0 * (r1 + (rms_f - kappa).relu().square().mean())


def test_penalty_is_r1_plus_fake_cap():
    D = sc.make_critic()
    reg = GradientPenalty(coeff=0.3, kappa=0.2)
    xr, xf = sc.critic_batch(1)
    pen, st = reg.penalty(D, xr, xf, 1)
    ref = _ref_penalty(D, xr, xf, 0.3, 0.2)
    assert torch.allclose(pen, ref, rtol=1e-12)
    assert set(st) == {"applied", "pen", "center", "r1", "fake_cap"}
    assert st["r1"] > 0 and st["fake_cap"] > 0
    assert abs(st["pen"] - 0.15 * (st["r1"] + st["fake_cap"])) < 1e-12
    params = list(D.parameters())[:-1]  # the output bias does not move an input gradient
    grads = torch.autograd.grad(pen, params, retain_graph=True)
    ref_grads = torch.autograd.grad(ref, params)
    assert all(torch.allclose(a, b, rtol=1e-10, atol=1e-14) for a, b in zip(grads, ref_grads))


def test_fake_cap_is_one_sided_and_r1_is_zero_centred():
    D = nn.Linear(2, 1, dtype=sc.DT)
    with torch.no_grad():
        D.weight.copy_(torch.tensor([[0.3, -0.4]], dtype=sc.DT))  # slope RMS .5/sqrt(2) < kappa 1
    reg = GradientPenalty()
    xr, xf = sc.critic_batch(1)
    pen, st = reg.penalty(D, xr, xf, 1)
    assert st["fake_cap"] == 0.0  # below the cap: no fake-side penalty
    assert abs(st["r1"] - 0.25 / 2) < 1e-12  # ||w||^2 / d, pulled toward 0 however small
    assert torch.allclose(pen, torch.tensor(0.5 * 0.125, dtype=sc.DT), rtol=1e-12)


def test_penalty_draws_no_randomness():
    D = sc.make_critic()
    xr, xf = sc.critic_batch(3)
    state = torch.get_rng_state()
    a = GradientPenalty(kappa=0.1).penalty(D, xr, xf, 1)[0]
    b = GradientPenalty(kappa=0.1).penalty(D, xr, xf, 1)[0]
    assert torch.equal(a, b) and torch.equal(torch.get_rng_state(), state)


def test_lazy_penalty_applies_every_kth_step_with_k_times_the_coefficient():
    D = sc.make_critic()
    xr, xf = sc.critic_batch(4)
    once = GradientPenalty(coeff=0.3, kappa=0.2)
    lazy = GradientPenalty(coeff=0.3, kappa=0.2, lazy_k=4)
    skipped, st = lazy.penalty(D, xr, xf, 3)
    assert skipped.item() == 0.0 and st == {"applied": False, "pen": 0.0}
    full = once.penalty(D, xr, xf, 1)[0]
    hit = lazy.penalty(D, xr, xf, 4)[0]
    assert torch.allclose(hit, 4 * full, rtol=1e-14, atol=0)


def _assert_critic_equal(ref, new):
    assert len(ref["trace"]) == len(new["trace"])
    for a, b in zip(ref["trace"], new["trace"]):
        assert a["applied"] == b["applied"], a["t"]
        assert torch.equal(a["pen"], b["pen"]), a["t"]
        assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"])), a["t"]


@pytest.mark.parametrize("lazy_k", [1, 2])
def test_penalty_and_guard_match_the_frozen_k3p_at_constant_lr(frozen, lazy_k):
    """At a constant LR the frozen K3P never leaves phase A, which is the shipped penalty."""
    ref = frozen["critic" if lazy_k == 1 else "lazy"]
    assert all(row["s"] == 1.0 for row in ref["trace"]) and ref["anchor_started_call"] is None
    objs, new = sc.package_critic(lazy_k=lazy_k)
    _assert_critic_equal(ref, new)
    assert ref["clipped"] >= 1 and objs["guard"].clipped_tensors == ref["clipped"]  # the step-7 spike
    assert ref["calls"] == objs["reg"].state_dict()["calls"] == sc.STEPS // lazy_k


def _recipe_critic(lazy_k, split=None):
    """The same scenario through recipe.make_critic_optimizer / make_critic_penalty."""
    from particlegan import get_recipe
    recipe = get_recipe(reg_coeff=1.0, reg_kappa=0.5, reg_every=lazy_k, amsgrad=False,
                        d_lr_mult=1.0, lr=sc.LR0, d_guard_min_steps=3)

    def build():
        D = sc.make_critic()
        opt = recipe.make_critic_optimizer(D, foreach=False)
        return D, opt, recipe.make_critic_penalty(opt, collect_stats=True)

    def run(D, opt, penalty, steps, trace):
        for t in steps:
            xr, xf = sc.critic_batch(t)
            pen = penalty(D, xr, xf)
            loss = F.softplus(-D(xr)).mean() + F.softplus(D(xf)).mean()
            if t == sc.SPIKE_STEP:
                loss = loss * 1000.0
            opt.zero_grad(set_to_none=True)
            (loss + pen).backward()
            opt.step()
            trace.append(dict(t=t, pen=pen.detach().clone(), applied=penalty.last_stats["applied"],
                              params=[p.detach().clone() for p in D.parameters()]))
        return trace

    D, opt, penalty = build()
    trace = run(D, opt, penalty, range(1, (split or sc.STEPS) + 1), [])
    if split is not None:
        saved = copy.deepcopy(dict(D=D.state_dict(), opt=opt.state_dict()))
        D, opt, penalty = build()
        D.load_state_dict(saved["D"])
        opt.load_state_dict(saved["opt"])
        run(D, opt, penalty, range(split + 1, sc.STEPS + 1), trace)
    return opt, dict(trace=trace)


@pytest.mark.parametrize("lazy_k", [1, 2])
def test_recipe_optimizer_and_penalty_match_the_frozen_k3p(frozen, lazy_k):
    ref = frozen["critic" if lazy_k == 1 else "lazy"]
    opt, new = _recipe_critic(lazy_k)
    _assert_critic_equal(ref, new)
    assert opt.guard.clipped_tensors == ref["clipped"] >= 1
    # Resuming from the usual D / opt_d state_dicts after the spike is bit-exact.
    _, resumed = _recipe_critic(lazy_k, split=9)
    _assert_critic_equal(ref, resumed)


def test_latent_row_damping_matches_frozen_a2(frozen):
    ref = frozen["latent"]

    def package(split=None):
        prior, G, opt = sc.make_latent()
        damp = LatentRowDamping(prior.z, torch.zeros_like(prior.z))

        def step(o):
            with damp.around(o):
                o.step()
        if split is None:
            return sc.run_latent(prior, G, opt, range(1, 15), step), damp
        trace = sc.run_latent(prior, G, opt, range(1, split + 1), step)
        saved = copy.deepcopy(dict(prior=prior.state_dict(), G=G.state_dict(), opt=opt.state_dict(),
                                   damp=damp.state_dict(), history=damp.history))
        prior, G, opt = sc.make_latent(seed=99)
        prior.load_state_dict(saved["prior"]); G.load_state_dict(saved["G"]); opt.load_state_dict(saved["opt"])
        damp = LatentRowDamping(prior.z, saved["history"].clone())
        damp.load_state_dict(saved["damp"])
        return trace + sc.run_latent(prior, G, opt, range(split + 1, 15), step), damp

    new, damp = package()
    assert ref["scoped_calls"] > 0
    for a, b in zip(ref["trace"], new):
        assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"])), a["t"]
        assert torch.equal(a["exp_avg"], b["exp_avg"]), a["t"]
    assert damp.started and 0 < ref["scoped_calls"] < 9  # rate >= 1/2 switches scoping off on some sparse steps
    resumed, _ = package(split=7)
    for a, b in zip(new, resumed):
        assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"])), a["t"]


def test_two_critics_independent():
    from torch.optim import optimizer as optim_mod
    hooks = (len(optim_mod._global_optimizer_pre_hooks), len(optim_mod._global_optimizer_post_hooks))
    original_penalty = GradientPenalty.penalty
    alone, trace_alone = sc.package_critic()
    A = sc.package_critic(steps=[])[0]
    B = sc.package_critic(steps=[])[0]
    for t in range(1, sc.STEPS + 1):
        ra = sc.run_package_critic(A, [t])
        xr, xf = sc.critic_batch(t, seed=7)
        pen = B["reg"](B["D"], xr, xf, t)
        B["opt"].zero_grad(set_to_none=True)
        (F.softplus(B["D"](xf)).mean() + pen).backward()
        B["guard"].apply_(B["opt"])
        B["opt"].step()
        B["reg"].after_critic_step(B["opt"])
        assert torch.equal(ra["trace"][0]["pen"], trace_alone["trace"][t - 1]["pen"])
    assert all(torch.equal(x, y) for x, y in zip(A["D"].parameters(), alone["D"].parameters()))
    assert not all(torch.equal(x, y) for x, y in zip(A["D"].parameters(), B["D"].parameters()))
    assert GradientPenalty.penalty is original_penalty
    assert (len(optim_mod._global_optimizer_pre_hooks), len(optim_mod._global_optimizer_post_hooks)) == hooks


class _TwoRole(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(2, 8, dtype=sc.DT)
        self.heads = nn.ModuleDict({r: nn.Linear(8, 1, dtype=sc.DT) for r in ("a", "b")})

    def critic_for(self, role):
        return lambda x: self.heads[role](torch.tanh(self.trunk(x)))


def test_shared_module_multi_role():
    d = _TwoRole()
    opt = torch.optim.Adam(d.parameters(), lr=0.05, betas=(0.0, 0.999))
    reg = GradientPenalty()
    for t in range(1, 9):
        xr, xf = sc.critic_batch(t)
        loss = 0
        for role in ("a", "b"):
            pen, st = reg.penalty(d.critic_for(role), xr, xf, t)
            assert torch.allclose(pen, _ref_penalty(d.critic_for(role), xr, xf, 1.0, 1.0), rtol=1e-12)
            loss = loss + pen + F.softplus(d.critic_for(role)(xf)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        reg.after_critic_step(opt)
    assert reg.state_dict()["observed_steps"] == 8 and reg.state_dict()["calls"] == 16


def test_k3p_state_roundtrip_bit_exact():
    _, full = sc.package_critic()
    split = 9  # after the guarded spike
    first, part = sc.package_critic(steps=range(1, split + 1))
    saved = copy.deepcopy(dict(D=first["D"].state_dict(), opt=first["opt"].state_dict(),
                               reg=first["reg"].state_dict(), guard=first["guard"].state_dict()))

    def restore(objs):
        objs["D"].load_state_dict(saved["D"])
        objs["opt"].load_state_dict(saved["opt"])
        objs["reg"].load_state_dict(saved["reg"])
        objs["guard"].load_state_dict(saved["guard"])
    second, rest = sc.package_critic(steps=range(split + 1, sc.STEPS + 1), setup=restore)
    trace = part["trace"] + rest["trace"]
    assert len(trace) == len(full["trace"])
    for a, b in zip(full["trace"], trace):
        assert torch.equal(a["pen"], b["pen"]), a["t"]
        assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"])), a["t"]
    assert second["guard"].clipped_tensors == first["guard"].clipped_tensors >= 1


def test_step_record_loads_checkpoints_from_the_anchor_formulation():
    record = CriticStepRecord()
    record.load_state_dict({"lr_max": 0.5, "lr_last": 0.25, "anchor_started": True, "calls": 3, "observed_steps": 2})
    assert record.state_dict() == {"lr_max": 0.5, "lr_last": 0.25, "calls": 3, "observed_steps": 2}
    with pytest.raises(ValueError):
        record.load_state_dict({"lr_max": 0.5})


def test_guard_threshold_and_min_steps():
    p = nn.Parameter(torch.zeros(4, dtype=sc.DT))
    opt = torch.optim.Adam([p], lr=0.1, betas=(0.0, 0.999))
    guard = CriticSpikeGuard(ratio=5.0, min_steps=2)
    for scale in (1.0, 1.0):
        p.grad = torch.full_like(p, scale)
        assert int(guard.apply_(opt) if opt.state else 0) == 0
        opt.step()
    p.grad = torch.full_like(p, 4.0)  # below 5x RMS
    assert int(guard.apply_(opt)) == 0 and torch.equal(p.grad, torch.full_like(p, 4.0))
    p.grad = torch.full_like(p, 100.0)
    assert int(guard.apply_(opt)) == 1
    vhat = opt.state[p]["exp_avg_sq"].mean() / (1 - 0.999 ** 2)
    assert torch.allclose(p.grad.square().mean().sqrt(), 5.0 * vhat.sqrt())
    assert guard.state_dict() == {"clipped_tensors": 1}
    young = CriticSpikeGuard(ratio=5.0, min_steps=3)
    p.grad = torch.full_like(p, 100.0)
    assert int(young.apply_(opt)) == 0


def test_guard_reads_exp_avg_sq_under_amsgrad():
    """The guard compares against Adam's running second moment, not AMSGrad's max (as gs2 was run)."""
    p = nn.Parameter(torch.zeros(4, dtype=sc.DT))
    opt = torch.optim.Adam([p], lr=0.1, betas=(0.0, 0.5), amsgrad=True)
    for scale in (10.0, 0.01, 0.01):  # the running max stays at the first, large gradient
        p.grad = torch.full_like(p, scale)
        opt.step()
    st = opt.state[p]
    grad = 25.0  # a spike against exp_avg_sq, not against the max buffer AMSGrad steps with
    bc = 1 - 0.5 ** 3
    assert grad / (st["max_exp_avg_sq"].mean() / bc).sqrt() < 5.0 < grad / (st["exp_avg_sq"].mean() / bc).sqrt()
    p.grad = torch.full_like(p, grad)
    assert int(CriticSpikeGuard(ratio=5.0, min_steps=0).apply_(opt)) == 1
    assert torch.allclose(p.grad.square().mean().sqrt(), 5.0 * (st["exp_avg_sq"].mean() / bc).sqrt())


def test_k3p_validation():
    for kwargs in (dict(coeff=-1.0), dict(kappa=float("nan")), dict(lazy_k=0)):
        with pytest.raises(ValueError):
            GradientPenalty(**kwargs)
    for removed in (dict(lr_floor=0.01), dict(anchor_weight=1.0), dict(anchor=None)):
        with pytest.raises(TypeError):
            GradientPenalty(**removed)  # the LR blend and the EMA anchor are gone
    D = sc.make_critic()
    reg = GradientPenalty()
    with pytest.raises(ValueError):
        reg.load_state_dict({"lr_max": 1.0})
    xr = torch.zeros(4, 2, dtype=sc.DT)
    with pytest.raises(ValueError):
        reg.penalty(D, xr, torch.zeros(4, 3, dtype=sc.DT))
    z = nn.Parameter(torch.zeros(8, 2))
    with pytest.raises(ValueError):
        LatentRowDamping(z, torch.zeros(8, 3))
    opt = torch.optim.Adam([z], betas=(0.9, 0.999))
    z.grad = torch.ones_like(z)
    damp = LatentRowDamping(z, torch.zeros_like(z))
    with pytest.raises(ValueError, match="beta1"):
        damp.begin(opt)


def test_after_critic_step_accepts_tensor_lr():
    reg = GradientPenalty()
    reg.after_critic_step(torch.tensor(0.5))
    reg.after_critic_step(torch.tensor(0.25, dtype=torch.float64))
    assert reg.state_dict()["lr_max"] == 0.5 and reg.state_dict()["lr_last"] == 0.25


def _k3p_trainer():
    from particlegan import GANTrainer, get_recipe
    torch.manual_seed(0)
    recipe = get_recipe("gan", num_particles=8, z_dim=2, batch_size=4, total_steps=10)
    G = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).double()
    D = nn.Sequential(nn.Linear(2, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 1)).double()
    return GANTrainer(recipe, G, D)


def test_trainer_runs_k3p_without_an_ema_critic_and_resumes_exactly():
    reals = [torch.randn(4, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(i)) for i in range(10)]
    full = _k3p_trainer()
    assert not hasattr(full, "ema_D") and not hasattr(full.opt_d, "ema_critic")
    penalties = []
    for real in reals:
        out = full.step(real, collect_stats=True)
        assert set(out["penalty_stats"]) == {"applied", "pen", "center", "r1", "fake_cap"}
        penalties.append(out["penalty"])
    critic_state = full.state_dict()["optimizers"][1]["regularizer"]
    assert set(critic_state) == {"record", "guard"} and critic_state["record"]["calls"] == 10

    first = _k3p_trainer()
    for real in reals[:5]:
        first.step(real)
    checkpoint = first.state_dict()
    resumed = _k3p_trainer()
    resumed.load_state_dict(checkpoint)
    for i, real in enumerate(reals[5:], start=5):
        assert torch.equal(resumed.step(real)["penalty"], penalties[i])
    for a, b in zip(resumed.D.parameters(), full.D.parameters()):
        assert torch.equal(a, b)
    # A checkpoint from the EMA-anchor formulation (EMA critic and anchor flag in
    # the critic optimizer state) loads with the EMA dropped.
    old = copy.deepcopy(checkpoint)
    regularizer = old["optimizers"][1]["regularizer"]
    regularizer["ema"] = copy.deepcopy(first.D.state_dict())
    regularizer["record"]["anchor_started"] = True
    old["recipe"] = {**old["recipe"], "reg_anchor_decay": 0.999, "reg_anchor_weight": 1.0}
    again = _k3p_trainer()
    again.load_state_dict(old)
    assert torch.equal(again.step(reals[5])["penalty"], penalties[5])
