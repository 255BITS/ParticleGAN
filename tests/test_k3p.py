"""K3P package components: the R1-free critic penalty, the EMA anchor, guard and A2 (CPU, float64)."""
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

from particlegan.k3p import CriticAnchor, CriticSpikeGuard, LatentRowDamping  # noqa: E402
from particlegan.grad_regularizers import CriticStepRecord, GradientPenalty, path_points  # noqa: E402

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
# The formulation promoted from the k3p-no-r1 study (nr_pathcap_anchor), written
# out term by term: c/2 * [path cap + fake cap + w * mean ||g_r - gbar_r||^2 / d].
def _ref_rms(D, x):
    x = x.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(D(x).sum(), x, create_graph=True)[0]
    return torch.sqrt(g.pow(2).flatten(1).sum(1) + 1e-12) / x[0].numel() ** 0.5


def _ref_penalty(D, Dbar, xr, xf, u, coeff, kappa, weight):
    n = min(len(xr), len(xf))
    x_hat = xr[:n] + u[:n, None] * (xf[:n] - xr[:n])
    path = (_ref_rms(D, x_hat) - kappa).relu().square().mean()
    fake = (_ref_rms(D, xf) - kappa).relu().square().mean()
    total = path + fake
    if Dbar is not None:
        x = xr.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(D(x).sum(), x, create_graph=True)[0]
        xb = xr.detach().clone().requires_grad_(True)
        gb = torch.autograd.grad(Dbar(xb).sum(), xb)[0]
        total = total + weight * (g - gb).pow(2).flatten(1).sum(1).mean() / xr[0].numel()
    return coeff / 2.0 * total


def _nudged(D, scale=0.3, seed=11):
    """A copy of D with perturbed weights (a stand-in for a moved EMA)."""
    other = copy.deepcopy(D).requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return other


def test_penalty_is_path_cap_plus_fake_cap_plus_anchor():
    D = sc.make_critic()
    ema = copy.deepcopy(D).requires_grad_(False)
    reg = GradientPenalty(coeff=0.3, kappa=0.2, anchor=CriticAnchor(D, ema))
    xr, xf = sc.critic_batch(1)
    # First call starts the anchor: EMA == D, prox exactly 0, no R1 anywhere.
    pen, st = reg.penalty(D, xr, xf, 1, generator=torch.Generator().manual_seed(3))
    u = torch.rand(len(xr), generator=torch.Generator().manual_seed(3), dtype=torch.float32)
    assert reg.record.anchor_started and st["prox"] == 0.0
    assert all(torch.equal(e, p) for e, p in zip(ema.parameters(), D.parameters()))
    assert torch.allclose(pen, _ref_penalty(D, None, xr, xf, u.to(sc.DT), 0.3, 0.2, 1.0), rtol=1e-12)
    assert st["path_cap"] > 0 and st["fake_cap"] > 0
    # Later calls add the anchor term against the (moved) EMA critic.
    reg.after_critic_step(0.1)
    moved = _nudged(D)
    ema.load_state_dict(moved.state_dict())
    pen, st = reg.penalty(D, xr, xf, 2, generator=torch.Generator().manual_seed(4))
    u = torch.rand(len(xr), generator=torch.Generator().manual_seed(4), dtype=torch.float32)
    ref = _ref_penalty(D, moved, xr, xf, u.to(sc.DT), 0.3, 0.2, 1.0)
    assert st["prox"] > 0 and torch.allclose(pen, ref, rtol=1e-12)
    params = list(D.parameters())[:-1]  # the output bias does not move an input gradient
    grads = torch.autograd.grad(pen, params, retain_graph=True)
    ref_grads = torch.autograd.grad(ref, params)
    assert all(torch.allclose(a, b, rtol=1e-10, atol=1e-14) for a, b in zip(grads, ref_grads))


def test_caps_are_one_sided_and_there_is_no_r1():
    D = nn.Linear(2, 1, dtype=sc.DT)
    with torch.no_grad():
        D.weight.copy_(torch.tensor([[0.3, -0.4]], dtype=sc.DT))  # slope RMS .5/sqrt(2) < kappa 1
    reg = GradientPenalty(anchor_weight=0.0)
    xr, xf = sc.critic_batch(1)
    pen, st = reg.penalty(D, xr, xf, 1, generator=torch.Generator().manual_seed(0))
    assert pen.item() == 0.0 and st["path_cap"] == 0.0 and st["fake_cap"] == 0.0


def test_anchor_weight_scales_and_zero_removes_the_anchor():
    D = sc.make_critic()
    xr, xf = sc.critic_batch(2)
    plain = GradientPenalty(kappa=0.2, anchor_weight=0.0)  # no anchor, no EMA needed
    pen0, st0 = plain.penalty(D, xr, xf, 1, generator=torch.Generator().manual_seed(1))
    assert st0["prox"] == 0.0 and not plain.record.anchor_started
    moved = _nudged(D)
    for weight in (1.0, 2.5):
        reg = GradientPenalty(kappa=0.2, anchor_weight=weight, anchor=CriticAnchor(D, copy.deepcopy(moved)))
        reg.record.anchor_started = True  # as after the first call
        pen, st = reg.penalty(D, xr, xf, 1, ema_critic=moved, generator=torch.Generator().manual_seed(1))
        u = torch.rand(len(xr), generator=torch.Generator().manual_seed(1), dtype=torch.float32).to(sc.DT)
        assert torch.allclose(pen, _ref_penalty(D, moved, xr, xf, u, 1.0, 0.2, weight), rtol=1e-12)
        assert torch.allclose(pen - pen0, torch.tensor(0.5 * st["prox"], dtype=sc.DT), rtol=1e-9)


def test_path_positions_come_from_the_generator_and_pair_by_index():
    D = sc.make_critic()
    reg = GradientPenalty(kappa=0.1, anchor_weight=0.0)
    xr, xf = sc.critic_batch(3)
    a = reg.penalty(D, xr, xf, 1, generator=torch.Generator().manual_seed(7))[0]
    b = reg.penalty(D, xr, xf, 1, generator=torch.Generator().manual_seed(7))[0]
    c = reg.penalty(D, xr, xf, 1, generator=torch.Generator().manual_seed(8))[0]
    assert torch.equal(a, b) and not torch.equal(a, c)
    u = torch.tensor([0.0, 1.0, 0.5])
    r = torch.zeros(3, 2, dtype=sc.DT)
    f = torch.ones(4, 2, dtype=sc.DT)
    assert torch.equal(path_points(r, f, u), torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=sc.DT))


def test_lazy_penalty_applies_every_kth_step_with_k_times_the_coefficient():
    D = sc.make_critic()
    xr, xf = sc.critic_batch(4)
    once = GradientPenalty(coeff=0.3, kappa=0.2, anchor_weight=0.0)
    lazy = GradientPenalty(coeff=0.3, kappa=0.2, lazy_k=4, anchor_weight=0.0)
    skipped, st = lazy.penalty(D, xr, xf, 3)
    assert skipped.item() == 0.0 and st == {"applied": False, "pen": 0.0}
    full = once.penalty(D, xr, xf, 1, generator=torch.Generator().manual_seed(2))[0]
    hit = lazy.penalty(D, xr, xf, 4, generator=torch.Generator().manual_seed(2))[0]
    assert torch.allclose(hit, 4 * full, rtol=1e-12)


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


def test_missing_after_critic_step_raises():
    D = sc.make_critic()
    xr, xf = sc.critic_batch(1)
    reg = GradientPenalty()
    reg.penalty(D, xr, xf, 1, ema_critic=copy.deepcopy(D))
    with pytest.raises(RuntimeError, match="after_critic_step"):
        reg.penalty(D, xr, xf, 2, ema_critic=copy.deepcopy(D))
    lazy = GradientPenalty(lazy_k=4)
    lazy.penalty(D, xr, xf, 4, ema_critic=copy.deepcopy(D))
    with pytest.raises(RuntimeError):
        lazy.penalty(D, xr, xf, 8, ema_critic=copy.deepcopy(D))


def test_anchor_needed_unless_weight_zero():
    D = sc.make_critic()
    xr, xf = sc.critic_batch(1)
    with pytest.raises(ValueError, match="CriticAnchor/ema_critic"):
        GradientPenalty().penalty(D, xr, xf, 1)
    GradientPenalty(anchor_weight=0.0).penalty(D, xr, xf, 1)
    GradientPenalty().penalty(D, xr, xf, 1, ema_critic=copy.deepcopy(D))  # a per-call ema_critic is enough


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
        pen = B["reg"](B["D"], xr, xf, t, generator=B["gen"])
        B["opt"].zero_grad(set_to_none=True)
        (F.softplus(B["D"](xf)).mean() + pen).backward()
        B["guard"].apply_(B["opt"])
        B["opt"].step()
        B["reg"].after_critic_step(B["opt"])
        assert torch.equal(ra["trace"][0]["pen"], trace_alone["trace"][t - 1]["pen"])
    assert all(torch.equal(x, y) for x, y in zip(A["ema"].parameters(), alone["ema"].parameters()))
    assert not all(torch.equal(x, y) for x, y in zip(A["ema"].parameters(), B["ema"].parameters()))
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
    ema = copy.deepcopy(d).requires_grad_(False)
    opt = torch.optim.Adam(d.parameters(), lr=0.05, betas=(0.0, 0.999))
    reg = GradientPenalty(anchor=CriticAnchor(d, ema))
    gen = torch.Generator().manual_seed(0)
    prox = []
    for t in range(1, 9):
        xr, xf = sc.critic_batch(t)
        loss = 0
        for role in ("a", "b"):
            pen, st = reg.penalty(d.critic_for(role), xr, xf, t, ema_critic=ema.critic_for(role), generator=gen)
            loss = loss + pen + F.softplus(d.critic_for(role)(xf)).mean()
            prox.append(st["prox"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        reg.after_critic_step(opt)
    assert prox[0] == 0.0 and all(p > 0 for p in prox[2:])
    assert reg.state_dict()["observed_steps"] == 8 and reg.state_dict()["calls"] == 16
    assert not all(torch.equal(x, y) for x, y in zip(ema.parameters(), d.parameters()))


def test_k3p_state_roundtrip_bit_exact():
    _, full = sc.package_critic()
    split = 9  # anchor running, after the guarded spike
    first, part = sc.package_critic(steps=range(1, split + 1))
    saved = copy.deepcopy(dict(D=first["D"].state_dict(), ema=first["ema"].state_dict(), opt=first["opt"].state_dict(),
                               reg=first["reg"].state_dict(), guard=first["guard"].state_dict(),
                               gen=first["gen"].get_state()))

    def restore(objs):
        objs["D"].load_state_dict(saved["D"])
        objs["ema"].load_state_dict(saved["ema"])
        objs["opt"].load_state_dict(saved["opt"])
        objs["reg"].load_state_dict(saved["reg"])
        objs["guard"].load_state_dict(saved["guard"])
        objs["gen"].set_state(saved["gen"])
    second, rest = sc.package_critic(steps=range(split + 1, sc.STEPS + 1), setup=restore)
    trace = part["trace"] + rest["trace"]
    assert len(trace) == len(full["trace"])
    for a, b in zip(full["trace"], trace):
        assert torch.equal(a["pen"], b["pen"]), a["t"]
        assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"])), a["t"]
        assert all(torch.equal(x, y) for x, y in zip(a["ema"], b["ema"])), a["t"]
    assert full["prox"][0] == 0.0 and all(p > 0 for p in full["prox"][1:])
    assert second["guard"].clipped_tensors == first["guard"].clipped_tensors >= 1


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


def test_guard_reads_the_amsgrad_max_second_moment():
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
    assert int(CriticSpikeGuard(ratio=5.0, min_steps=0).apply_(opt)) == 0
    assert torch.equal(p.grad, torch.full_like(p, grad))


def test_k3p_validation():
    for kwargs in (dict(coeff=-1.0), dict(kappa=float("nan")), dict(anchor_weight=-1.0), dict(lazy_k=0)):
        with pytest.raises(ValueError):
            GradientPenalty(**kwargs)
    with pytest.raises(TypeError):
        GradientPenalty(lr_floor=0.01)  # the LR-driven blend is gone
    D = sc.make_critic()
    with pytest.raises(ValueError, match="either anchor"):
        GradientPenalty(anchor=CriticAnchor(D, copy.deepcopy(D)), record=CriticStepRecord())
    with pytest.raises(ValueError):
        CriticAnchor(D, D)
    with pytest.raises(ValueError):
        CriticAnchor(D, nn.Linear(2, 1, dtype=sc.DT))
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


def test_one_regularizer_rejects_a_second_critic_without_explicit_ema():
    torch.manual_seed(0)
    A, B = nn.Linear(2, 1).double(), nn.Linear(2, 1).double()
    x_r, x_f = torch.randn(4, 2, dtype=torch.float64), torch.randn(4, 2, dtype=torch.float64)
    anchored = GradientPenalty(anchor=CriticAnchor(A, copy.deepcopy(A).requires_grad_(False)))
    anchored.penalty(A, x_r, x_f)
    with pytest.raises(ValueError, match="anchor tracks"):
        anchored.penalty(B, x_r, x_f)
    plain_anchor = GradientPenalty(anchor=CriticAnchor(A, copy.deepcopy(A).requires_grad_(False)))
    plain_anchor.anchor.critic = None  # an anchor that does not name its critic
    plain_anchor.penalty(A, x_r, x_f)
    with pytest.raises(ValueError, match="different critic"):
        plain_anchor.penalty(B, x_r, x_f)
    # An explicit per-call EMA critic remains the multi-role path.
    emaB = copy.deepcopy(B).requires_grad_(False)
    anchored.penalty(B, x_r, x_f, ema_critic=emaB)


def _k3p_trainer():
    from particlegan import GANTrainer, get_recipe
    torch.manual_seed(0)
    recipe = get_recipe("gan", num_particles=8, z_dim=2, batch_size=4, total_steps=10)
    G = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).double()
    D = nn.Sequential(nn.Linear(2, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 1)).double()
    return GANTrainer(recipe, G, D)


def test_trainer_runs_k3p_with_anchor_from_step_one_and_resumes_exactly():
    reals = [torch.randn(4, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(i)) for i in range(10)]
    full = _k3p_trainer()
    prox, penalties = [], []
    for real in reals:
        out = full.step(real, collect_stats=True)
        prox.append(out["penalty_stats"]["prox"])
        penalties.append(out["penalty"])
    assert prox[0] == 0.0 and all(p > 0 for p in prox[1:])
    critic_state = full.state_dict()["optimizers"][1]["regularizer"]
    assert critic_state["record"]["anchor_started"]
    assert "1.running_mean" in critic_state["ema"]

    first = _k3p_trainer()
    for real in reals[:5]:
        first.step(real)
    checkpoint = first.state_dict()
    assert "ema_D" not in checkpoint["models"] and checkpoint["optimizers"][1]["regularizer"]["ema"] is not None
    resumed = _k3p_trainer()
    resumed.load_state_dict(checkpoint)
    for i, real in enumerate(reals[5:], start=5):
        assert torch.equal(resumed.step(real)["penalty"], penalties[i])
    for a, b in zip(resumed.D.parameters(), full.D.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(resumed.ema_D.parameters(), full.ema_D.parameters()):
        assert torch.equal(a, b)
