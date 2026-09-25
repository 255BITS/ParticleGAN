"""K3P package components vs the frozen research mechanism (CPU, float64)."""
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

from particlegan.k3p import (CriticAnchor, CriticSpikeGuard, DirectParticleResponse,  # noqa: E402
                             LatentRowDamping)
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


def _assert_critic_equal(ref, new):
    assert len(ref["trace"]) == len(new["trace"])
    for a, b in zip(ref["trace"], new["trace"]):
        assert a["applied"] == b["applied"], a["t"]
        assert torch.equal(a["pen"], b["pen"]), a["t"]
        assert a["s"] == b["s"], a["t"]
        assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"])), a["t"]
        assert (a["ema"] is None) == (b["ema"] is None), a["t"]
        if a["ema"] is not None:
            assert all(torch.equal(x, y) for x, y in zip(a["ema"], b["ema"])), a["t"]
    assert ref["prox"] == new["prox"]


def test_k3p_matches_frozen_mechanism(frozen):
    ref = frozen["critic"]
    objs, new = sc.package_critic()
    _assert_critic_equal(ref, new)
    s = [row["s"] for row in new["trace"]]
    assert s[:9] == [1.0] * 9 and 0.0 < s[9] < 1.0 and s[-1] == 0.0  # a, blend, b phases all covered
    assert next(r["t"] for r in new["trace"] if r["ema"] is not None) == 10
    assert any(p > 0 for p in new["prox"])
    assert ref["clipped"] >= 1 and objs["guard"].clipped_tensors == ref["clipped"]
    assert ref["calls"] == objs["reg"].state_dict()["calls"] == sc.STEPS


def test_k3p_lazy_k_matches_frozen(frozen):
    ref = frozen["lazy"]
    objs, new = sc.package_critic(lazy_k=2)
    _assert_critic_equal(ref, new)
    rows = new["trace"]
    assert all(float(r["pen"]) == 0.0 and not r["applied"] for r in rows if r["t"] % 2)
    assert any(r["s"] < 1.0 for r in rows if not r["t"] % 2)  # still K3P when applied
    assert objs["reg"].state_dict()["observed_steps"] == sc.STEPS  # s advances every critic step
    # applied steps carry 2x the coefficient
    D = sc.make_critic()
    xr, xf = sc.critic_batch(2)
    one = GradientPenalty(kappa=0.5).penalty(D, xr, xf, 1)[0]
    two = GradientPenalty(kappa=0.5, lazy_k=2).penalty(D, xr, xf, 2)[0]
    assert torch.allclose(two, 2 * one, rtol=1e-14, atol=0)


def _package_latent(split=None):
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


def test_latent_row_damping_matches_frozen_a2(frozen):
    ref = frozen["latent"]
    new, damp = _package_latent()
    assert ref["scoped_calls"] > 0
    for a, b in zip(ref["trace"], new):
        assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"])), a["t"]
        assert torch.equal(a["exp_avg"], b["exp_avg"]), a["t"]
    assert damp.started and 0 < ref["scoped_calls"] < 9  # rate >= 1/2 switches scoping off on some sparse steps
    resumed, _ = _package_latent(split=7)
    for a, b in zip(new, resumed):
        assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"])), a["t"]


def test_direct_particle_response_matches_frozen(frozen):
    ref = frozen["direct"]["trace"]
    particles, opt = sc.make_direct()
    resp = DirectParticleResponse([particles], torch.zeros(particles.numel(), dtype=sc.DT))

    def step(o):
        with resp.around(o):
            o.step()
        return resp.last_gain
    new = sc.run_direct(particles, opt, range(1, 11), step)
    assert any(r["gain"] > 1.0 for r in ref)
    for a, b in zip(ref, new):
        assert a["gain"] == b["gain"] and a["lr"] == b["lr"] and a["betas"] == b["betas"], a["t"]
        assert torch.equal(a["params"], b["params"]), a["t"]


def _recipe_critic(lazy_k=1, split=None):
    """The same critic scenario through the recipe's optimizer + penalty (plain opt.step()).

    With ``split``, checkpoint with the usual ``D``/``opt_d`` state_dicts after
    ``split`` steps and resume into freshly built objects.
    """
    from particlegan import get_recipe
    recipe = get_recipe(reg_kappa=0.5, reg_coeff=1.0, reg_every=lazy_k, network_lr_floor=0.01,
                        reg_anchor_decay=0.999, d_guard_ratio=5.0, d_guard_min_steps=3, lr=sc.LR0)

    def build(D):
        opt = recipe.make_critic_optimizer(D, ema_critic=copy.deepcopy(D), foreach=False)
        return opt, recipe.make_critic_penalty(opt, collect_stats=True)
    D = sc.make_critic()
    opt, penalty = build(D)
    started, prox = [], []

    class Adapter:  # run_critic's reg.penalty(D, xr, xf, t) -> (pen, stats)
        def penalty(self, critic, xr, xf, t):
            assert t == opt.record.observed_steps + 1  # the step comes from the optimizer
            pen = penalty(critic, xr, xf)
            if penalty.last_stats.get("phase") in ("blend", "b"):
                prox.append(penalty.last_stats["prox"])
            return pen, penalty.last_stats

    def after(o):
        started.append([e.clone() for e in o.ema_critic.parameters()] if o.record.anchor_started else None)
    if split is None:
        trace = sc.run_critic(Adapter(), D, opt, range(1, sc.STEPS + 1), after=after)
    else:
        trace = sc.run_critic(Adapter(), D, opt, range(1, split + 1), after=after)
        saved = copy.deepcopy({"D": D.state_dict(), "opt_d": opt.state_dict()})
        D = sc.make_critic(seed=9)
        opt, penalty = build(D)
        D.load_state_dict(saved["D"])
        opt.load_state_dict(saved["opt_d"])
        trace += sc.run_critic(Adapter(), D, opt, range(split + 1, sc.STEPS + 1), after=after)
    for row, ema in zip(trace, started):
        row["ema"] = ema
    return opt, dict(trace=trace, prox=prox)


@pytest.mark.parametrize("lazy_k", [1, 2])
def test_recipe_optimizer_and_penalty_match_frozen_mechanism(frozen, lazy_k):
    ref = frozen["critic" if lazy_k == 1 else "lazy"]
    opt, new = _recipe_critic(lazy_k)
    _assert_critic_equal(ref, new)
    assert opt.guard.clipped_tensors == ref["clipped"] >= (1 if lazy_k == 1 else 0)
    if lazy_k == 1:
        assert opt.record.calls == ref["calls"] == sc.STEPS
    # Resuming from the usual D / opt_d state_dicts inside the blend is bit-exact.
    _, resumed = _recipe_critic(lazy_k, split=11)
    _assert_critic_equal(ref, resumed)


def _optimizer_latent(split=None):
    from particlegan.k3p import K3PGeneratorAdam

    def build(seed=2):
        prior, G, _ = sc.make_latent(seed)
        opt = K3PGeneratorAdam([{"params": list(G.parameters())}, {"params": list(prior.parameters())}],
                               latent_table=prior.z, lr=0.02, betas=(0.0, 0.999), foreach=False)
        return prior, G, opt
    prior, G, opt = build()
    if split is None:
        return sc.run_latent(prior, G, opt, range(1, 15), lambda o: o.step()), opt
    trace = sc.run_latent(prior, G, opt, range(1, split + 1), lambda o: o.step())
    saved = copy.deepcopy(dict(prior=prior.state_dict(), G=G.state_dict(), opt=opt.state_dict()))
    prior, G, opt = build(seed=99)
    prior.load_state_dict(saved["prior"]); G.load_state_dict(saved["G"]); opt.load_state_dict(saved["opt"])
    return trace + sc.run_latent(prior, G, opt, range(split + 1, 15), lambda o: o.step()), opt


def test_generator_optimizer_matches_frozen_a2_and_direct(frozen):
    from particlegan.k3p import K3PGeneratorAdam
    for new, opt in (_optimizer_latent(), _optimizer_latent(split=7)):
        for a, b in zip(frozen["latent"]["trace"], new):
            assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"])), a["t"]
            assert torch.equal(a["exp_avg"], b["exp_avg"]), a["t"]
        assert opt.latent_damping.started
    particles, _ = sc.make_direct()
    opt = K3PGeneratorAdam([{"params": [particles], "_comparison_prior": True}], direct_particles=[particles],
                           lr=0.03, betas=(0.0, 0.999), foreach=False)

    def step(o):
        o.step()
        return o.direct_response.last_gain
    new = sc.run_direct(particles, opt, range(1, 11), step)
    for a, b in zip(frozen["direct"]["trace"], new):
        assert a["gain"] == b["gain"] and a["lr"] == b["lr"] and a["betas"] == b["betas"], a["t"]
        assert torch.equal(a["params"], b["params"]), a["t"]


def test_constant_lr_s_is_one_without_anchor():
    D = sc.make_critic()
    opt = sc.critic_optimizer(D)
    reg = GradientPenalty(kappa=0.5)
    trace = sc.run_critic(reg, D, opt, range(1, 9), after=reg.after_critic_step, lr_fn=lambda t: 1.0)
    assert all(r["s"] == 1.0 for r in trace)
    xr, xf = sc.critic_batch(3)
    pen, st = reg.penalty(D, xr, xf, 9)
    assert st["phase"] == "a" and st["prox"] == 0.0
    ref = GradientPenalty
    real = ref._grad_norm(D, xr, squared=True) / 2
    fake = (ref._grad_norm(D, xf) / 2 ** 0.5 - 0.5).relu().square()
    assert torch.equal(pen, 0.5 * (real.mean() + fake.mean()))


def test_blend_without_anchor_raises():
    D = sc.make_critic()
    reg = GradientPenalty()
    xr, xf = sc.critic_batch(1)
    reg.after_critic_step(1.0)
    reg.after_critic_step(0.1)
    assert reg.blend_weight() < 1.0
    with pytest.raises(ValueError, match="needs CriticAnchor/ema_critic"):
        reg.penalty(D, xr, xf, 3)
    reg.penalty(D, xr, xf, 3, ema_critic=copy.deepcopy(D))  # a per-call ema_critic is enough


def test_missing_after_critic_step_raises():
    D = sc.make_critic()
    xr, xf = sc.critic_batch(1)
    reg = GradientPenalty()
    reg.penalty(D, xr, xf, 1)
    with pytest.raises(RuntimeError, match="after_critic_step"):
        reg.penalty(D, xr, xf, 2)
    lazy = GradientPenalty(lazy_k=4)
    lazy.penalty(D, xr, xf, 4)
    with pytest.raises(RuntimeError):
        lazy.penalty(D, xr, xf, 8)


def test_two_critics_independent():
    from torch.optim import optimizer as optim_mod
    hooks = (len(optim_mod._global_optimizer_pre_hooks), len(optim_mod._global_optimizer_post_hooks))
    original_penalty = GradientPenalty.penalty
    a, trace_a = sc.package_critic()
    b_objs = {}

    def setup(objs):
        b_objs.update(objs)
    # Interleave critic B (constant LR, different data) step by step with a fresh critic A.
    A = sc.package_critic(steps=[])[0]
    B = sc.package_critic(steps=[], setup=setup)[0]
    for t in range(1, sc.STEPS + 1):
        ra = sc.run_package_critic(A, [t])
        for group in B["opt"].param_groups:
            group["lr"] = sc.LR0
        xr, xf = sc.critic_batch(t, seed=7)
        pen, st = B["reg"].penalty(B["D"], xr, xf, t)
        B["opt"].zero_grad(set_to_none=True)
        (F.softplus(B["D"](xf)).mean() + pen).backward()
        B["guard"].apply_(B["opt"])
        B["opt"].step()
        B["reg"].after_critic_step(B["opt"])
        assert st["s"] == 1.0
        assert torch.equal(ra["trace"][0]["pen"], trace_a["trace"][t - 1]["pen"])
    assert A["reg"].blend_weight() == 0.0 and B["reg"].blend_weight() == 1.0
    assert all(torch.equal(x, y) for x, y in zip(A["ema"].parameters(), a["ema"].parameters()))
    assert not B["reg"].state_dict()["anchor_started"]
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
    phases = []
    for t in range(1, 9):
        for group in opt.param_groups:
            group["lr"] = 0.05 if t < 4 else 0.0004
        xr, xf = sc.critic_batch(t)
        loss = 0
        for role in ("a", "b"):
            pen, st = reg.penalty(d.critic_for(role), xr, xf, t, ema_critic=ema.critic_for(role))
            loss = loss + pen + F.softplus(d.critic_for(role)(xf)).mean()
            phases.append(st["phase"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        reg.after_critic_step(opt)
    assert phases[:8] == ["a"] * 8 and set(phases[8:]) == {"b"}
    assert reg.state_dict()["observed_steps"] == 8 and reg.state_dict()["calls"] == 16
    assert not all(torch.equal(x, y) for x, y in zip(ema.parameters(), d.parameters()))


def test_k3p_state_roundtrip_bit_exact():
    _, full = sc.package_critic()
    split = 11  # blend phase, anchor running
    first, part = sc.package_critic(steps=range(1, split + 1))
    saved = copy.deepcopy(dict(D=first["D"].state_dict(), ema=first["ema"].state_dict(), opt=first["opt"].state_dict(),
                               reg=first["reg"].state_dict(), guard=first["guard"].state_dict()))

    def restore(objs):
        objs["D"].load_state_dict(saved["D"])
        objs["ema"].load_state_dict(saved["ema"])
        objs["opt"].load_state_dict(saved["opt"])
        objs["reg"].load_state_dict(saved["reg"])
        objs["guard"].load_state_dict(saved["guard"])
    second, rest = sc.package_critic(steps=range(split + 1, sc.STEPS + 1), setup=restore)
    assert 0.0 < part["trace"][-1]["s"] < 1.0
    _assert_critic_equal(full, dict(trace=part["trace"] + rest["trace"], prox=part["prox"] + rest["prox"]))
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


def test_k3p_validation():
    for kwargs in (dict(lr_floor=0.5), dict(lr_floor=-0.1), dict(coeff=-1.0), dict(kappa=float("nan")),
                   dict(anchor_weight=-1.0), dict(lazy_k=0)):
        with pytest.raises(ValueError):
            GradientPenalty(**kwargs)
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
    plain = GradientPenalty()
    plain.penalty(A, x_r, x_f)
    with pytest.raises(ValueError, match="different critic"):
        plain.penalty(B, x_r, x_f)
    # An explicit per-call EMA critic remains the multi-role path.
    emaB = copy.deepcopy(B).requires_grad_(False)
    anchored.penalty(B, x_r, x_f, ema_critic=emaB)


def _k3p_trainer():
    from particlegan import GANTrainer, get_recipe
    torch.manual_seed(0)
    recipe = get_recipe("gan", num_particles=8, z_dim=2, batch_size=4,
                        total_steps=10, lr_anneal_start=0.1)
    G = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).double()
    D = nn.Sequential(nn.Linear(2, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 1)).double()
    return GANTrainer(recipe, G, D)


def test_trainer_runs_k3p_through_blend_and_resumes_exactly():
    reals = [torch.randn(4, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(i)) for i in range(10)]
    full = _k3p_trainer()
    phases, penalties = [], []
    for real in reals:
        out = full.step(real, collect_stats=True)
        phases.append(out["penalty_stats"]["phase"])
        penalties.append(out["penalty"])
    assert phases[0] == "a" and "blend" in phases
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
