"""The shared problem-only toy runner and its reference example (mode_hold)."""
import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks import toy_runner
from benchmarks.locked_shared import mode_hold
from benchmarks.toy_runner import Networks, Sample, ToyProblem, ToyRun, View, run
from particlegan import get_recipe, learning_rate_scales

ROOT = Path(__file__).resolve().parents[1]


def _mlp(i, o, h=16):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, o))


class Blob(ToyProblem):
    """A 2-D Gaussian blob; the smallest 1G + prior + 1D problem."""
    name = "blob"

    def __init__(self, steps=12):
        self.center = torch.tensor([1.0, -1.0])
        self.steps = steps

    def recipe(self):
        return get_recipe(z_dim=2, num_particles=32, batch_size=16, total_steps=self.steps,
                          network_lr_horizon_cap=6, d_guard_min_steps=2)

    def networks(self, recipe, seed):
        return Networks(generator=_mlp(2, 2), critics=_mlp(2, 1))

    def real(self, n, stream):
        return self.center + 0.1 * torch.randn(n, 2, generator=stream)

    def metrics(self, model):
        return {"err": float((model.sample(256).x.mean(0) - self.center).norm())}

    def verdict(self, metrics):
        return "PASS" if metrics["err"] < 10 else "FAIL"

    def shift(self):
        self.center.neg_()


def test_mode_hold_is_problem_only():
    source = (ROOT / "benchmarks/locked_shared/mode_hold.py").read_text()
    tree = ast.parse(source)
    names = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not {"torch.optim", "torch.optim.Adam", "learning_rate_scale", "schedule_optimizer"} & names
    assert '["lr"]' not in source and "GANLoss" not in source and "GradientPenalty" not in source


def test_mode_hold_runs_on_the_recipe_and_is_deterministic():
    problem = mode_hold.ModeHold()
    recipe = problem.recipe().replace(total_steps=30)
    first = run(problem, recipe=recipe, observe_every=10)
    again = run(mode_hold.ModeHold(), recipe=recipe, observe_every=10)
    assert first["live"] == again["live"] and first["ema"] == again["ema"]
    assert [p["step"] for p in first["curve"]] == [10, 20, 30]
    assert first["verdict"] == mode_hold.verdict(first["live"])
    assert first["recipe"]["lr"] == get_recipe().lr  # shipped optimizer settings, task shape only
    legacy = mode_hold.train_mode_hold(diagnostics=True, steps=20)
    assert {"modes", "hq", "verdict", "live", "live_curve"} <= set(legacy)
    assert "support" in legacy["live"]


def test_extended_hold_and_shift_protocols(tmp_path):
    problem = Blob(steps=8)
    log = tmp_path / "blob.log"
    result = run(problem, steps=16, shift_step=10, observe_every=2, log_path=log)
    assert result["extended_from"] == 8 and result["shift"]["step"] == 10
    assert torch.equal(problem.center, torch.tensor([-1.0, 1.0]))
    lines = log.read_text().splitlines()
    assert '"event": "shift"' in lines[5] and '"event": "final"' in lines[-1]
    assert result["hold"]["observations"] == 8
    toy = ToyRun(Blob(steps=8))
    for _ in range(12):
        toy.step()
    network, prior = learning_rate_scales(12, toy.recipe)
    assert (network, prior) == (toy.recipe.network_lr_floor, toy.recipe.lr_floor)  # held past the budget
    assert toy.opt_g.param_groups[0]["lr"] == toy.recipe.lr * learning_rate_scales(11, toy.recipe)[0]


def test_checkpoint_resumes_exactly():
    full = ToyRun(Blob())
    outs = [full.step() for _ in range(10)]
    part = ToyRun(Blob())
    for _ in range(6):
        part.step()
    state = part.state_dict()
    resumed = ToyRun(Blob(), seed=5)
    resumed.load_state_dict(state)
    for expected in outs[6:]:
        got = resumed.step()
        assert torch.equal(got["loss_d"], expected["loss_d"]) and torch.equal(got["loss_g"], expected["loss_g"])


class TwoCriticsAE(Blob):
    """AE + D with two critics, a per-role critic recipe, a conditional view and an extra loss."""
    name = "two_critics_ae"

    def networks(self, recipe, seed):
        slow = recipe.replace(d_lr_mult=0.5)
        return Networks(generator=_mlp(2, 2), encoder=_mlp(2, 2),
                        critics={"marginal": _mlp(2, 1), "joint": _Joint()}, recipes={"joint": slow})

    def real(self, n, stream):
        x = super().real(n, stream)
        return Sample(x, condition=(torch.ones(n, 1),))

    def views(self, nets, real, fake):
        return [View("marginal", real.x, fake.x), View("joint", real.x, fake.x, real.condition, weight=0.5)]

    def losses(self, role, nets, real, fake):
        if role != "generator":
            return {}
        return {"reconstruction": (nets.generator(nets.encoder(real.x)) - real.x).pow(2).mean()}


class _Joint(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _mlp(3, 1)

    def forward(self, x, c):
        return self.net(torch.cat([x, c], 1))


def test_multi_role_extension_points():
    toy = ToyRun(TwoCriticsAE())
    out = toy.step()
    assert set(toy.opt_d) == {"marginal", "joint"} and "reconstruction" in out
    assert toy.opt_d["joint"].param_groups[0]["base_lr"] == 0.5 * toy.opt_d["marginal"].param_groups[0]["base_lr"]
    encoder_ids = {id(p) for p in toy.nets.encoder.parameters()}
    assert encoder_ids <= {id(p) for g in toy.opt_g.param_groups for p in g["params"]}
    assert [g["role"] for g in toy.opt_g.param_groups] == ["network", "prior"]


class Particles(Blob):
    """Particles-only: the samples are a parameter table; no prior, no generator network."""
    name = "particles"

    def networks(self, recipe, seed):
        table = nn.Parameter(torch.zeros(16, 2))
        holder = nn.Module()
        holder.table = table
        return Networks(generator=holder, critics=_mlp(2, 1), prior=None, direct_particles=[table])

    def fake(self, nets, n, stream, real):
        return nets.generator.table[torch.randint(16, (n,), generator=stream)]


class Student(Blob):
    """Student-only: no critic; the generator trains on its supervised loss alone."""
    name = "student"

    def networks(self, recipe, seed):
        return Networks(generator=_mlp(2, 2), critics={}, prior=None)

    def fake(self, nets, n, stream, real):
        x = torch.randn(n, 2, generator=stream)
        return Sample(nets.generator(x), condition=(x,))

    def losses(self, role, nets, real, fake):
        return {"mse": (fake.x - 2 * fake.condition[0]).pow(2).mean()}


def test_particles_only_and_student_only():
    toy = ToyRun(Particles())
    toy.step()
    assert toy.opt_g.direct_response is not None
    assert not torch.equal(toy.nets.generator.table, torch.zeros(16, 2))
    student = ToyRun(Student(steps=60))
    x = torch.randn(256, 2, generator=torch.Generator().manual_seed(3))
    before = float((student.nets.generator(x) - 2 * x).pow(2).mean().detach())
    assert "loss_d" not in student.step() and not student.opt_d
    for _ in range(59):
        student.step()
    assert float((student.nets.generator(x) - 2 * x).pow(2).mean().detach()) < before


def test_runner_rejects_unknown_role_recipes():
    class Bad(Blob):
        def networks(self, recipe, seed):
            return Networks(generator=_mlp(2, 2), critics=_mlp(2, 1), recipes={"nope": recipe})
    with pytest.raises(ValueError, match="unknown roles"):
        ToyRun(Bad())
    assert toy_runner.RECIPE_PRIOR is Networks(None, {}).prior
