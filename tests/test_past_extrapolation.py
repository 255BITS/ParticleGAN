"""CUDA software checks for the paper equations and public trainer lifecycle."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, init
from particlegan.extrapolation import PastExtrapolation, stateless_directions
from particlegan.optim.dualnorm import NormalizedOptimizer
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.state import state_digest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; no CPU fallback")


def test_paper_equations_contract_bilinear_game_on_cuda():
    x = nn.Parameter(torch.tensor([1.], device="cuda:0", dtype=torch.float64))
    y = nn.Parameter(torch.tensor([0.], device="cuda:0", dtype=torch.float64))
    ox = NormalizedOptimizer([x], family="sgda", lr=.2)
    oy = NormalizedOptimizer([y], family="sgda", lr=.2)
    past = PastExtrapolation({"x": x, "y": y}, (ox, oy))
    expected = torch.cat((x, y)).detach()
    cached = torch.zeros_like(expected)
    for _ in range(80):
        look = expected - .2 * cached
        fresh_expected = torch.stack((look[1], -look[0]))
        expected = expected - .2 * fresh_expected
        cached = fresh_expected
        with past.lookahead():
            x.grad, y.grad = y.detach().clone(), -x.detach().clone()
            fresh = past.fresh_directions()
        ox.step()
        oy.step()
        past.previous = fresh
        assert torch.allclose(torch.cat((x, y)), expected, atol=1e-14, rtol=1e-14)
    assert float((x.square() + y.square()).detach()) < .05
    assert ox.state[x]["step"] == oy.state[y]["step"] == 80


def test_normalized_cache_matches_step_and_preserves_prior_row_ownership():
    weight = nn.Parameter(torch.eye(2, device="cuda:0"))
    table = nn.Parameter(torch.zeros(4, 2, device="cuda:0"))
    opt = NormalizedOptimizer([dict(params=[weight], role="generator"),
                              dict(params=[table], role="prior")], lr=.03)
    weight.grad = torch.tensor([[1., 2.], [3., 1.]], device="cuda:0")
    table.grad = torch.ones_like(table)  # Dense regularization grants no rows.
    rows = torch.tensor([1, 3, 1], device="cuda:0")
    opt.set_sampled_rows(table, rows)
    before = state_digest(opt.state_dict())
    directions = stateless_directions(opt)
    assert state_digest(opt.state_dict()) == before
    initial = weight.detach().clone(), table.detach().clone()
    opt.step()
    assert torch.allclose(weight, initial[0] - .03 * directions[weight])
    assert torch.equal(table, initial[1] - .03 * directions[table])
    assert torch.equal(table[[0, 2]], torch.zeros(2, 2, device="cuda:0"))
    past = PastExtrapolation({"weight": weight, "table": table}, (opt,))
    past.previous = {"weight": directions[weight], "table": directions[table]}
    base = table.detach().clone()
    with pytest.raises(RuntimeError, match="probe"):
        with past.lookahead():
            assert torch.equal(table[[0, 2]], base[[0, 2]])
            raise RuntimeError("probe")
    assert torch.equal(table, base)


def build(mode):
    recipe = get_recipe("bcap", optimizer_family="dualnorm", game_update=mode,
                        num_particles=16, z_dim=2, batch_size=8, total_steps=8)
    g = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 1)).cuda()
    d = nn.Sequential(nn.Linear(1, 8), nn.LeakyReLU(.2), nn.Linear(8, 1)).cuda()
    init.deterministic_orthogonal_(g)
    init.deterministic_orthogonal_(d)
    return GANTrainer(recipe, g, d, serial_backward=True)


@reproducible_execution
def resume_probe(*, device):
    real = torch.linspace(1., 3., 8, device=device)[:, None]
    trainer = build("extrapolation_from_past")
    for _ in range(3):
        trainer.step(real)
    saved = trainer.state_dict()
    for _ in range(3):
        trainer.step(real)
    expected = trainer.state_dict()
    restored = build("extrapolation_from_past")
    restored.load_state_dict(saved)
    for _ in range(3):
        restored.step(real)
    assert state_digest(expected) == state_digest(restored.state_dict())
    before = state_digest(restored.state_dict())
    bad = deepcopy(restored.state_dict())
    next(iter(bad["extrapolation"]["previous"].values())).fill_(float("nan"))
    with pytest.raises(ValueError, match="direction"):
        restored.load_state_dict(bad)
    assert state_digest(restored.state_dict()) == before


def test_cuda_checkpoint_exact_resume_and_invalid_cache_is_atomic():
    resume_probe(device="cuda:0")


@reproducible_execution
def joint_probe(*, device):
    past, simultaneous, alternating = [build(mode) for mode in
        ("extrapolation_from_past", "simultaneous", "alternating")]
    initial = past.state_dict()
    for trainer in (simultaneous, alternating):
        adapted = deepcopy(initial)
        adapted["recipe"] = trainer.recipe.to_dict()
        adapted.pop("extrapolation")
        trainer.load_state_dict(adapted)
    real = torch.linspace(1., 3., 8, device=device)[:, None]
    for trainer in (past, simultaneous, alternating):
        trainer.step(real)
    assert state_digest(past.state_dict()["models"]) == state_digest(simultaneous.state_dict()["models"])
    assert state_digest(past.G.state_dict()) != state_digest(alternating.G.state_dict())
    for trainer in (past, simultaneous, alternating):
        assert all(state["step"] == 1 for opt in (trainer.opt_g, trainer.opt_d) for state in opt.state.values())
        assert state_digest(trainer.state_dict()["streams"]) == state_digest(past.state_dict()["streams"])
    before = state_digest(past.state_dict()["streams"])
    past.sample(64)
    after = past.state_dict()["streams"]
    # Public default sampling owns only the eval stream.
    after["eval_generator"] = initial["streams"]["eval_generator"]
    assert state_digest(after) == before


def test_first_update_joint_state_and_no_extra_training_draws():
    joint_probe(device="cuda:0")


@pytest.mark.parametrize("overrides", [dict(optimizer_family="adam"), dict(optimizer_momentum=.5),
                                       dict(ema_decay=.9), dict(conditioning="conditional", num_classes=2)])
def test_unsupported_recipe_rejected(overrides):
    with pytest.raises(ValueError, match="joint game"):
        get_recipe("bcap", **{"optimizer_family": "dualnorm", "game_update": "extrapolation_from_past", **overrides})


def test_alternating_default_preserves_old_recipe_packets():
    assert "game_update" not in get_recipe("bcap").to_dict()
    recipe = get_recipe("bcap", optimizer_family="dualnorm", game_update="extrapolation_from_past")
    assert recipe.to_dict()["game_update"] == "extrapolation_from_past"
