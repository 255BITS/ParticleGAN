import importlib.util
from pathlib import Path

import pytest
import torch

from particlegan import GANTrainer, get_recipe
from benchmarks.toy_runner import ToyRun, run
from lib.toy_models import sample_100gaussians


def load_example():
    spec = importlib.util.spec_from_file_location("grid_example", Path(__file__).parents[1] / "examples/100gaussians.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small(**overrides):
    return get_recipe(**{"num_particles": 32, "batch_size": 16, "total_steps": 20, **overrides})


@pytest.mark.parametrize("overrides", [{}, {"reg_kappa": 1.25, "reg_coeff": 3., "lr": .00051, "prior_reg": .05}])
def test_runner_matches_public_trainer_updates(overrides):
    """The problem-only example on the shared runner trains bit-for-bit like GANTrainer."""
    torch.set_num_threads(1)
    problem, recipe = load_example().Gaussians100(), small(**overrides)
    toy = ToyRun(problem, recipe=recipe)
    nets = problem.networks(recipe, 0)
    trainer = GANTrainer(recipe, nets.generator, nets.critics, prior=nets.prior, seed=0)
    data = torch.Generator().manual_seed(0)
    for _ in range(recipe.total_steps):
        toy.step()
        trainer.step(sample_100gaussians(16, "cpu", generator=data),
                     generator_real=lambda: sample_100gaussians(16, "cpu", generator=data))
    pairs = [(toy.nets.generator, trainer.G), (toy.nets.critics, trainer.D), (toy.nets.prior, trainer.prior),
             (toy.ema_nets.generator, trainer.ema_G), (toy.ema_nets.prior, trainer.ema_prior)]
    for ours, theirs in pairs:
        for key, tensor in ours.state_dict().items():
            assert torch.equal(tensor, theirs.state_dict()[key]), key


def test_grid_observer_preserves_training():
    torch.set_num_threads(1)
    problem, recipe = load_example().Gaussians100(), small(total_steps=6)
    plain, observed = ToyRun(problem, recipe=recipe), ToyRun(problem, recipe=recipe)
    for _ in range(recipe.total_steps):
        plain.step()
        observed.step()
        torch.randn(100)
        observed.measure()
        observed.measure(ema=True)
    a, b = plain.state_dict(), observed.state_dict()
    for x, y in zip(a["generator_side"] + a["ema"], b["generator_side"] + b["ema"]):
        for key in x:
            assert torch.equal(x[key], y[key]), key


@pytest.mark.parametrize("prior_kind", ["particles", "mog", "frozen_gaussian", "fresh_gaussian"])
def test_every_prior_control_runs_on_the_recipe(prior_kind):
    torch.set_num_threads(1)
    result = run(load_example().Gaussians100(prior_kind, distribution=True), recipe=small(total_steps=4))
    assert result["verdict"] in ("PASS", "FAIL")
    assert {"modes", "hq", "distribution"} <= set(result["ema"])


def test_grid_study_stock_arm_is_the_example_recipe_at_study_shape():
    from benchmarks.locked_shared.grid_study import ARMS, arm_recipe
    stock = arm_recipe(dict(ARMS)["stock"])
    assert (stock.batch_size, stock.num_particles, stock.total_steps) == (256, 20_000, 7000)


def test_grid_study_distribution_metrics_are_final_only(tmp_path, monkeypatch):
    """Checkpoints score coverage only; the grid distribution diagnostics run at the end."""
    import lib.toy_metrics
    from benchmarks.locked_shared import grid_study
    calls = []
    moments = lib.toy_metrics.per_mode_moments
    monkeypatch.setattr(lib.toy_metrics, "per_mode_moments", lambda *a, **k: calls.append(1) or moments(*a, **k))
    monkeypatch.setattr(grid_study, "TOTAL_STEPS", 4)
    monkeypatch.setattr(grid_study, "EXPECTED_STEPS", [2, 4])
    monkeypatch.setattr(grid_study, "ARMS", (("stock", {}),))
    monkeypatch.setattr(grid_study, "arm_recipe", lambda overrides: small(total_steps=4))
    torch.set_num_threads(1)
    row, = grid_study.run(tmp_path / "grid", torch.device("cpu"))["rows"]
    assert row.get("finished"), row.get("error")
    assert [p["step"] for p in row["curve"]] == [2, 4]
    assert {"sw1", "mode_tv"} <= set(row["distribution"]["live"]) and {"sw1"} <= set(row["distribution"]["ema"])
    assert len(calls) == 3  # the runner's final observation plus its final live and EMA measurements


def test_toml_runner_observers_preserve_training(tmp_path):
    """The TOML runner's d_gap and snapshots read the run without touching its training RNG."""
    from experiments.train_100gaussians import critic_gap, save_fake_scatter
    torch.set_num_threads(1)
    problem, recipe = load_example().Gaussians100(), small(total_steps=4)
    plain, observed = ToyRun(problem, recipe=recipe), ToyRun(problem, recipe=recipe)
    real = sample_100gaussians(64, "cpu", generator=torch.Generator().manual_seed(1))
    for step in range(recipe.total_steps):
        plain.step()
        observed.step()
        gap = critic_gap(observed, 16, 0)
        assert isinstance(gap, float) and gap == gap
        save_fake_scatter(observed.ema_nets.generator, observed.ema_nets.prior, tmp_path / f"{step}.png", real)
    assert len(list(tmp_path.glob("*.png"))) == recipe.total_steps
    a, b = plain.state_dict(), observed.state_dict()
    for x, y in zip(a["generator_side"] + a["ema"], b["generator_side"] + b["ema"]):
        for key in x:
            assert torch.equal(x[key], y[key]), key
