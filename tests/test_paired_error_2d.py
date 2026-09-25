import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import pytest
import torch

from benchmarks.paired_error_2d.run import compare_states, freeze_and_evaluate, provenance
from benchmarks.paired_error_2d.task import Game, PROTOCOL, data, metrics, noise_std, targets


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def test_known_maps_and_correspondence():
    x, y = data("swirl2", "validation")
    torch.testing.assert_close(x.norm(dim=1), y.norm(dim=1))
    phi = 1.7 * (x / (3 ** .5)).square().sum(1)
    restored = torch.stack((phi.cos() * y[:, 0] + phi.sin() * y[:, 1],
                           -phi.sin() * y[:, 0] + phi.cos() * y[:, 1]), dim=1)
    torch.testing.assert_close(x, restored, atol=5e-7, rtol=1e-5)
    assert metrics(y, y)["nmse"] == 0
    assert metrics(y.flip(0), y)["nmse"] > 1
    torch.testing.assert_close(targets(torch.zeros(1, 2), "affine2"), torch.tensor([[.2, -.3]]))
    with pytest.raises(ValueError):
        targets(torch.zeros(1, 3), "swirl2")


@pytest.mark.parametrize("cloud", ["movable", "fixed"])
def test_cloud_gradient_no_mse_and_pure_forward(cloud):
    x, y = data("swirl2", "train")
    game = Game(y, cloud=cloud)
    before = game.model.particles.detach().clone()
    with patch("torch.nn.functional.mse_loss", side_effect=AssertionError("Output MSE used")):
        for step in range(1, 5):
            game.update(x, y, step)
    if cloud == "fixed":
        assert torch.equal(before, game.model.particles)
        assert game.model.particles.grad is None
    else:
        assert not torch.equal(before, game.model.particles)
        assert game.model.particles.grad.norm() > 0
    saved = copy.deepcopy(game.state_dict())
    torch.testing.assert_close(game.model(x[:8]), torch.cat([game.model(v[None]) for v in x[:8]]))
    assert compare_states(saved, game.state_dict())["exact"]


def test_configured_penalty_and_schedule():
    p = dict(PROTOCOL, steps=20)
    x, y = data("affine2", "train", p)
    game = Game(y, "cap-cosine", protocol=p)
    scale = torch.tensor(2., requires_grad=True)
    penalty, _ = game.cap.penalty(lambda v: scale * v[:, 0], y, y, step=4)
    assert float(penalty.detach()) == pytest.approx(3 * 4 * .75**2)
    assert float(torch.autograd.grad(penalty, scale)[0]) == pytest.approx(18.)
    assert game.cap.penalty(lambda v: scale * v[:, 0], y, y, step=3)[0] == 0
    game.update(x, y, 17)  # 16 completed updates = 80%, half-way through cosine.
    assert game.g.param_groups[0]["lr"] == pytest.approx(.0006 * .85 * .525)
    assert noise_std(6000, game.critic) == pytest.approx(.03 ** .75)


def test_exact_resume_and_fixed_vic_control():
    x, y = data("affine2", "train")
    one = Game(y, "cap-cosine", "fixed")
    for step in range(1, 5):
        one.update(x, y, step)
    resumed = Game(y, "cap-cosine", "fixed")
    resumed.load_state_dict(copy.deepcopy(one.state_dict()))
    for step in range(5, 9):
        one.update(x, y, step)
        resumed.update(x, y, step)
    assert compare_states(one.state_dict(), resumed.state_dict())["exact"]
    other = Game(y, "cap-cosine-vic005", "fixed")
    for step in range(1, 9):
        other.update(x, y, step)
    assert compare_states(one.state_dict(), other.state_dict())["exact"]


def test_audit_reports_discrepancies_and_test_waits_for_full_matrix(tmp_path):
    assert not compare_states({"a": torch.tensor([1.])}, {"a": torch.tensor([1.00001])})["exact"]
    assert not compare_states({"lr": .1}, {"lr": .10000000001})["exact"]
    assert compare_states({"a": torch.ones(1)}, {"a": torch.ones(1)})["exact"]
    with pytest.raises(ValueError, match="twelve"):
        freeze_and_evaluate(tmp_path, [])
    assert not (tmp_path / "selection.json").exists()


def test_worker_module_can_be_imported_by_spawn():
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        value = pool.submit(provenance).result(timeout=30)
    assert value["protocol"]["seed"] == 0
