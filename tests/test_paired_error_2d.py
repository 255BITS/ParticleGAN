from unittest.mock import patch

import pytest
import torch

from benchmarks.paired_error_2d.task import NOISE_FLOOR, PairedError2D, data, metrics, noise_std, predict, targets
from benchmarks.toy_runner import ToyRun


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


def test_noise_schedule():
    assert noise_std(1, 2.) == pytest.approx(NOISE_FLOOR ** (1 / 8000))  # hold above start is ignored
    assert noise_std(6000, 2.) == pytest.approx(NOISE_FLOOR ** .75)
    assert noise_std(9000, .5) == pytest.approx(.65)  # held at 1.3x the target scale


@pytest.mark.parametrize("cloud", ["movable", "fixed"])
def test_cloud_gradient_no_mse_and_pure_forward(cloud):
    toy = ToyRun(PairedError2D("swirl2", cloud))
    before = toy.nets.prior.z.detach().clone()
    with patch("torch.nn.functional.mse_loss", side_effect=AssertionError("Output MSE used")):
        for _ in range(4):
            toy.step()
    assert torch.equal(before, toy.nets.prior.z) == (cloud == "fixed")
    x = toy.problem.x[:8]
    torch.testing.assert_close(predict(toy.nets, x), torch.cat([predict(toy.nets, v[None]) for v in x]))
    assert toy.measure(ema=True)["verdict"] in ("PASS", "FAIL")


def test_exact_resume():
    one = ToyRun(PairedError2D("affine2", "movable"))
    for _ in range(4):
        one.step()
    state = one.state_dict()
    problem = PairedError2D("affine2", "movable")
    problem.draws = one.problem.draws  # the problem's data-noise clock is not in ToyRun's checkpoint
    resumed = ToyRun(problem)
    resumed.load_state_dict(state)
    for _ in range(3):
        one.step()
        resumed.step()
    for a, b in zip(one.nets.generator.parameters(), resumed.nets.generator.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(one.nets.prior.z, resumed.nets.prior.z)
