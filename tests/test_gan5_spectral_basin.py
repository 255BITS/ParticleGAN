"""The GAN-5 basin term is a real-sample concavity penalty, not a coverage loss."""

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from reports.toy100.gan5_spectral_basin import BASIN_COEFF, real_laplacian, gan5_spectral_basin


def test_laplacian_is_negative_at_a_quadratic_maximum():
    points = torch.tensor([[0.2, -0.4], [1.0, 0.3]])

    def bowl(x):
        return -(x * x).sum(-1)

    lap = real_laplacian(bowl, points, eps=0.1)
    assert lap.shape == (2,)
    assert torch.allclose(lap, torch.full((2,), -4.0), atol=1e-4)


def test_nonplanar_input_is_a_noop():
    assert real_laplacian(lambda x: x.sum(-1), torch.zeros(4, 16)) is None


def test_penalty_coefficient_is_fixed_and_host_imports():
    assert BASIN_COEFF == 0.1
    model = SimpleMLPDiscriminator(2, hidden_dim=8, n_hidden=1, fourier=0)
    with gan5_spectral_basin(task="mode_hold", basin=True) as (recorder, source):
        recorder.enabled = True
        recorder.passthrough = False
        recorder.phase = 0
        recorder.row = {}
        real = torch.tensor([[0.0, 0.0], [0.5, -0.2]])
        fake = real + 0.3
        from particlegan.grad_regularizers import GradRegularizer
        penalty = GradRegularizer("b_cap", coeff=0.0, kappa=1.0)
        value = penalty(model, real, fake, step=1)
        assert torch.isfinite(value)
        assert "basin_penalty" in recorder.row
    assert "train_mode_hold" in source
