import pytest
import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_translation_trust import (
    TranslationTrustRecorder, pr84_translation_trust, translation_trust_factor,
)


def test_endorsed_rising_translation_is_kept():
    # v(a) = a, so the probe is still rising at the proposed length.
    assert translation_trust_factor(0., 1., -1., 1.) == 1.


def test_refused_translation_is_rejected():
    assert translation_trust_factor(0., -1., 1., 1.) == 0.


def test_concave_translation_stops_at_the_quadratic_peak():
    # v(a) = a - a^2 peaks at 1/2. A proposed length of 1 keeps half.
    assert translation_trust_factor(0., 0., -2., 1.) == pytest.approx(.5)


def test_zero_translation_is_not_a_shrink():
    assert translation_trust_factor(1., 1., 1., 0.) == 1.


def test_bias_correction_removes_only_the_common_mode():
    torch.manual_seed(0)
    generator = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    bias = generator[-1].bias
    z = torch.randn(12, 4)
    y0 = generator(z).detach()
    with torch.no_grad():
        bias.add_(torch.tensor([.4, -.2]))
    y1 = generator(z).detach()
    translation = (y1 - y0).mean(0)
    residual = (y1 - y0) - translation
    assert residual.norm() == pytest.approx(0., abs=1e-5)
    factor = .25
    with torch.no_grad():
        bias.add_((factor - 1.) * translation)
    y = generator(z).detach()
    assert torch.allclose((y - y0).mean(0), factor * translation, atol=1e-5)
    assert torch.allclose(y - y.mean(0), y1 - y1.mean(0), atol=1e-5)


def test_factory_keeps_stall_reach_and_curvature_floors():
    with pr84_translation_trust(task="mode_hold") as (recorder, _source):
        assert isinstance(recorder, TranslationTrustRecorder)
        assert recorder.ramp == "stall"
        assert recorder.reach == pytest.approx(.5)
        assert recorder.curvature_bound == pytest.approx(base.G_CURVATURE_BOUND)
        assert recorder.d_curvature_bound == pytest.approx(base.D_CURVATURE_BOUND)
    assert base.SmoothedBothBoundRecorder is not TranslationTrustRecorder
