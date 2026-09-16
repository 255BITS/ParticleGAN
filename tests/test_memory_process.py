import numpy as np
import pytest

from experiments.diagnose_memory_process import counterfactuals, measured_process


def test_process_interventions_preserve_handoff_phase_and_noise():
    t = np.arange(1100)
    phase = .7-.25*t
    clean = np.stack((np.cos(phase), np.sin(phase)), -1)[None]+np.array([.2, -.3])
    noise = np.full((1, 32, 2), .01)
    variants = counterfactuals(clean, clean[:, :32]+noise)
    measurements = {}
    for name, (reference, observed, center, radius, omega) in variants.items():
        np.testing.assert_allclose(observed-reference[:, :32], noise, atol=2e-7)
        np.testing.assert_allclose((reference[:, 32]-center)/radius[:, None],
                                   clean[:, 32]-center, atol=1e-7)
        measured = measured_process(reference[:, 32:1056], center, omega)
        np.testing.assert_allclose(measured['radius'], radius, atol=1e-7)
        np.testing.assert_allclose(measured['signed_speed'], omega, atol=1e-7)
        assert measured['direction'].mean() == 1
        measurements[name] = measured
    assert (measurements['radius_high']['radius']-measurements['radius_low']['radius']).mean()/.7 == pytest.approx(1)
    assert -(measurements['speed_high']['signed_speed']-measurements['speed_low']['signed_speed']).mean()/.22 == pytest.approx(1)
