"""Null calibration of the row-evidence statistics (review finding #11); CPU, a few seconds.

Metric: per-row P(p <= .05) under H0 (zero-mean touch gradients), pooled at three
late updates, and the largest Benjamini-Hochberg flagged fraction. Pass: the declared
``statistic="hotelling"`` option keeps P(p <= .05) in [.02, .075] and flags at most
.5% of rows under isotropic and anisotropic noise. The legacy sphericity statistic is
asserted to stay miscalibrated under anisotropic noise (it is kept bit-identical).
"""
import warnings

import pytest
import torch

from particlegan.row_evidence import RowEvidence, RowEvidenceUntestableWarning, f_survival


def _null_run(statistic, anisotropic, rows=800, d=8, window=25, updates=150):
    generator = torch.Generator().manual_seed(0)
    scales = (torch.logspace(-1.5, 1.5, d, dtype=torch.float64) if anisotropic
              else torch.ones(d, dtype=torch.float64))
    rotation, _ = torch.linalg.qr(torch.randn(d, d, generator=generator, dtype=torch.float64))
    evidence = RowEvidence(torch.zeros(rows, d, dtype=torch.float64), window=window, statistic=statistic)
    rejections, flagged = [], []
    for step in range(updates):
        noise = torch.randn(rows, d, generator=generator, dtype=torch.float64) * scales @ rotation.T
        evidence.update(noise)
        if step in (updates - 51, updates - 26, updates - 1):
            rejections.append(float((evidence.p_values <= .05).double().mean()))
            flagged.append(evidence.fraction)
    return sum(rejections) / len(rejections), max(flagged)


@pytest.mark.parametrize("anisotropic", [False, True])
def test_hotelling_option_is_null_calibrated(anisotropic):
    rate, flagged = _null_run("hotelling", anisotropic)
    assert .02 <= rate <= .075, rate
    assert flagged <= .005, flagged


def test_legacy_sphericity_statistic_is_miscalibrated_under_anisotropic_noise():
    isotropic, _ = _null_run("sphericity", False)
    anisotropic, _ = _null_run("sphericity", True)
    assert .02 <= isotropic <= .09
    assert anisotropic > .1  # documented legacy defect; default kept for qualified evidence


def test_hotelling_detects_a_persistent_mean():
    generator = torch.Generator().manual_seed(1)
    d, rows = 4, 64
    evidence = RowEvidence(torch.zeros(rows, d, dtype=torch.float64), window=25, statistic="hotelling")
    shift = torch.zeros(rows, d, dtype=torch.float64)
    shift[:8, 0] = 1.5
    for _ in range(80):
        evidence.update(shift + torch.randn(rows, d, generator=generator, dtype=torch.float64))
    assert bool(evidence.flag[:8].all()) and int(evidence.flag[8:].sum()) <= 1


def test_f_survival_matches_closed_forms():
    f = torch.tensor([.5, 1., 4.], dtype=torch.float64)
    d2 = torch.full_like(f, 7.)
    # F(2, n) survival: (1 + 2 f / n)^(-n / 2)
    assert torch.allclose(f_survival(f, 2., d2), (1 + 2 * f / d2).pow(-d2 / 2), atol=1e-12)


def test_untestable_configuration_warns_once_and_flags(monkeypatch):
    from particlegan import row_evidence
    monkeypatch.setattr(row_evidence, "_WARNED", set())  # module-global: isolate from earlier constructions
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = RowEvidence(torch.zeros(4, 34), window=50)
        RowEvidence(torch.zeros(4, 34), window=50)
    assert first.untestable and first.diagnostics()["untestable"]
    assert sum(issubclass(w.category, RowEvidenceUntestableWarning) for w in caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", RowEvidenceUntestableWarning)
        assert not RowEvidence(torch.zeros(4, 8), window=50).untestable
    with pytest.raises(ValueError, match="never test"):
        RowEvidence(torch.zeros(4, 99), window=50, statistic="hotelling")


def test_legacy_state_is_unchanged_and_hotelling_state_round_trips():
    legacy = RowEvidence(torch.zeros(4, 2)).state_dict()
    assert set(legacy) == {"M", "Qs", "W", "S", "flag", "fraction", "valid", "counters", "scale_c", "config"}
    assert "statistic" not in legacy["config"]
    source = RowEvidence(torch.zeros(6, 3, dtype=torch.float64), window=5, statistic="hotelling")
    for step in range(12):
        source.update(torch.randn(6, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(step)))
    restored = RowEvidence(torch.zeros(6, 3, dtype=torch.float64), window=5, statistic="hotelling")
    restored.load_state_dict(source.state_dict())
    assert torch.equal(restored.C, source.C)
    with pytest.raises(ValueError):
        RowEvidence(torch.zeros(6, 3, dtype=torch.float64), window=10).load_state_dict(source.state_dict())


def test_routed_evidence_counts_flagged_rows_once_per_update():
    from types import SimpleNamespace
    from particlegan.routing import RoutedEvidence
    spec = SimpleNamespace(min_observations=1, min_effect=0., persistence_threshold=0.)
    evidence = RoutedEvidence(torch.zeros(3, 2), spec)
    evidence.update(torch.tensor([[1., 1.], [0., 0.], [0., 0.]]))
    weights = torch.ones(4, 3, dtype=torch.float64)
    evidence.observe_effect(0, weights, torch.full((4,), 2., dtype=torch.float64),
                            torch.ones(4, dtype=torch.float64), 0)
    assert bool(evidence.flag[0]) and evidence.counters["flagged_rows_sum"] == 0
    evidence.update(torch.tensor([[1., 1.], [0., 0.], [0., 0.]]))
    assert evidence.counters["flagged_rows_sum"] == 1
