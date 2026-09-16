import numpy as np

from experiments.diagnose_memory_recurrence import history_retention, recurrence


def test_recurrence_finds_known_period_and_history_alignment_uses_absolute_time():
    t = np.arange(1100)
    path = np.stack((np.sin(2*np.pi*t/201), np.cos(2*np.pi*t/201)), -1)[None]
    result = recurrence(path[:, :1024])
    assert result['median_best_lag'] == 201
    assert result['median_best_error'] < 1e-20
    aligned = history_retention(path[:, :1024], path[:, 32:1056], 32)
    assert aligned['normalized_same_particle_cold_warm_error_median'] == 0
    shifted = history_retention(path[:, :1024], path[:, :1024], 32)
    assert shifted['normalized_same_particle_cold_warm_error_median'] > .1
