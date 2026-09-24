"""No-training checks for the descriptive long-continuation grade."""

import pytest

from reports.toy100 import sample_anchor_practical_long_probe as probe


def test_failure_runs_and_recovery_preserve_strict_misses():
    rows = [dict(step=step, modes=8, hq=1., passed=True)
            for step in range(probe.START + 1, probe.END + 1)]
    for step, modes, hq in ((2401, 7, 1.), (2402, 7, .923), (2500, 8, .8)):
        rows[step - probe.START - 1].update(modes=modes, hq=hq, passed=False)
    result = probe.summarize_quality(rows)
    assert result['checks'] == 9600 and result['failing_checks'] == 3
    assert result['passing_checks'] == 9597 and result['first_failure_step'] == 2401
    assert result['failure_runs'] == [
        dict(first=2401, last=2402, min_modes=7, min_hq=.923,
             length=2, recovered_at=2403),
        dict(first=2500, last=2500, min_modes=8, min_hq=.8,
             length=1, recovered_at=2501)]
    assert result['endpoint']['passed'] and result['last_200_passing'] == 200


def test_missing_checkpoint_is_invalid():
    rows = [dict(step=step, modes=8, hq=1., passed=True)
            for step in range(probe.START + 2, probe.END + 1)]
    with pytest.raises(RuntimeError, match='missing or duplicated'):
        probe.summarize_quality(rows)
