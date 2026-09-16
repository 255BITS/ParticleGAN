from pathlib import Path

import pytest

from experiments.analyze_memory_core import analyze


def test_invalidated_experiments_cannot_reenter_leaderboard(tmp_path):
    bad = tmp_path/'invalidated'
    (bad/'runs'/'model').mkdir(parents=True)
    (bad/'INVALIDATED.md').write_text('incorrect gradient')
    with pytest.raises(ValueError, match='invalidated queue'):
        analyze(bad/'runs', tmp_path/'report')
    good = tmp_path/'valid'
    (good/'runs').mkdir(parents=True)
    with pytest.raises(ValueError, match='invalidated baseline'):
        analyze(good/'runs', tmp_path/'report', baselines=[bad/'runs'/'model'])
