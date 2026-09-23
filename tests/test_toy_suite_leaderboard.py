"""The existing leaderboard CLI exposes the complete shared-recipe gate."""
import json

import pytest

from experiments import leaderboard


def test_leaderboard_fails_incomplete_common_suite(tmp_path):
    output = tmp_path / 'missing-runs'
    assert leaderboard.main(['--toy-suite-output', str(output)]) == 1
    report = json.loads((output / 'compatibility.json').read_text())
    assert report['status'] == 'INCOMPLETE'
    assert report['required'] == 22
    assert report['observed_passes'] == 0


def test_common_suite_cannot_be_scoped_to_one_problem(tmp_path):
    with pytest.raises(SystemExit) as error:
        leaderboard.main(['--toy-suite-output', str(tmp_path), '--toy100-problem', 'grid100'])
    assert error.value.code == 2
