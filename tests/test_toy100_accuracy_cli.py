"""The accuracy CLI must fail a coverage-only pass and preserve individual scope."""
import json

import pytest

from benchmarks.toy100 import __main__ as cli


@pytest.mark.parametrize('accuracy_status,exit_code', [('FAIL', 1), ('PASS', 0)])
@pytest.mark.parametrize('flags', [[], ['--require-accuracy']])
def test_required_accuracy_controls_exit_status(tmp_path, monkeypatch, accuracy_status, exit_code, flags):
    config = tmp_path / 'recipe.json'
    config.write_text(json.dumps(dict(steps=1000)))
    monkeypatch.setattr(cli, 'train', lambda *_: dict(status='complete'))
    monkeypatch.setattr(cli, 'evaluate_suite', lambda *_, **__: dict(
        status='PASS', passed_problems=1, required_problems=1))
    scopes = []
    def accuracy(output, *, problem):
        scopes.append(problem)
        return dict(status=accuracy_status, passed_problems=int(accuracy_status == 'PASS'),
                    required_problems=1)
    monkeypatch.setattr(cli, 'evaluate_accuracy_suite', accuracy)
    assert cli.main(['run', '--config', str(config), '--output', str(tmp_path / 'run'),
                     '--problem', 'rotated100', '--no-render', *flags]) == exit_code
    assert scopes == ['rotated100']


def test_coverage_only_is_an_explicit_diagnostic_opt_out(tmp_path, monkeypatch):
    config = tmp_path / 'recipe.json'
    config.write_text(json.dumps(dict(steps=1000)))
    monkeypatch.setattr(cli, 'train', lambda *_: dict(status='complete'))
    monkeypatch.setattr(cli, 'evaluate_suite', lambda *_, **__: dict(
        status='PASS', passed_problems=1, required_problems=1))
    monkeypatch.setattr(cli, 'evaluate_accuracy_suite', lambda *_, **__: pytest.fail(
        'coverage-only must not silently invoke the stricter gate'))
    assert cli.main(['run', '--config', str(config), '--output', str(tmp_path / 'run'),
                     '--problem', 'grid100', '--no-render', '--coverage-only']) == 0
