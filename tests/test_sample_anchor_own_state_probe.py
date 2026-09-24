"""No-training gates and source wiring for own-acquired continuation."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from reports.toy100 import sample_anchor_own_state_probe as probe


REPO = Path(__file__).resolve().parents[1]


def candidate_root():
    path = Path(os.environ.get('SAMPLE_ANCHOR_REST_ROOT', REPO)).resolve()
    if not (path / 'reports/toy100/sample_anchor_rest_candidate.py').is_file():
        pytest.skip('guard candidate is not yet present in this checkout')
    torch.set_num_threads(1)
    return path


def test_factory_binding_and_exact_resume_ast_are_read_only():
    root = candidate_root()
    from reports.toy100 import pr84_model_error_recovery as recovery
    from reports.toy100.pr84_critic_refinement_resume import resumed_source

    original = recovery.pr84_critic_refinement_finite
    rng = torch.get_rng_state().clone()
    for entrypoint in (probe.REST_FACTORY, probe.PRESTART_FACTORY):
        if entrypoint == probe.PRESTART_FACTORY and not (
                root/'reports/toy100/sample_anchor_prestart_candidate.py').is_file():
            continue
        factory, method, filename = probe.load_factory(root, entrypoint)
        assert filename.startswith('reports/toy100/') and method
        with probe.bound_factory(recovery, factory):
            with recovery.pr84_critic_refinement_finite() as (recorder, generated):
                assert recorder.correction is True and recorder.task == 'mode_hold'
                _, resumed = resumed_source(generated, 1200, 2400)
                assert 'range(1200, 2400)' in resumed
                assert resumed.count('_refinement_resume.before_step(step, locals())') == 1
    assert recovery.pr84_critic_refinement_finite is original
    assert torch.equal(torch.get_rng_state(), rng)


def test_incomplete_cold_rejected_before_output_or_training(tmp_path):
    root = candidate_root()
    cold = tmp_path / 'cold'
    cold.mkdir()
    (cold / 'declaration.json').write_text(json.dumps(dict(method='sample_anchor_rest_test',
        factory=probe.REST_FACTORY, phase='cold', source={})))
    (cold / 'summary.json').write_text(json.dumps(dict(method='sample_anchor_rest_test',
        factory=probe.REST_FACTORY, phase='cold', status='FAIL')))
    output = tmp_path / 'must-not-exist'
    command = [sys.executable, '-m', 'reports.toy100.sample_anchor_own_state_probe',
               '--phase', 'hold', '--factory', probe.REST_FACTORY,
               '--root', str(root), '--cold', str(cold),
               '--output', str(output)]
    run = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    assert run.returncode != 0
    assert 'cold gate is not for the explicit factory and method' in run.stderr
    assert not output.exists()


def test_incomplete_hold_cannot_enable_response(tmp_path):
    hold = tmp_path / 'hold'
    hold.mkdir()
    (hold / 'hold.json').write_text(json.dumps(dict(status='FAIL_FIRST_CHECK', method='test',
        factory=probe.REST_FACTORY, cold_ring_state_file_sha256='abc', source={}, receipt={})))
    declaration = dict(method='test', factory=probe.REST_FACTORY,
                       cold_ring_state_file_sha256='abc', source={},
                       cold_ring_state_file='mode_hold-final-state.pt')
    with pytest.raises(RuntimeError, match='complete own-acquired dense1200 hold'):
        probe.require_hold(hold, tmp_path, declaration)
