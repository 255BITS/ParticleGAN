"""No-training checks for the source-bound long stationary diagnostic."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from reports.toy100 import sample_anchor_long_stationary_probe as probe


REPO = Path(__file__).resolve().parents[1]


def test_existing_fit_condition_receipts_are_summarized_without_queries():
    rows = [dict(fit=dict(records=[dict(rank=24, singular_min=2.,
        singular_max=20., trials=[{}, {}]), dict(rank=23,
        singular_min=0., singular_max=5., trials=[{}])]))]
    result = probe._fit_condition(rows)
    assert result['corrections'] == 1
    assert result['jacobians'] == 2
    assert result['nonlinear_trials'] == 3
    assert result['rank_min'] == 23 and result['rank_max'] == 24
    assert result['raw_svd_condition_min'] == 10.
    assert result['zero_or_nonfinite_singular_min'] == 1


def test_model_and_adam_summary_rejects_nonfinite_tensor():
    assert probe._tensor_stats([torch.tensor([3., 4.])]) == dict(
        elements=2, l2=5., abs_max=4., finite=True)
    with pytest.raises(FloatingPointError, match='nonfinite model or Adam'):
        probe._tensor_stats([torch.tensor([float('inf')])])


def test_qualified_preflight_and_exact_ast_do_not_train(tmp_path):
    root = Path(os.environ.get('SAMPLE_ANCHOR_LONG_ROOT',
                               '/ml2/hypergan/ParticleGAN-continuous-learning')).resolve()
    artifacts = root / 'artifacts/continuous-learning/round6'
    cold = artifacts / 'sample-anchor-prestart-cold-v2'
    hold = artifacts / 'sample-anchor-prestart-own-hold'
    response = artifacts / 'sample-anchor-prestart-response'
    if not all((path / file).is_file() for path, file in (
            (cold, 'summary.json'), (hold, 'hold.json'), (response, 'response.json'))):
        pytest.skip('qualified local research artifacts are unavailable')
    script = '''\
import importlib.util, json, pathlib, sys, torch
root=pathlib.Path(sys.argv[1]); sys.path.insert(0,str(root)); torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('long_preflight',sys.argv[2]); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
base=root/'artifacts/continuous-learning/round6'
factory,method,saved,*_=module.require_ready(base/'sample-anchor-prestart-cold-v2',base/'sample-anchor-prestart-own-hold',base/'sample-anchor-prestart-response',root,'reports.toy100.sample_anchor_prestart_candidate:sample_anchor_prestart_candidate')
with factory(task='mode_hold',start_step=0,correction=True) as (_,source):
 from reports.toy100.pr84_critic_refinement_resume import resumed_source
 _,continued=resumed_source(source,2400,12000)
assert 'range(2400, 12000)' in continued and continued.count('_refinement_resume.before_step(step, locals())')==1
assert saved['noise']['step_calls']==2400
print(json.dumps({'method':method,'steps':saved['noise']['step_calls'],'ast_sha256':module.sha(continued.encode())}))
'''
    run = subprocess.run([sys.executable, '-c', script, str(root), str(REPO /
        'reports/toy100/sample_anchor_long_stationary_probe.py')], cwd=root,
        capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result['steps'] == 2400 and len(result['ast_sha256']) == 64

    # The long entrypoint must refuse a failed response before it creates an
    # output directory or enters the host. The source archive remains exact.
    bad_response = tmp_path / 'failed-response'
    bad_response.mkdir()
    shutil.copyfile(response / 'declaration.json', bad_response / 'declaration.json')
    value = json.loads((response / 'response.json').read_text())
    value['verdict']['status'] = 'FAIL_LOCAL_RESPONSE_FILTER'
    (bad_response / 'response.json').write_text(json.dumps(value))
    (bad_response / 'source').symlink_to(response / 'source', target_is_directory=True)
    output = tmp_path / 'must-not-start'
    rejected = subprocess.run([sys.executable, str(REPO /
        'reports/toy100/sample_anchor_long_stationary_probe.py'),
        '--factory', 'reports.toy100.sample_anchor_prestart_candidate:sample_anchor_prestart_candidate',
        '--cold', str(cold), '--hold', str(hold), '--response', str(bad_response),
        '--root', str(root), '--output', str(output)], cwd=root,
        env={**os.environ, 'PYTHONPATH': str(root)},
        capture_output=True, text=True, check=False)
    assert rejected.returncode != 0
    assert 'paired same-target response did not pass' in rejected.stderr
    assert not output.exists()
