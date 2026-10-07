"""Numerical smoke/continuation reducers and CUDA stream ownership contracts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.gaussian_tasks import (build, bounds, grade, schedule, training_digest, validate_task)
from experiments.forge.views import load_tasks, load_view, grade_result
from benchmarks.toy_audit.gaussian_smoke_study import stability_task

ROOT = Path(__file__).resolve().parents[1]
GOOD = dict(sample_count=4096, finite_fraction=1., mean_error_sigma=0., std_ratio=1., cdf_ks=.01)


def task(name='gaussian1d_smoke'):
    return json.loads((ROOT / 'configs/forge/tasks' / (name+'.json')).read_text())


def evidence():
    rows = [dict(step=step, **GOOD) for step in schedule(0,1000)]
    confirms = [dict(step=step, metrics=deepcopy(GOOD), primary_state_sha256='a'*64,
                     confirmed_state_sha256='a'*64, independent_stream='eval/live/smoke_confirmation',
                     training_state_unchanged=True) for step in schedule(0,1000)]
    return dict(completed_steps=1000, observations=rows, confirmations=confirms)


def test_smoke_accepts_a_single_confirmed_hit_without_terminal_hold():
    raw=evidence()
    for row in raw['observations'][1:]:row['cdf_ks']=.2
    for row in raw['confirmations'][1:]:row['metrics']['cdf_ks']=.2
    result=grade(task(),raw)
    assert result['gate_status']=='PASS'
    assert result['evaluator_result']['confirmed_steps']==[42]


@pytest.mark.parametrize('delta,status', [('missing','INCOMPLETE'),('off_cadence','INCOMPLETE'),
                                          ('duplicate','INVALID'),('no_confirmation','INCOMPLETE'),
                                          ('wrong_state','INVALID'),('nonfinite','FAIL'),('short','INCOMPLETE')])
def test_smoke_rejects_incomplete_or_invalid_evidence(delta,status):
    raw=evidence()
    if delta=='missing':raw['observations'].pop()
    elif delta=='off_cadence':raw['observations'][0]['step']=43
    elif delta=='duplicate':raw['observations'][1]['step']=42
    elif delta=='no_confirmation':raw['confirmations'].pop()
    elif delta=='wrong_state':raw['confirmations'][0]['confirmed_state_sha256']='b'*64
    elif delta=='nonfinite':raw['observations'][0]['finite_fraction']=.99
    else:raw['completed_steps']=999
    assert grade(task(),raw)['gate_status']==status


def test_smoke_requires_independent_confirmation_to_pass_full_bounds():
    raw=evidence()
    for row in raw['confirmations']:row['metrics']['cdf_ks']=.2
    assert grade(task(),raw)['gate_status']=='FAIL'


def test_stability_requires_every_hold_and_five_terminal_reacquisition():
    value=task('gaussian1d_stability')
    rows=[dict(step=step,**GOOD) for step in schedule(1000,4000)+schedule(4000,6000)]
    raw=dict(completed_steps=6000, observations=rows,
             frozen_observations=[dict(step=step,**GOOD) for step in schedule(4000,6000)],
             continuity=dict(prefix_steps=1000,restored_exactly=True,history_reset=False,
                             frozen_completed_steps=4000,matched_frozen_draws=True))
    assert grade(value,raw)['gate_status']=='PASS'
    raw['observations'][71]['cdf_ks']=.2
    assert grade(value,raw)['gate_status']=='FAIL'
    raw['observations'][71]['cdf_ks']=.01
    raw['observations'][95]['cdf_ks']=.2
    assert grade(value,raw)['evaluator_result']['reacquisition_status']=='FAIL'
    raw['observations'][95]['cdf_ks']=.01
    raw['observations'][119]['cdf_ks']=.2
    assert grade(value,raw)['gate_status']=='FAIL'


def test_new_task_placement_preserves_archived_task():
    tasks=load_tasks(ROOT);view=load_view(ROOT,'discriminator_stability')
    assert tasks['gaussian1d_acquisition']['execution']['prior']['sigma']==.025
    assert tasks['gaussian1d_acquisition']['evaluation']['minimum_stable_checks']==5
    assert next(a for a in view['assignments'] if a['task']=='gaussian1d_smoke')['qualification_tier']==1
    assert next(a for a in view['assignments'] if a['task']=='gaussian1d_stability')['qualification_tier']==2
    assert stability_task(tasks['gaussian1d_smoke'])['execution']==tasks['gaussian1d_stability']['execution']
    changed=deepcopy(tasks['gaussian1d_smoke']);changed['evaluation']['thresholds'][5][2]=.1
    with pytest.raises(ValueError,match='fixed'):validate_task(changed)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA contract requires GPU')
def test_cuda_confirmation_and_exact_resume_are_isolated():
    from experiments.forge.state import state_digest
    from benchmarks.toy_audit.reproducibility import reproducible_execution
    candidate=json.loads(next((ROOT/'configs/forge/configurations').glob('bcap-dualnorm--7beb*.json')).read_text())
    request=dict(candidate=candidate,protocol={'seed':0})
    @reproducible_execution
    def check(*,device):
        context,trainer,_=build(request,task(),device)
        data=context.streams.generator('data',component='target',purpose='training',device='cpu')
        evaluate=context.streams.generator('eval',component='live',purpose='samples')
        confirm=context.streams.generator('eval',component='live',purpose='smoke_confirmation')
        before=context.streams.audit();state_before=training_digest(context)
        trainer.sample(4096,generator=evaluate,output_noise=False)
        trainer.sample(4096,generator=confirm,output_noise=False)
        after=context.streams.audit()
        assert training_digest(context)==state_before
        allowed=[key for key,binding in context.streams.manifest()['bindings'].items() if binding['family']=='eval']
        assert context.streams.compare(before,after,allowed=allowed)['unintended_rng_deviations']==0
        trainer.step((torch.randn(128,1,generator=data)*.5+2).to(device))
        saved=deepcopy(context.state_dict())
        resumed,resumed_trainer,_=build(request,task(),device)
        resumed.load_state_dict(deepcopy(saved))
        assert state_digest(resumed.state_dict())==state_digest(saved)
        real=torch.randn(128,1,generator=data)*.5+2
        trainer.step(real.to(device))
        resumed_trainer.step(real.to(device))
        # Data stream was consumed outside trainer and is explicitly restored in the resumed context.
        resumed.streams.generator('data',component='target',purpose='training',device='cpu').set_state(data.get_state())
        assert state_digest(context.state_dict())==state_digest(resumed.state_dict())
        with pytest.raises(ValueError,match='compatible passing own'):
            from experiments.forge.gaussian_tasks import _restore_parent
            _restore_parent(request,task('gaussian1d_stability'),{},resumed)
    check(device='cuda:1')

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA contract requires GPU')
def test_tiny_public_execution_keeps_receipt_outside_strict_artifact_tree(tmp_path,monkeypatch):
    """Two-update software fixture; no full-budget numerical qualification."""
    import experiments.forge.gaussian_tasks as gaussian
    from experiments.forge.artifacts import verify_artifacts
    from experiments.forge.state import state_digest
    from benchmarks.toy_audit.reproducibility import reproducible_execution
    value=task();value['id']='gaussian_software_fixture';value['execution']['steps']=2
    candidate=json.loads(next((ROOT/'configs/forge/configurations').glob('bcap-dualnorm--7beb*.json')).read_text())
    request=dict(candidate=candidate,protocol={'seed':0})
    monkeypatch.setattr(gaussian,'validate_task',lambda _:None)
    monkeypatch.setattr(gaussian,'schedule',lambda start,stop:[1,2])
    @reproducible_execution
    def check(*,device):
        receipt=gaussian.run_gaussian(request,value,tmp_path/'software',device)
        evidence=receipt['evidence']
        assert Path(evidence['artifact_root'])==tmp_path/'software/evaluator'
        assert (tmp_path/'software/adapter-receipt.json').is_file()
        verify_artifacts(evidence['artifact_root'],evidence['artifact_manifest'])
        saved=torch.load(Path(evidence['artifact_root'])/'state.pt',weights_only=True,map_location='cpu')
        context,trainer,_=gaussian.build(request,value,device)
        context.load_state_dict(saved)
        assert state_digest(context.state_dict())==state_digest(saved)
        assert trainer.completed_steps==2 and receipt['gaussian_grade']['gate_status']=='INCOMPLETE'
        (Path(evidence['artifact_root'])/'unexpected.txt').write_text('must fail')
        with pytest.raises(ValueError,match='file set changed'):
            verify_artifacts(evidence['artifact_root'],evidence['artifact_manifest'])
    check(device='cuda:1')
