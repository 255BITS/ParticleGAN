"""Model-free structural controls; these fabricated arrays confer no science."""
from contextlib import nullcontext
from collections import OrderedDict
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def load(name):
    spec=importlib.util.spec_from_file_location('_retest_test_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def observer(): return load('goal_observer')


@pytest.fixture
def protocol(): return load('protocol')


def fake_case():
    return {'media_steps':[1,2],'resolved_recipe':{'synthetic_only':True},
            'original_definition':{'original_host':{'steps':2}}}


class FakeOwner:
    completed_steps=1
    def __init__(self): self.state={'recipe':{'synthetic_only':True},'policy':{'served_source':'averaged'},'x':np.array([1.])}
    def state_dict(self): return deepcopy(self.state)


def recorder(observer,tmp_path,owner,*,state=None,rng=None):
    return observer.Recorder(tmp_path,fake_case(),source_guard=lambda:None,deadline_guard=lambda:None,
                             state_reader=state or (lambda t:t.state_dict()),
                             rng_reader=rng or (lambda:{'python':random.getstate(),'numpy':np.random.get_state()}))


def test_complete_protocol_is_original_full_config(protocol):
    card=protocol.load(ROOT)
    assert len(card['rows'])==19 and sum(protocol.CAPS)==9810
    assert all(r['resolved_recipe']['lr']==.00425 and r['resolved_recipe']['prior_lr_mult']==2 for r in card['rows'])
    assert all(r['resolved_recipe']['serve_average']==4 and r['resolved_recipe']['total_steps'] is None for r in card['rows'])
    assert [r['original_definition']['original_host']['seed'] for r in card['rows']]==[0]*13+[1234]*6
    assert card==protocol.validate(json.loads(json.dumps(card)))


@pytest.mark.parametrize('mutation', ['cap','gate','clock','recipe','id','scope','runtime'])
def test_protocol_tampering_rejected(protocol,mutation):
    card=protocol.load(ROOT)
    if mutation=='cap': card['rows'][0]['proposed_inclusive_allowance_seconds']+=1
    elif mutation=='gate': card['rows'][0]['original_definition']['original_requirements'][1][2]=0
    elif mutation=='clock': card['rows'][0]['media_steps'][-1]=599
    elif mutation=='recipe': card['rows'][0]['resolved_recipe']['output_noise_std']=0
    elif mutation=='id': card['rows'][-1]['id']=card['rows'][0]['id']
    elif mutation=='scope': card['claims']['current26_qualification']=True
    else: card['runtime']['physical_gpu']=0
    with pytest.raises(ValueError): protocol.validate(card)


def test_current_complete_package_equals_successful_source(protocol):
    proof=protocol.source_equivalence(ROOT)
    assert proof['successful_commit']==protocol.SUCCESSFUL_COMMIT
    assert len([p for p in proof['files_sha256'] if p.startswith('particlegan/')])>=30
    assert proof['old_metric_renderer_excluded']


@pytest.mark.parametrize('kind', ['screen','vector','moving'])
def test_overlay_is_reversible_and_adds_no_scientific_call(observer,kind):
    # Markers fabricated from the explicit declaration exercise exact-count and
    # reversibility without importing the real host or an optimizer.
    source='\n'.join(before for before,_ in observer.patches(kind))
    derived,receipt=observer.overlay(source,kind)
    assert observer.remove_overlay(derived,kind)==source
    assert receipt['original_sha256']!=receipt['executed_sha256']
    assert len(receipt['operations'])==len(observer.patches(kind))
    with pytest.raises(ValueError): observer.overlay(source+source,kind)
    with pytest.raises(ValueError): observer.remove_overlay(derived+'\n'+observer.patches(kind)[0][1],kind)


def test_actual_screen_overlay_compiles_and_restores_exact_bytes(observer):
    path=ROOT/'reports/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/validation-ra15/screen_current.py'
    source=path.read_text();derived,proof=observer.overlay(source,'screen')
    ast.parse(derived)
    assert observer.remove_overlay(derived,'screen').encode()==path.read_bytes()
    assert proof['original_sha256']==observer.sha(path)


def test_capture_copies_typed_values_and_preserves_all_rng(observer,tmp_path):
    owner=FakeOwner();r=recorder(observer,tmp_path,owner)
    samples=np.arange(8,dtype=np.float32).reshape(4,2);target=-samples.copy()
    before=observer.typed_digest(owner.state_dict());rng=observer.typed_digest(observer.global_rng_state())
    r.capture(owner,1,samples,target,source='synthetic_only',output_sigma=.029)
    assert observer.typed_digest(owner.state_dict())==before
    assert observer.typed_digest(observer.global_rng_state())==rng
    samples[:]=0
    with np.load(tmp_path/r.records[0]['file'],allow_pickle=False) as a:
        assert np.array_equal(a['samples'],np.arange(8,dtype=np.float32).reshape(4,2))
    assert r.records[0]['selected_source']=='averaged'
    assert r.records[0]['purity']['pure'] is True


@pytest.mark.parametrize('mutation',['state','rng','recipe','clock','duplicate','nonfinite'])
def test_capture_refuses_impure_or_forged_state(observer,tmp_path,mutation):
    owner=FakeOwner();counter=[0]
    def bad_state(t):
        counter[0]+=1
        return {'x':counter[0]}
    def bad_rng():
        counter[0]+=1
        return {'x':counter[0]}
    r=recorder(observer,tmp_path,owner,state=bad_state if mutation=='state' else None,
               rng=bad_rng if mutation=='rng' else None)
    values=np.zeros((4,2))
    if mutation=='recipe': owner.state['recipe']={'changed':True}
    if mutation=='clock': owner.completed_steps=2
    if mutation=='nonfinite': values[0,0]=np.nan
    if mutation=='duplicate': r.capture(owner,1,values,values,source='synthetic_only',output_sigma=.029)
    with pytest.raises(ValueError): r.capture(owner,1,values,values,source='synthetic_only',output_sigma=.029)


def test_ema_clean_and_unselected_clock_capture_no_new_data(observer,tmp_path):
    owner=FakeOwner();r=recorder(observer,tmp_path,owner);observer.configure(r)
    values=np.zeros((4,2))
    observer.primary(None,owner,1,values,values,ema=True)
    observer.primary(None,owner,1,values,values,phase=False)
    observer.primary(None,owner,3,values,values)
    assert r.records==[] and list(r.folder.iterdir())==[]
    with observer.vector_phase(None,owner,1,False,.029,.029):
        observer.vector_reference(values,values,1)
    assert len(r.records)==1
    with pytest.raises(ValueError): observer.vector_reference(values,values,1)


def test_typed_digest_keeps_container_dtype_shape_and_nan_bits(observer):
    assert observer.typed_digest({'x':np.array([1],dtype=np.float32)})!=observer.typed_digest({'x':np.array([1],dtype=np.float64)})
    assert observer.typed_digest([1])!=observer.typed_digest((1,))
    assert observer.typed_digest({'diagnostic':float('nan')})==observer.typed_digest({'diagnostic':float('nan')})
    with pytest.raises(ValueError): observer.typed_digest(object())


def test_public_ordered_model_state_and_version_metadata(observer):
    weights=OrderedDict([('weight',np.ones((2,2))),('bias',np.zeros(2))])
    weights._metadata=OrderedDict([('',{'version':1})])
    state={'models':{'G':weights},'optimizers':[{'state':{0:{'exp_avg':np.zeros(2)}}}]}
    assert observer.typed_digest(state)==observer.typed_digest(deepcopy(state))
    assert observer.typed_digest(weights)!=observer.typed_digest(dict(weights))
    changed=deepcopy(weights);changed._metadata['']['version']=2
    assert observer.typed_digest(changed)!=observer.typed_digest(weights)
    reordered=OrderedDict(reversed(list(weights.items())))
    reordered._metadata=deepcopy(weights._metadata)
    assert observer.typed_digest(reordered)!=observer.typed_digest(weights)


def test_ring_overlay_preserves_one_sample_and_diversity_call(observer):
    before,after=next(p for p in observer.patches('screen') if 'value.sample(4096' in p[0])
    assert before.count('value.sample(')==after.count('value.sample(')==1
    assert before.count('host.diversity(')==after.count('host.diversity(')==1
    assert 'output_noise=True' in after and 'ema=ema' in after and 'generator=isolated' in after


def test_moving_has_four_real_clocks_not_nine(protocol):
    rows=protocol.load(ROOT)['rows']
    assert all(r['media_steps']==[0,500,1000,1500] for r in rows if r['group']=='moving')
    assert next(r for r in rows if r['task']=='ring_shift')['media_steps'][3:5]==[2400,2410]


def test_native_caption_separates_coverage_from_recorded_fidelity(observer):
    point={'pass':True,'acc_accuracy_pass':False,'acc_frozen_pass':True,'acc_passed':False}
    label=observer.clock_caption('native',point)
    assert label=='coverage-read PASS · fidelity FAIL · frozen PASS · combined-read FAIL'
    assert 'fidelity UNAVAILABLE' in observer.clock_caption('native',{'pass':True})
    assert observer.clock_caption('moving',{})=='ungated movie state'
    assert observer.clock_caption('portability',{'pass':False})=='read FAIL'
    moving=observer.clock_caption('moving',{'rule':'modes>=95 and HQ>=.9*pre_turn_hq','pre_turn_hq':.95})
    assert 'gate metadata only' in moving and 'pre-turn HQ 0.95' in moving


@pytest.mark.parametrize('group',['native','moving'])
def test_synthetic_renderer_binds_caption_inputs_and_fixed_window(observer,tmp_path,group):
    # Two fabricated display arrays test renderer receipts, not scientific gates.
    owner=FakeOwner();case=fake_case();case.update(group=group,task='synthetic_only')
    r=observer.Recorder(tmp_path,case,source_guard=lambda:None,deadline_guard=lambda:None,
                        state_reader=lambda t:t.state_dict())
    for step in case['media_steps']:
        owner.completed_steps=step
        samples=np.asarray([[step,0.],[step+1.,1.]])
        r.capture(owner,step,samples,np.asarray([[-1.,0.],[0.,1.]]),source='synthetic_only',output_sigma=.029)
    if group=='native':
        metric_file='metrics.jsonl'
        points=[{'step':step,'pass':True,'acc_accuracy_pass':False,'acc_frozen_pass':True,'acc_passed':False}
                for step in case['media_steps']]
        (tmp_path/metric_file).write_text(''.join(json.dumps(p)+'\n' for p in points))
    else:
        metric_file='frames.npz.verdict.json'
        (tmp_path/metric_file).write_text(json.dumps({'periods':[{'period_end':2,'rule':'synthetic only'}]}))
    pin=observer.sha(tmp_path/metric_file)
    receipt=observer.render(tmp_path,case,'FAIL',guard_callback=lambda:None)
    assert receipt['input_files'][metric_file]==pin
    assert receipt['input_files_bytes'][metric_file]==(tmp_path/metric_file).stat().st_size
    assert receipt['caption_metrics_unchanged'] is True
    assert receipt['frames']==2 and receipt['original_gate']=='FAIL'
    assert receipt['verdict_caption']=='FINAL full-protocol verdict'
    assert receipt['fixed_comparison_limits']==[[-1.2,-.05],[3.2,1.05]]
    assert observer.sha(tmp_path/metric_file)==pin
