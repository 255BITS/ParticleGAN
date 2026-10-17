import copy
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.evaluate_gym_sparse_action import (make_controller, score,
    verify_protocol, verify_training_protocol)
from lib.gym_data import sha256


def test_training_must_match_frozen_labels_arrays_and_normalization():
    scaler=SimpleNamespace(state_dict=lambda:{'scale':torch.tensor([1.])})
    p=dict(selection={'labeled_episode_ids':[3,15,53,5,22]}, labeled_data={'count':1010},
        expert_data={'count':9297,'arrays':{'label_mask':{'sha256':'mask'}}})
    config=dict(seed=24003,steps=2500,batch_size=256)
    bundle=dict(scaler=scaler,config=config,provenance=p)
    protocol=dict(training_config={'seed':24003},training_updates=2500,training_batch_size=256,
        supervision=dict(selection=copy.deepcopy(p['selection']),labeled_records=1010,all_state_pairs=9297,
            array_hashes={'label_mask':'mask'},normalization_statistics={'scale':[1.]}))
    verify_training_protocol(bundle,protocol)
    variants=[]
    for field,value in [('labeled_records',1011),('array_hashes',{'label_mask':'changed'}),
                        ('normalization_statistics',{'scale':[2.]})]:
        changed=copy.deepcopy(protocol);changed['supervision'][field]=value;variants.append(changed)
    changed=copy.deepcopy(protocol);changed['training_config']['seed']=1;variants.append(changed)
    for changed in variants:
        with pytest.raises(RuntimeError): verify_training_protocol(bundle,changed)


def test_full_label_reference_uses_existing_format_and_ignores_previous_action(monkeypatch):
    calls=[]
    bundle={'config':{'arm':'probes'}}
    monkeypatch.setattr('lib.gym_state_control.load_state_control_checkpoint',lambda p,d:bundle)
    monkeypatch.setattr('experiments.evaluate_gym_sparse_action.verify_full_label_checkpoint',lambda b,p:calls.append(('verify',p)))
    monkeypatch.setattr('lib.gym_state_control.control_action_details',lambda b,s,c:(s,c))
    fn,loaded=make_controller('full_label','reference.pt','cpu')
    assert loaded is bundle
    assert fn(None,'measured-state','ignored-previous-command','terrain')==('measured-state','terrain')
    assert calls==[('verify','reference.pt')]


def test_sparse_protocol_and_cached_traces_detect_changes(tmp_path,monkeypatch):
    source=tmp_path/'source.py';source.write_text('initial')
    protocol=dict(simulator={},sources={str(source):sha256(source)},prior_protocols={},
        source_episodes=str(source),source_episodes_sha256=sha256(source),
        reference_checkpoint=str(source),reference_checkpoint_sha256=sha256(source))
    path=tmp_path/'protocol.json';path.write_text(json.dumps(protocol))
    monkeypatch.setattr('experiments.evaluate_gym_sparse_action.simulator_provenance',lambda:{})
    assert verify_protocol(path)==protocol
    source.write_text('changed')
    with pytest.raises(RuntimeError,match='artifact changed'): verify_protocol(path)
    trace=tmp_path/'trace.npz';trace.write_text('initial')
    cached=dict(checkpoint_sha256=None,protocol_sha256=sha256(path),traces=str(trace),traces_sha256=sha256(trace))
    (tmp_path/'evaluations').mkdir()
    (tmp_path/'evaluations/expert_test.json').write_text(json.dumps(cached))
    trace.write_text('changed')
    with pytest.raises(RuntimeError,match='traces changed'): score('expert',None,'test','expert',tmp_path,protocol,'cpu')
