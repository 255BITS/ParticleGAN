import json
import numpy as np
import pytest
import torch
from experiments.evaluate_gym_state_control import check_disjoint, score, state_metrics, verify_protocol
from lib.gym_data import sha256
from lib.gym_transition import GymTransitionScaler


def test_fresh_worlds_reject_previous_eval_and_collection_overlap():
    episodes=[dict(seed=1)]
    prior=[dict(validation_seeds=[2],test_seeds=[3])]
    assert check_disjoint(episodes,prior,[4,5],[6])==[1,2,3]
    for validation,test in [([1],[6]),([2],[6]),([3],[6]),([4],[4]),([4,4],[6])]:
        with pytest.raises(ValueError): check_disjoint(episodes,prior,validation,test)


def test_state_metrics_keep_continuous_and_contact_scores_separate():
    scaler=GymTransitionScaler(torch.zeros(6),torch.ones(6)*2,torch.zeros(2),torch.ones(2))
    raw=torch.zeros(2,8)
    target=torch.ones(2,8)
    metrics=state_metrics(raw,target,scaler)
    assert metrics['standardized_continuous_mse']==.25
    assert metrics['physical_continuous_mse']==1
    assert metrics['contact_brier']==.25
    assert metrics['contact_bce']==pytest.approx(np.log(2))


def test_protocol_and_trace_tampering_rejected(tmp_path,monkeypatch):
    source=tmp_path/'source.py'; source.write_text('initial')
    protocol=dict(simulator={},sources={str(source):sha256(source)},prior_protocols={},
        source_episodes=str(source),source_episodes_sha256=sha256(source),
        reference_checkpoint=str(source),reference_checkpoint_sha256=sha256(source))
    path=tmp_path/'protocol.json'; path.write_text(json.dumps(protocol))
    monkeypatch.setattr('experiments.evaluate_gym_state_control.simulator_provenance',lambda:{})
    assert verify_protocol(path)==protocol
    source.write_text('changed')
    with pytest.raises(RuntimeError,match='artifact changed'): verify_protocol(path)
    trace=tmp_path/'trace.npz'; trace.write_text('initial')
    cached=dict(checkpoint_sha256=None,protocol_sha256=sha256(path),traces=str(trace),traces_sha256=sha256(trace))
    (tmp_path/'evaluations').mkdir()
    (tmp_path/'evaluations/expert_test.json').write_text(json.dumps(cached))
    trace.write_text('changed')
    with pytest.raises(RuntimeError,match='traces changed'):
        score('expert',None,'test','expert',tmp_path,protocol,'cpu')


def test_auxiliary_diagnostics_use_measured_inputs_without_targets(monkeypatch):
    from experiments.evaluate_gym_state_control import auxiliary_metrics
    scaler=GymTransitionScaler(torch.zeros(6),torch.ones(6),torch.zeros(2),torch.ones(2))
    records=dict(states=np.zeros((2,8),np.float32),actions=np.ones((2,2),np.float32),
        next_states=np.concatenate([np.ones((2,6)),np.zeros((2,2))],1).astype(np.float32),
        terrain=np.zeros((2,11),np.float32))
    seen=[]
    def predict(bundle,states,terrain):
        seen.append((states.copy(),terrain.copy()))
        return torch.zeros(len(states),18), None
    monkeypatch.setattr('lib.gym_state_control.predict_state_control',predict)
    metrics=auxiliary_metrics(dict(scaler=scaler,device='cpu'),records)
    assert metrics['next_state']['standardized_continuous_mse']==1
    assert metrics['persistence_next_state']['standardized_continuous_mse']==1
    assert metrics['standardized_action_mse']==1
    np.testing.assert_array_equal(seen[0][0],records['states'])
    records['actions'] *= -1
    records['next_states'][:,:6] *= 2
    auxiliary_metrics(dict(scaler=scaler,device='cpu'),records)
    np.testing.assert_array_equal(seen[0][0],seen[1][0])
    np.testing.assert_array_equal(seen[0][1],seen[1][1])
