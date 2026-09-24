"""Strict continued-state44 gate for fixed16-bank averaged native Adam fields.

The prior independent-point screen tested immediate repair from already-bad
original states. This instead preserves each candidate branch through all
updates, which directly tests stability after avoiding an original collapse.
"""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reports.toy100 import pr84_prediction_state_filter as replay
from reports.toy100.pr84_critic_refinement_filter import BRANCHES, EXPECTED_CAPTURE, grade
from reports.toy100.pr84_mean16_replay_candidate_v2 import METHOD, pr84_mean16_replay_candidate_v2

SOURCES=('reports/toy100/pr84_mean16_continuous_filter_v2.py','reports/toy100/pr84_mean16_replay_candidate_v2.py',
    'reports/toy100/pr84_mean16_replay_candidate.py',
        'reports/toy100/pr84_finite_bank_adam_control.py',
        'reports/toy100/pr84_finite_bank_vr_diagnostic.py', 'reports/toy100/pr84_critic_relaxation.py',
        'reports/toy100/sample_anchor_candidate.py',
        'reports/toy100/sample_anchor_rest_candidate.py','reports/toy100/sample_group_anchor.py',
    'reports/toy100/reallocation_smoothed_candidate.py',
    'reports/toy100/chamfer_discrete_reallocation.py','reports/toy100/chamfer_pullback.py',
    'reports/toy100/joint_output_pullback.py','reports/toy100/pr84_critic_refinement_capture.py',
    'reports/toy100/pr84_prediction_state_filter.py','reports/toy100/pr84_critic_refinement_filter.py',
    'reports/toy100/pr84_opponent_prediction.py','reports/toy100/pr84_smoothed_candidate.py',
    'reports/toy100/alternating_curvature_scratch.py','reports/toy100/extra_adam_scratch.py',
    'reports/toy100/coverage_fixed_eval.py','benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py','particlegan/gan_loss.py','particlegan/grad_regularizers.py',
    'configs/toy100/constraints_simple_regularization.json')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    source_dir=args.output/'source'
    source_dir.mkdir()
    raw=(args.capture/'selected-states.pt').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==EXPECTED_CAPTURE
    diagnosis=json.loads((args.capture/'diagnosis.json').read_text())
    assert diagnosis['status']=='EXACT_REFERENCE_PARITY'
    assert diagnosis['selected_states_sha256']==EXPECTED_CAPTURE
    states=torch.load(args.capture/'selected-states.pt',weights_only=True)
    reference={row['step']:row for row in diagnosis['rows']}
    hashes={}
    for name in SOURCES:
        raw=(ROOT/name).read_bytes()
        target=source_dir/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        hashes[name]=hashlib.sha256(raw).hexdigest()
    declaration=dict(method=METHOD,scope='continued saved-state rejection filter; isolated scratch constructors; original V1 pre-quality harness error retained',
        update_scope='full D/G/prior and both Adam states from a mean16 clone of the actual pre-step; native RNG consumed once',
        sources=hashes,branches=BRANCHES,states_sha256=EXPECTED_CAPTURE,
        required='all44 live checkpoints8modes/HQ>=.9; no warm run on any failure',
        update='fixed16-bank mean D then mean G, actual once-updated Adam metrics, original D3/G.25 bounds and PR84 smoothing',
        averaging='16 independent frozen native-sized banks per role from copied stream states; float64 gradient average rounded to nativefloat32',
        compute_budget='32 useful averaged-field gradient calls per role plus3 originalcallbacks/player; total70/player-fields',
        nominal_rates=dict(g=.00425,d=.00425,prior=.0085),noise_horizon=1200,
        seed=0,shared_gate_eligible=False,
        caveat='short late-state replay adapter; no data objective, target geometry or quality gate in update; not a full-bank VR controller')
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED',**declaration)),flush=True)
    config=json.loads((ROOT/SOURCES[-1]).read_text())

    @contextmanager
    def selected(*,task,prediction):
        with pr84_mean16_replay_candidate_v2(task=task,correction=prediction) as value:
            yield value

    results=[]
    with patch.object(replay.prediction_module,'pr84_opponent_prediction',selected):
        for start,end in BRANCHES:
            variants={}
            for name,switch in (('original','current'),('reallocation','predicted')):
                try:
                    value,_=replay.run_local(config,states[start]['pre_step'],
                        start=start,end=end,opponent=switch,source_dir=source_dir)
                    value['opponent']=name
                    if name=='original':
                        for row in value['accepted_states']:
                            if row['step'] in states:
                                assert row['accepted_state_sha256']==replay.state_hash(states[row['step']]['post_bounded_g'])
                        for row in value['points']:
                            assert row['support']==reference[row['step']]['stages']['bounded_joint']
                        for i,row in enumerate(value['dynamics']['records']):
                            assert row==dict(reference[start+i]['stages']['record'],outer_step=i+1)
                    else:
                        assert value['rng_final_sha256']==variants['original']['rng_final_sha256']
                        assert value['noise']==variants['original']['noise']
                        assert len(value['dynamics']['corrections'])==end-start+1
                    value['local_gate']=grade(value['points'])
                    variants[name]=value
                    (args.output/f'{name}-{start}-{end}.json').write_text(json.dumps(value,allow_nan=False)+'\n')
                    print(json.dumps(dict(event='BRANCH_DONE',variant=name,start=start,end=end,
                        **value['local_gate'])),flush=True)
                except BaseException as error:
                    (args.output/f'{name}-{start}-{end}.error.json').write_text(json.dumps(dict(error=repr(error)))+'\n')
                    raise
            results.append(dict(start=start,end=end,variants=variants))
    okay=all(row['variants']['reallocation']['local_gate']['pass_all'] for row in results)
    result=dict(status='PASS' if okay else 'FAIL',warm_eligible=okay,
        shared_gate_eligible=False,declaration=declaration,branches=results)
    (args.output/'summary.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE',status=result['status'],warm_eligible=okay)),flush=True)


if __name__=='__main__':
    main()
