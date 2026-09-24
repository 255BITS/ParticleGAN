"""Declared two eight-update native screens of GAN-only nonlocal proposals."""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reports.toy100 import pr84_prediction_state_filter as replay
from reports.toy100.pr84_critic_refinement_filter import EXPECTED_CAPTURE, grade
from reports.toy100.pr84_critic_refinement_capture import snapshot as complete_snapshot
from reports.toy100.pr84_adversarial_reallocation_candidate import METHOD, adversarial_reallocation

BRANCHES=((1324,1331),(1530,1537))
SOURCES=(
    'reports/toy100/pr84_adversarial_reallocation_filter.py',
    'reports/toy100/pr84_adversarial_reallocation_candidate.py',
    'reports/toy100/reallocation_smoothed_candidate.py',
    'reports/toy100/chamfer_discrete_reallocation.py','reports/toy100/chamfer_pullback.py',
    'reports/toy100/joint_output_pullback.py','reports/toy100/pr84_critic_refinement_capture.py',
    'reports/toy100/pr84_prediction_state_filter.py','reports/toy100/pr84_critic_refinement_filter.py',
    'reports/toy100/pr84_opponent_prediction.py','reports/toy100/pr84_smoothed_candidate.py',
    'reports/toy100/alternating_curvature_scratch.py','reports/toy100/extra_adam_scratch.py',
    'reports/toy100/coverage_fixed_eval.py','benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py','benchmarks/transfer_suite/legacy_noise_adapters.py',
    'benchmarks/transfer_suite/toy100_compatibility.py','particlegan/gan_loss.py',
    'particlegan/grad_regularizers.py','particlegan/particle_prior.py',
    'tests/test_pr84_adversarial_reallocation_candidate.py',
    'configs/toy100/constraints_simple_regularization.json',
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    started=time.perf_counter()
    raw=(args.capture/'selected-states.pt').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED_CAPTURE:
        raise RuntimeError('wrong exact capture-v2 states')
    diagnosis=json.loads((args.capture/'diagnosis.json').read_text())
    if diagnosis['status']!='EXACT_REFERENCE_PARITY' or diagnosis['selected_states_sha256']!=EXPECTED_CAPTURE:
        raise RuntimeError('capture lacks exact original provenance')
    states=torch.load(args.capture/'selected-states.pt',weights_only=True)
    references={row['step']:row for row in diagnosis['rows']}
    hashes={}
    source_dir=args.output/'source';source_dir.mkdir()
    for name in SOURCES:
        data=(ROOT/name).read_bytes();target=source_dir/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    declaration=dict(method=METHOD,sources=hashes,branches=BRANCHES,
        captured_states_sha256=EXPECTED_CAPTURE,
        diagnosis_sha256=hashlib.sha256((args.capture/'diagnosis.json').read_bytes()).hexdigest(),
        scope='two short warm native rejection screens; no full warm/cold authorization',
        proposal='one best donor/native-D-real-point replacement by native paired Rp G loss',
        comparison='same native G batch/output draw/phase1 stencil and accepted D for pre/native/fitted states',
        policy='fit pre-G target; require converged fit and loss below BOTH pre/native; otherwise lower native/rest',
        added_G_objective=None,new_critic_fits=0,seed_changes=0,
        nonlinear_budget=dict(iterations=20,halvings=12,svd_rtol=1e-6,tolerance=1e-5),
        noise_horizon=1200,input_noise_scope='zero only',rates=dict(d=.00425,g=.00425,prior=.0085),
        required='all16 live checks retain8modes/HQ>=.9; every original replay exact; no quality selection',
        moments='once per native outer update/player; no GN moment changes',
        curvature_scope='D/G own bounds cover native proposals only; extra global output jumps are disclosed',
        shared_gate_eligible=False)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED',**declaration)),flush=True)
    config=json.loads((ROOT/SOURCES[-1]).read_text())
    progress={}
    @contextmanager
    def selected(*,task,prediction):
        with adversarial_reallocation(task=task,correction=prediction) as value:
            recorder,_=value
            ordinary_phases=recorder.phases
            def observed_phases(step,opt_d,opt_g,local):
                progress['last_pre_step']=complete_snapshot(local)
                progress['step']=step+1
                yield from ordinary_phases(step,opt_d,opt_g,local)
            recorder.phases=observed_phases
            try:
                yield value
            finally:
                if recorder._local is not None:
                    progress['terminal_state']=complete_snapshot(recorder._local)
    results=[]
    with patch.object(replay.prediction_module,'pr84_opponent_prediction',selected):
        for start,end in BRANCHES:
            variants={}
            for name,switch in (('original','current'),('adversarial','predicted')):
                progress.clear()
                try:
                    value,_=replay.run_local(config,states[start]['pre_step'],start=start,end=end,
                                             opponent=switch,source_dir=source_dir)
                    value['opponent']=name
                    if name=='original':
                        for row in value['accepted_states']:
                            if row['step'] in states:
                                assert row['accepted_state_sha256']==replay.state_hash(states[row['step']]['post_bounded_g'])
                        for row in value['points']:
                            assert row['support']==references[row['step']]['stages']['bounded_joint']
                        for i,row in enumerate(value['dynamics']['records']):
                            assert row==dict(references[start+i]['stages']['record'],outer_step=i+1)
                        value['all_original_controls_exact']=True
                    else:
                        assert value['rng_final_sha256']==variants['original']['rng_final_sha256']
                        assert value['noise']==variants['original']['noise']
                        receipt=value['dynamics']
                        assert receipt['native_G_loss_exact_checks']==8
                        assert receipt['native_batch_checks']==receipt['native_index_and_noise_checks']==24
                        assert receipt['correction_rng_checks']==receipt['correction_owner_checks']==8
                        for row in receipt['corrections']:
                            assert row['final_loss']<=min(row['pre_loss'],row['native_loss'])
                            if row['selected']=='joint_fit':
                                assert row['fit']['status']=='CONVERGED'
                                assert row['final_loss']<min(row['pre_loss'],row['native_loss'])
                    value['local_gate']=grade(value['points'])
                    variants[name]=value
                    torch.save(dict(progress,terminal_scope='after successful host return'),
                               args.output/f'{name}-{start}-{end}-states.pt')
                    (args.output/f'{name}-{start}-{end}.json').write_text(json.dumps(value,allow_nan=False)+'\n')
                    print(json.dumps(dict(event='BRANCH_DONE',variant=name,start=start,end=end,
                        observations=[dict(step=row['step'],modes=row['grade']['modes'],hq=row['grade']['hq']) for row in value['points']],
                        **value['local_gate'])),flush=True)
                except BaseException as error:
                    torch.save(dict(progress,terminal_scope='partial failed update; NONRESTARTABLE'),
                               args.output/f'{name}-{start}-{end}-error-states.pt')
                    (args.output/f'{name}-{start}-{end}.error.json').write_text(json.dumps(dict(error=repr(error),
                        partial_terminal_restartable=False,safe_state='last_pre_step; after native set_step'))+'\n')
                    raise
            results.append(dict(start=start,end=end,variants=variants))
    okay=all(row['variants']['adversarial']['local_gate']['pass_all'] for row in results)
    result=dict(status='PASS' if okay else 'FAIL',declaration=declaration,branches=results,
                seconds=time.perf_counter()-started,shared_gate_eligible=False,warm_eligible=False,
                scope='short falsifier only; even a pass needs a separately declared longer test')
    (args.output/'summary.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE',status=result['status'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
