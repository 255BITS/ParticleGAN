"""Three native updates: confirm two real banks, then stress one D bank.

All branches restore the same exact completed update-2400 host state. The first
two D banks are ordinary native draws; the third alone is conditioned to omit
one component or contain a single noisy sample from it. No label or HQ score
enters the candidate policy. The conditioning is diagnostic, not a training
policy or an unconditional event probability estimate.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100 import sample_anchor_own_state_probe as own
from reports.toy100.sample_anchor_memory_candidate import METHOD,sample_anchor_memory_candidate
from reports.toy100.sample_anchor_memory_filter import SOURCES as MEMORY_SOURCES
from reports.toy100.pr84_critic_refinement_capture import _sha

SOURCE_NAMES=set(MEMORY_SOURCES)|{
    'reports/toy100/sample_group_two_bank_native3.py',
    'reports/toy100/sample_anchor_own_state_probe.py',
    'reports/toy100/pr84_model_error_recovery.py',
    'reports/toy100/pr84_critic_refinement_resume.py',
    'benchmarks/transfer_suite/toy100_compatibility.py',
}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def singleton_tail(ordinary,recorder,means,n,sigma,stream):
    """Condition labels after native draws, preserving both RNG calls."""
    native=ordinary(means,n,sigma,stream)
    nearest=torch.cdist(native,means).argmin(1)
    noise=native-means[nearest]
    position=int(torch.linalg.vector_norm(noise,dim=1).argmax())
    labels=nearest.remainder(7)+1
    labels[position]=0
    adjusted=means[labels]+noise
    if not torch.equal(recorder.phase_samples[-1],native):
        raise RuntimeError('native sample capture was not the current D bank')
    recorder.phase_samples[-1]=adjusted.detach().clone()
    return adjusted,dict(singleton_position=position,
                         singleton_tail_norm=float(torch.linalg.vector_norm(noise[position])),
                         conditioned_component0_count=1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True,exist_ok=False)
    saved_bytes=a.state.read_bytes()
    saved=torch.load(a.state,weights_only=True,map_location='cpu')
    if saved['noise']['step_calls']!=2400 or saved['noise_policy']['total_steps']!=1200:
        raise RuntimeError('input is not the exact passed update-2400 state')
    config_bytes=a.config.read_bytes()
    config=json.loads(config_bytes)
    if config.get('lr_floor')!=1. or config.get('lr_anneal_start')!=0.:
        raise RuntimeError('recipe must retain constant nominal rates')
    recipe,noise,_=declared_recipe(config)
    source={}
    for name in sorted(SOURCE_NAMES):
        raw=(ROOT/name).read_bytes()
        source[name]=sha(raw)
        target=a.output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
    declaration=dict(method=METHOD,scope='exact saved update2400, three native updates only',
        input_state_path=str(a.state),input_file_sha256=sha(saved_bytes),
        input_state_sha256=_sha(saved),config_sha256=sha(config_bytes),
        source=source,branches=['ordinary','omitted_third_D','singleton_third_D'],
        first_two_banks='ordinary and identical in every branch',
        conditioned_bank='third update D real128 only; all three phase replays identical',
        nominal_rates=dict(d=.00425,g=.00425,prior=.0085),noise_horizon=1200,
        learner_first_bank_id=2401,shared_gate_eligible=False)
    (a.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED',declaration=declaration)),flush=True)

    branches={}
    for name in declaration['branches']:
        calls=[]
        @contextmanager
        def selected(**kwargs):
            kwargs['start_step']=2400
            with sample_anchor_memory_candidate(**kwargs) as (recorder,generated):
                ordinary=mode_hold.sample_ring
                def sample(means,n,sigma,stream):
                    index=len(calls)
                    update=index//6
                    phase=index%6//2
                    is_d=index%2==0
                    if update>2 or phase!=recorder.phase or n!=128:
                        raise RuntimeError('native three-update D/G sample call order changed')
                    conditioned=update==2 and is_d and name!='ordinary'
                    detail={}
                    if conditioned and name=='omitted_third_D':
                        value=ordinary(means[1:],n,sigma,stream)
                    elif conditioned and name=='singleton_third_D':
                        value,detail=singleton_tail(ordinary,recorder,means,n,sigma,stream)
                    else:
                        value=ordinary(means,n,sigma,stream)
                    calls.append(dict(update=2401+update,phase=phase,is_d=is_d,
                        conditioned=conditioned,bank_sha256=sha(value.detach().cpu().contiguous().numpy().tobytes()),
                        **detail))
                    return value
                with patch.object(mode_hold,'sample_ring',sample):
                    yield recorder,generated
        branch,generated=own.run_bound(saved,recipe,noise,selected,
                                        completed=2400,target=2403)
        if len(calls)!=18 or len(branch['dynamics']['corrections'])!=3:
            raise RuntimeError('three native updates or phase replays missing')
        if branch['receipt']['actual_adam_updates']!={'d':3,'g':3}:
            raise RuntimeError('Adam moments did not advance once per outer update')
        if not all(row['conditioned']==(row['update']==2403 and row['is_d'] and name!='ordinary')
                   for row in calls):
            raise RuntimeError('conditioning reached a bank outside the third D update')
        for update in range(3):
            group=[row for row in calls if row['update']==2401+update]
            if len(group)!=6 or len({row['bank_sha256'] for row in group if row['is_d']})!=1:
                raise RuntimeError('D bank did not replay identically across phases')
        if name=='singleton_third_D':
            if len({row['singleton_position'] for row in calls if row['conditioned']})!=1:
                raise RuntimeError('singleton tail changed across phase replay')
        branch_state=a.output/f'{name}-final-state.pt'
        torch.save(branch['state'],branch_state)
        branches[name]=dict(receipt=branch['receipt'],dynamics=branch['dynamics'],
            calls=calls,final_state_sha256=_sha(branch['state']),
            final_file_sha256=sha(branch_state.read_bytes()),
            rng_final_sha256=_sha(branch['state']['rng']),
            generated_source_sha256=sha(generated.encode()))
        (a.output/f'{name}.json').write_text(json.dumps(branches[name],allow_nan=False)+'\n')
        print(json.dumps(dict(event='BRANCH',name=name,
            checkpoints=branch['receipt']['checkpoints'],
            statuses=[row['grouping']['observation']['status'] for row in branch['dynamics']['corrections']])),flush=True)
    control=branches['ordinary']
    for name in ('omitted_third_D','singleton_third_D'):
        branch=branches[name]
        if branch['calls'][:12]!=control['calls'][:12]:
            raise RuntimeError('conditioned branch changed one of the two confirmation banks')
        if branch['dynamics']['corrections'][:2]!=control['dynamics']['corrections'][:2]:
            raise RuntimeError('conditioned branch changed bootstrap corrections')
        if branch['dynamics']['records'][:2]!=control['dynamics']['records'][:2]:
            raise RuntimeError('conditioned branch changed bootstrap game updates')
        if branch['receipt']['checkpoints'][:2]!=control['receipt']['checkpoints'][:2]:
            raise RuntimeError('conditioned branch changed bootstrap grade')
        if branch['rng_final_sha256']!=control['rng_final_sha256']:
            raise RuntimeError('conditioned branch consumed extra training randomness')
    summary=dict(status='COMPLETE',declaration=declaration,branches=branches,
        first_two_updates_exact_across_branches=True,
        rng_final_exact_across_branches=True,shared_gate_eligible=False)
    (a.output/'summary.json').write_text(json.dumps(summary,allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE',status='COMPLETE',
        grades={k:v['receipt']['checkpoints'][-1] for k,v in branches.items()})),flush=True)


if __name__=='__main__':main()
