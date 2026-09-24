"""One real neural update with the D bank conditioned to omit component0.

The G real bank and fixed eight-mode target are unchanged. Replay every
native phase identically. This is a rare same-target minibatch diagnostic.
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
from reports.toy100.pr84_critic_refinement_capture import _sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--cold',type=Path,required=True)
    p.add_argument('--hold',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    entrypoint=own.PRESTART_FACTORY
    factory,method,filename=own.load_factory(ROOT,entrypoint)
    _,declaration,_,config=own.require_cold(args.cold,ROOT,factory,entrypoint,method,filename)
    hd=json.loads((args.hold/'declaration.json').read_text())
    saved,hold=own.require_hold(args.hold,args.cold,hd)
    for name,digest in hd['source'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    recipe,noise,_=declared_recipe(config)
    args.output.mkdir(parents=True,exist_ok=False)
    sources=dict(hd['source']);name='reports/toy100/anchor_missing_batch_neural.py'
    sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in sources:
        dest=args.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    report=dict(scope='one actual native neural update2401; only its D real128 bank conditioned absent component0',
        event_probability=(7/8)**128,initial_state_sha256=_sha(saved),source=sources,branches={})
    for omitted in (False,True):
        calls=[]
        @contextmanager
        def selected(**kwargs):
            with factory(**kwargs) as (recorder,generated):
                ordinary=mode_hold.sample_ring
                def sample(means,n,sigma,stream):
                    is_d=len(recorder.phase_samples)==0
                    if omitted and is_d:
                        assert means.shape==(8,2) and n==128
                        means=means[1:]
                    value=ordinary(means,n,sigma,stream)
                    calls.append(dict(phase=recorder.phase,is_d=is_d,conditioned=omitted and is_d))
                    return value
                with patch.object(mode_hold,'sample_ring',sample):yield recorder,generated
        branch,generated=own.run_bound(saved,recipe,noise,selected,completed=2400,target=2401)
        label='omitted_D_bank' if omitted else 'ordinary'
        assert len(calls)==6 and sum(row['conditioned'] for row in calls)==3*omitted
        torch.save(branch['state'],args.output/(label+'-state.pt'))
        report['branches'][label]=dict(receipt=branch['receipt'],dynamics=branch['dynamics'],
            final_state_sha256=_sha(branch['state']),rng_final_sha256=_sha(branch['state']['rng']),
            data_calls=calls,generated_source_sha256=hashlib.sha256(generated.encode()).hexdigest())
    assert report['branches']['ordinary']['rng_final_sha256']==report['branches']['omitted_D_bank']['rng_final_sha256']
    (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:dict(grade=row['receipt']['checkpoints'][0],
        selected=row['dynamics']['corrections'][0]['selected'],
        fit=row['dynamics']['corrections'][0]['fit']['status'],
        groups=len(row['dynamics']['corrections'][0]['centers'])) for name,row in report['branches'].items()},indent=2))


if __name__=='__main__':main()
