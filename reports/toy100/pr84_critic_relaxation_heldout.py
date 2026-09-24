"""Check fitted-critic G guidance on the eight already-reserved heldout batches.

No critic fitting, new seed or live training update occurs. Each G proposal
starts independently from the same captured models and Adam state.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from reports.toy100 import pr84_critic_relaxation as fit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    state_file=args.capture/'selected-states.pt'
    if hashlib.sha256(state_file.read_bytes()).hexdigest()!=fit.STATE_SHA:
        raise RuntimeError('wrong frozen states')
    summary=json.loads((args.fit/'summary.json').read_text())
    if hashlib.sha256(Path(fit.__file__).read_bytes()).hexdigest()!=summary['declaration']['source']['reports/toy100/pr84_critic_relaxation.py']:
        raise RuntimeError('executed fit source changed')
    states=torch.load(state_file,weights_only=True)
    fitted_file=args.fit/'fitted-critics-and-banks.pt'
    fitted=torch.load(fitted_file,weights_only=True)
    recipe,_,_=fit.declared_recipe(json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text()))
    gan=recipe.make_loss()
    source=Path(__file__).read_bytes();(args.output/Path(__file__).name).write_bytes(source)
    declaration=dict(scope='heldout_G_field_check_only',training_candidate=False,shared_gate_eligible=False,
        new_critic_fits=0,heldout_batches_per_state=8,new_seeds=False,
        prior_fit_summary_sha256=hashlib.sha256((args.fit/'summary.json').read_bytes()).hexdigest(),
        fitted_critics_sha256=hashlib.sha256(fitted_file.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source).hexdigest(),states_sha256=fit.STATE_SHA)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    rows=[]
    outer=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        for step in fit.STATES:
            saved=states[step]['post_accepted_d'];generator,critic,prior=fit.modules(saved)
            _,heldout_batches,train,heldout,g_batch=fit.banks(states[step]['pre_step'],saved,generator,prior)
            expected=next(row for row in summary['rows'] if row['step']==step)
            if fit.state_hash(dict(train=train,heldout=heldout,g_batch=g_batch))!=expected['bank_sha256']:
                raise RuntimeError('reconstructed banks changed')
            if fit.state_hash(fitted[step]['critic'])!=expected['fitted_critic_sha256']:
                raise RuntimeError('fitted critic changed')
            variants={}
            for name,weights in (('learned',fit.unwrapped(saved['critic'])),('locally_fitted',fitted[step]['critic'])):
                critic.load_state_dict(weights)
                values=[]
                for number,bank in enumerate(heldout_batches):
                    batch=dict(real=bank['real'],indices=bank['indices'],noise=bank['noise'],sigma=saved['noise']['output_sigma'])
                    value,_,_=fit.g_proposal(saved,critic,batch,gan,step)
                    values.append(dict(batch=number,**value))
                variants[name]=values
            row=dict(step=step,variants=variants)
            rows.append(row);(args.output/f'step-{step}.json').write_text(json.dumps(row,allow_nan=False)+'\n')
            short={}
            for name,values in variants.items():
                short[name]=dict(raw_inward=sum(sum(v['raw_joint_parameter_descent']['radial_work'])<0 for v in values),
                    accepted_inward=sum(sum(v['accepted_joint']['radial_work'])<0 for v in values),
                    min_hq=min(v['grade_after']['hq'] for v in values),
                    passing=sum(v['grade_after']['modes']==8 and v['grade_after']['hq']>=.9 for v in values))
            print(json.dumps(dict(event='HELDOUT_G_DONE',step=step,variants=short)),flush=True)
    if not torch.equal(outer,torch.get_rng_state()):
        raise RuntimeError('heldout checker changed outside RNG')
    (args.output/'summary.json').write_text(json.dumps(dict(declaration=declaration,rows=rows),allow_nan=False)+'\n')


if __name__=='__main__':
    main()
