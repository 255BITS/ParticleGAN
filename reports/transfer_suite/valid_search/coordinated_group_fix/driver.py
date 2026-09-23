"""Research-only relative optimizer recipe across native behavioral hosts.

Each existing host keeps its target, architecture, budget and base LR/support.
The recipe applies the same G/D/particle LR factors and Adam betas everywhere.
ParticlePrior parameters are split out of mixed optimizer groups without changing
losses. Existing opt_p names identify direct particle-only hosts.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from particlegan import ParticlePrior
from benchmarks import learned_lr_evaluation as bridge
from benchmarks.smart_descent import evaluate, study
from benchmarks.transfer_suite import image_tasks, suite, vector_tasks
from benchmarks.transfer_suite.protocol import required_tasks, test_verdict

OUT=Path('/tmp/pr36-valid-recipe-host-group-fix')
CARD=dict(name='b999_lr075_d2_p30',betas=[0.,.999],g_factor=.75,d_factor=1.,prior_factor=2.25)


@contextmanager
def recipe_context(card, applied):
    prior_ids=set()
    original_prior=ParticlePrior.__init__
    original_adam=torch.optim.Adam.__init__
    original_role=bridge.optimizer_role
    original_control=evaluate.FixedControl
    def prior_init(self,*args,**kwargs):
        original_prior(self,*args,**kwargs)
        prior_ids.update(id(p) for p in self.parameters())
    def adam_init(self,params,*args,**kwargs):
        params=list(params)
        if not params or not isinstance(params[0],dict): params=[dict(params=params)]
        groups=[]
        for group in params:
            values=list(group['params'])
            for is_prior in (False,True):
                selected=[p for p in values if (id(p) in prior_ids)==is_prior]
                if selected: groups.append(dict(group,params=selected,_research_prior=is_prior,betas=tuple(card['betas'])))
        args=list(args)
        if len(args)>=2: args[1]=tuple(card['betas'])
        else: kwargs['betas']=tuple(card['betas'])
        original_adam(self,groups,*args,**kwargs)
    def role(optimizer,locals_):
        result=original_role(optimizer,locals_)
        if locals_.get('opt_p') is optimizer:
            for group in optimizer.param_groups: group['_research_prior']=True
        return result
    class RecipeControl(original_control):
        def step(self,optimizer,completed_updates,role):
            if optimizer not in self.base_rates:
                for group in optimizer.param_groups:
                    effective_role='d' if role=='d' else 'prior' if group.get('_research_prior') else 'g'
                    original_lr=group['lr']
                    group['lr'] *= card[f'{effective_role}_factor']
                    applied.append(dict(role=effective_role,base_lr=original_lr,lr=group['lr'],
                                        betas=list(group['betas']),parameters=sum(p.numel() for p in group['params'])))
            super().step(optimizer,completed_updates,role)
    with ExitStack() as stack:
        stack.enter_context(patch.object(ParticlePrior,'__init__',prior_init))
        stack.enter_context(patch.object(torch.optim.Adam,'__init__',adam_init))
        stack.enter_context(patch.object(bridge,'optimizer_role',role))
        for module in (evaluate,vector_tasks,image_tasks):
            stack.enter_context(patch.object(module,'FixedControl',RecipeControl))
        yield


def cleaned(v):
    if isinstance(v,dict): return {k:cleaned(x) for k,x in v.items() if k not in ('seconds','controller_seconds','stable_from_seconds','confirmed_seconds','created_at','protocol')}
    if isinstance(v,list): return [cleaned(x) for x in v]
    return v


def main():
    OUT.mkdir(exist_ok=False);(OUT/'episodes').mkdir();torch.set_num_threads(1)
    protocol=suite.snapshot(OUT)
    source=Path(__file__).read_bytes();(OUT/'driver.py').write_bytes(source)
    protocol['driver_sha256']=hashlib.sha256(source).hexdigest()
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    policy=vector_tasks.fixed_policy()
    parity=[]
    checks=[]  # Exact vector/image parity was verified in the original run; this corrects AE-only group overrides.
    for runner,name,card,path in checks:
        module=vector_tasks if runner=='vector' else image_tasks
        spec=next(dict(s) for s in module.TASKS if s['name']==name)
        if runner=='image': spec.update(architecture='residual_upsample',width=16,adam_betas=[0.,.999])
        print('PARITY',name,flush=True)
        applied=[]
        with recipe_context(card,applied): result=module.run_episode(spec,policy,fixed=True)
        expected=json.loads(gzip.decompress(path.read_bytes()))['result']
        match=cleaned(result)==cleaned(expected)
        parity.append(dict(name=name,exact=match,reference=str(path),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),applied=applied))
        (OUT/'parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        if not match:
            (OUT/f'parity_{name}.json').write_text(json.dumps(dict(result=result,expected=expected),indent=2)+'\n')
            raise AssertionError(f'Numerical parity failed: {name}')
    rows=[]
    jobs=[('required',s) for s in required_tasks() if s['name']=='ae_gan_hold']
    for runner,spec in jobs:
        suite.verify_source(protocol);print('START',runner,spec['name'],flush=True);applied=[]
        with recipe_context(CARD,applied):
            result=evaluate.fixed_toy(spec['name'],policy) if runner=='required' else image_tasks.run_episode(spec,policy,fixed=True)
        verdict=test_verdict(spec,result)
        row=dict(candidate=CARD,base=asdict(study.BASE),runner=runner,spec=spec,policy=policy,applied=applied,result=result,verdict=verdict)
        raw=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode()
        file=f'episodes/{CARD["name"]}__{spec["name"]}.json.gz';(OUT/file).write_bytes(gzip.compress(raw,mtime=0))
        rows.append({k:v for k,v in row.items() if k!='result'}|dict(artifact=file,uncompressed_sha256=hashlib.sha256(raw).hexdigest(),live=result.get('live'),ema=result.get('ema'),seconds=result['seconds']))
        (OUT/'index.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n')
        lines=['# Shared relative optimizer recipe: required and image verification','',
               'Rp logistic,b_cap3/κ1.25,prior regularization.05,no L2. Adam(0,.999); relative to each host baseline: G LR×.75,D LR×1,particle LR×2.25. Architecture is unchanged for required hosts and residual16 for images. Seed0, complete24 live checks, final5 passing. EMA separate.','',
               '| Host | Sustained live | Confirmed step | Final failing metrics |','| --- | --- | ---: | --- |']
        for r in rows:
            fails=', '.join(m['metric'] for m in r['verdict'].get('metrics',[]) if m['status']!='PASS') or '—'
            lines.append(f"| {r['spec']['name']} | {r['verdict']['status']} | {r['verdict'].get('convergence',{}).get('confirmed_step')} | {fails} |")
        (OUT/'README.md').write_text('\n'.join(lines)+'\n')
        print('DONE',spec['name'],verdict['status'],result.get('live'),result.get('error'),flush=True)
    suite.verify_source(protocol)


if __name__=='__main__': main()
