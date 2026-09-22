"""Rare-mode pointwise critic ensemble; fixed formulation and recipe."""
from copy import deepcopy
from functools import partial
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from particlegan import GradientPenalty
from benchmarks.transfer_suite import suite, vector_tasks
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.smooth_critic_research import SmoothFourierCritic

OUT=Path(__file__).resolve().parent
CARDS=[dict(name=f'ensemble{k}_softplus{beta}_d96',features='axis',activation='softplus',
            beta=float(beta),branches=k,width=96,combination='arithmetic_mean')
       for k in (2,4) for beta in (5,6)]


class EnsembleCritic(nn.Module):
    """Pointwise mean of independently initialized smooth critics; one D optimizer/loss."""
    def __init__(self,in_dim=2,hidden_dim=64,n_hidden=2,fourier=2,*,architecture):
        super().__init__()
        self.branches=nn.ModuleList([SmoothFourierCritic(in_dim,hidden_dim,n_hidden,fourier,architecture=architecture)
                                     for _ in range(architecture['branches'])])

    def forward(self,x):
        return torch.stack([branch(x) for branch in self.branches]).mean(0)


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    torch.set_num_threads(1)
    (OUT/'episodes').mkdir(exist_ok=False)
    protocol=suite.snapshot(OUT)
    protocol['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(OUT/'protocol.json',protocol)
    write(OUT/'plan.json',dict(cards=CARDS,selection='Rare first; every sustained rare winner runs all remaining five data toys. Original per-toy budgets; no extra steps or seeds.'))
    declared={s['name']:s for s in vector_tasks.TASKS if s['tier']=='ranking'}
    records=[];checks=[]
    for card in CARDS:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            d=EnsembleCritic(hidden_dim=card['width'],architecture=card)
            real=torch.randn(16,2);fake=torch.randn(16,2)
            penalty=GradientPenalty('b_cap',coeff=3.,kappa=1.25)(d,real,fake)
            (d(real).mean()+penalty).backward()
            assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)
            checks.append(dict(card=card,parameters=sum(p.numel() for p in d.parameters()),penalty=float(penalty.detach()),finite_gradients=True))
    write(OUT/'architecture_checks.json',checks)
    def run(card,name,phase):
        suite.verify_source(protocol)
        original=deepcopy(declared[name]);spec=dict(original,d_hidden=card['width'],research_discriminator=card)
        print('START',card['name'],name,flush=True)
        policy=vector_tasks.fixed_policy()
        with patch.object(vector_tasks,'SimpleMLPDiscriminator',partial(EnsembleCritic,architecture=card)):
            result=vector_tasks.run_episode(spec,policy,fixed=True)
        verdict=test_verdict(spec,result)
        value=dict(candidate=dict(name=card['name'],architecture=card),phase=phase,original_spec=original,spec=spec,
                   policy=policy,result=result,verdict=verdict,source_sha256=protocol['source_sha256'])
        raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode();file=f'episodes/{card["name"]}__{name}.json.gz'
        (OUT/file).write_bytes(gzip.compress(raw,mtime=0))
        records.append({k:v for k,v in value.items() if k!='result'}|dict(artifact=file,uncompressed_sha256=hashlib.sha256(raw).hexdigest(),live=result.get('live'),ema=result.get('ema'),seconds=result['seconds']))
        write(OUT/'index.json',dict(records=records))
        lines=['# Pointwise critic ensemble search','',
               'Same Rp logistic/b_cap3/kappa1.25/prior-reg.05, original beta99/LRs/G/256 particles/batch128. Only D architecture varies: arithmetic mean of 2 or 4 pointwise branches, one unchanged loss and optimizer. Extra D capacity is explicit. No data-derived statistics or labels. Seed0; final5 of24 live checks; EMA separate.','',
               '| D architecture | Toy | Sustained live | Final passing checks | Confirmed step | Final failing metrics |',
               '| --- | --- | --- | ---: | ---: | --- |']
        for r in records:
            v=r['verdict'];fails=', '.join(f"{m['metric']}={m['value']:.5g}" for m in v.get('metrics',[]) if m['status']!='PASS') or '—'
            c=v.get('convergence',{});lines.append(f"| {r['candidate']['name']} | {r['spec']['name']} | {v['status']} | {c.get('passing_suffix')} | {c.get('confirmed_step')} | {fails} |")
        (OUT/'README.md').write_text('\n'.join(lines)+'\n')
        print('DONE',card['name'],name,verdict['status'],result.get('live'),result.get('error'),flush=True)
        return verdict['passed']
    winners=[]
    for card in CARDS:
        if run(card,'vector_unequal_mass','rare_screen'):winners.append(card)
    write(OUT/'selection.json',dict(winners=winners))
    for card in winners:
        for name in declared:
            if name!='vector_unequal_mass':run(card,name,'full_data_validation')
    suite.verify_source(protocol)


if __name__=='__main__':main()
