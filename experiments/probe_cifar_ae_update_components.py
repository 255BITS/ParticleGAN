#!/usr/bin/env python
"""Counterfactual G versus prior contributions during eight original joint updates.

Independent fixed held-out panel measured immediately before D, after D and
then after G/prior. Hooks restore parameter flags and do not advance training
RNG. Trace output checkpoints are short diagnostic artifacts, not FID scouts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torchvision.datasets import CIFAR10
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import train_cifar_ae_discriminator as trainer
from experiments import diagnose_cifar_ae_discriminator as diag


def run(parent,out):
    ck=torch.load(parent,map_location='cpu',weights_only=False)
    cfg={**trainer.DEFAULTS,**ck['config'],'resume_checkpoint':str(parent.resolve()),'resume_sha256':diag.digest(parent),
         'out_dir':str(out.resolve()),'steps':ck['step']+8,'eval_samples':0,'final_samples':0,
         'eval_interval':1000000,'recon_samples':64,'log_interval':1,'d_warmstart':'','d_warmstart_sha256':''}
    array=CIFAR10(cfg['data_dir'],train=False,download=False).data
    images=torch.from_numpy(array).permute(0,3,1,2).contiguous().cuda()
    panel_stream=trainer.rng(cfg['seed']+80000)
    ids=torch.randperm(len(images),device='cuda',generator=panel_stream)[:512]
    real=images[ids].float()/127.5-1
    latent_state=panel_stream.get_state()
    captured={};rows=[];counter=ck['step']
    build=trainer.build_models;prior_factory=trainer.MoGParticlePrior
    adam_step=torch.optim.Adam.step;adam_zero=torch.optim.Adam.zero_grad
    def models(config):
        result=build(config);captured.update(zip(('g','d','e'),result));return result
    def prior_factory_capture(*args,**kwargs):
        prior=prior_factory(*args,**kwargs);captured['prior']=prior;return prior
    def measure(phase):
        g,d,prior=[captured[k] for k in ('g','d','prior')]
        flags=[(p,p.requires_grad) for m in (g,d,prior) for p in m.parameters()]
        before=trainer.state_hash([g,d,prior])
        cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        g.requires_grad_(False);d.requires_grad_(False);prior.requires_grad_(False)
        stream=trainer.rng(cfg['seed']);stream.set_state(latent_state)
        reals=[];fakes=[];norms=[]
        try:
            for x in real.split(64):
                with torch.no_grad():
                    z,_=prior.sample(len(x),stream);fake=g(z);reals.append(d(x))
                fake.requires_grad_(True);score=d(fake);fakes.append(score.detach())
                gradient,=torch.autograd.grad(score.sum(),fake)
                norms.append(gradient.flatten(1).norm(dim=1).detach())
            r,f=torch.cat(reals),torch.cat(fakes)
            row={'step':counter,'phase':phase,'auc':diag.auc(r,f),'real_minus_fake':float(r.mean()-f.mean()),
                 'fake_input_grad_norm':float(torch.cat(norms).mean())}
            rows.append(row);print('TRACE',json.dumps(row),flush=True)
            assert before==trainer.state_hash([g,d,prior])
            assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
        finally:
            for p,enabled in flags:p.requires_grad_(enabled)
    def zero(optimizer,*args,**kwargs):
        nonlocal counter
        if len(optimizer.param_groups)==1:
            counter+=1;measure('before_D')
        return adam_zero(optimizer,*args,**kwargs)
    def step(optimizer,*args,**kwargs):
        joint = len(optimizer.param_groups) != 1
        def snapshot(module):
            return {key:value.detach().clone() if isinstance(value,torch.Tensor) else value for key,value in module.state_dict().items()}
        if joint:
            g, prior = captured['g'], captured['prior']
            old_g, old_prior = snapshot(g), snapshot(prior)
        value=adam_step(optimizer,*args,**kwargs)
        measure('after_G_prior' if joint else 'after_D')
        if joint:
            new_g, new_prior = snapshot(g), snapshot(prior)
            g.load_state_dict(old_g)
            measure('prior_only_counterfactual')
            g.load_state_dict(new_g)
            prior.load_state_dict(old_prior)
            measure('G_only_counterfactual')
            prior.load_state_dict(new_prior)
        return value
    trainer.build_models=models;trainer.MoGParticlePrior=prior_factory_capture
    torch.optim.Adam.step=step;torch.optim.Adam.zero_grad=zero
    try:
        trainer.train(cfg)
    finally:
        trainer.build_models=build;trainer.MoGParticlePrior=prior_factory
        torch.optim.Adam.step=adam_step;torch.optim.Adam.zero_grad=adam_zero
    assert len(rows)==40
    result={'parent':str(parent),'parent_sha256':diag.digest(parent),'script_sha256':diag.digest(__file__),
            'trainer_sha256':diag.digest(trainer.__file__),'rows':rows,'heldout_panel_samples':512,
            'note':'G-only/prior-only counterfactuals use actual joint Adam updates computed at the same pre-update state, then temporarily roll back the other component. All states restored before next update. Live weights, production precision. Same held-out real images and same prior ID/noise draws at each phase. Probe preserves weights, flags and global RNG. Trace has no comparable FID or training-speed measurement.'}
    trainer.write_json(out/'TRACE.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.checkpoint,a.out)
