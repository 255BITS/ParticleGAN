#!/usr/bin/env python
"""TOML/YAML runner for the examples/100gaussians.py problem on the shared toy runner.

Config recipe fields become a particlegan Recipe; the runner owns the loop."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import read_config, recipe_defaults
from experiments.train_denoising import render, write_json
from experiments.run_grid import code_provenance
from lib.denoising_toy import GaussianGrid, grid_metrics
from particlegan import get_recipe
from benchmarks.toy_runner import ToyRun

DEFAULTS = {
    **recipe_defaults('100gaussians'),
    'fourier': 2,
    'log_interval': 1000,
    'snapshot_interval': 1000000,
    'seed': 1234,
    'prior_kind': 'particles',
    'sigma_rel': 0.0,
    'standardize': False,
    'particle_lr_multiplier': 1.0,
    'particle_beta1': None,
    'mog_metrics': False,
    'mog_pass_criteria': None,
    'reg_sync_stats': True,
    'fused_adam': False,
    'final_samples': 20000,
    'save_checkpoint': True,
    'out_dir': 'results/100gaussians/default',
}


def training_recipe(cfg):
    """The config's recipe fields as a particlegan Recipe (the runner builds everything from it)."""
    prior_betas=None if cfg['particle_beta1'] is None else (cfg['particle_beta1'],cfg['beta2'])
    return get_recipe(z_dim=cfg['z_dim'],num_particles=cfg['num_particles'],batch_size=cfg['batch_size'],
                      total_steps=cfg['epochs']*cfg['steps_per_epoch'],lr=cfg['lr'],d_lr_mult=cfg['d_lr_mult'],
                      prior_lr_mult=cfg['prior_lr_mult']*cfg['particle_lr_multiplier'],betas=(cfg['beta1'],cfg['beta2']),
                      prior_betas=prior_betas,reg_coeff=cfg['reg_coeff'],reg_kappa=cfg['reg_kappa'],
                      reg_every=cfg['reg_every'],prior_reg=cfg['lambda_ep'],ema_decay=cfg['ema_decay'],
                      lr_anneal_start=cfg['lr_anneal_start'],lr_floor=cfg['lr_floor'])


def train(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment runner')
    for key in ('epochs','steps_per_epoch','batch_size','num_particles','log_interval','snapshot_interval','final_samples'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    removed=[k for k in ('reg_arm','loss_type','gan_mode','reg_method','reg_fd_eps') if k in cfg]
    if removed:
        raise ValueError(f'{removed} were removed: the objective and critic penalty are the recipe default')
    unsupported=[k for k in ('mog_metrics','fused_adam') if cfg[k]]
    if unsupported:
        # mog_metrics read the in-loop D gap / t_cover of the old hand-written loop;
        # the shared toy runner owns the loop and the recipe owns the optimizers.
        raise ValueError(f'{unsupported} are not supported on the shared toy runner (examples/100gaussians.py)')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/cfg['out_dir'];out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        raise FileExistsError(out)
    provenance=code_provenance(__file__,sys.executable)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    write_json(out/'provenance.json',provenance)
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in provenance['sources']:
            archive.write(ROOT/name,name)
    env={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
         'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'tf32':False}
    write_json(out/'environment.json',env)
    spec=importlib.util.spec_from_file_location('particle_100gaussians',ROOT/'examples/100gaussians.py')
    example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)
    problem=example.Gaussians100(cfg['prior_kind'],fourier=cfg['fourier'],sigma_rel=cfg['sigma_rel'],standardize=cfg['standardize'])
    start=time.perf_counter()
    toy=ToyRun(problem,recipe=training_recipe(cfg),seed=cfg['seed'],device='cuda')
    steps=cfg['epochs']*cfg['steps_per_epoch']
    for _ in range(steps):
        losses=toy.step()
        if toy.completed_steps%cfg['log_interval']==0 or toy.completed_steps==steps:
            print(json.dumps({'step':toy.completed_steps,**toy.measure(ema=True),
                              **{k:float(v) for k,v in losses.items() if k!='step'}},default=float),flush=True)
    torch.cuda.synchronize()
    train_seconds=time.perf_counter()-start
    g,prior=toy.ema_nets.generator.eval(),toy.ema_nets.prior.eval()
    grid=GaussianGrid('cuda',.03,1)
    n=cfg['final_samples'];c=torch.zeros(n,device='cuda',dtype=torch.long)
    rng=torch.Generator('cuda').manual_seed(cfg['seed']+999)
    with torch.no_grad():
        x=g(prior.sample(n,generator=rng)[0])
        real=grid.sample(c,rng)
        final=grid_metrics(x,c,grid,real)
        floor=grid_metrics(grid.sample(c,rng),c,grid,grid.sample(c,rng))
        final['unique_outputs']=len(torch.unique(x,dim=0))
    np.savez_compressed(out/'final_samples.npz',x=x.cpu().numpy(),c=c.cpu().numpy())
    render(out,x.cpu().numpy(),c.cpu().numpy(),grid,None)
    if cfg['save_checkpoint']:
        torch.save({'config':cfg,'G':g.state_dict(),'prior':prior.state_dict(),'run':toy.state_dict()},out/'final.pt')
    import subprocess
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    summary={'git_sha':git_sha,'config':cfg,'final':final,'reference_floor':floor,'train_seconds':train_seconds,
             'samples_per_second':steps*cfg['batch_size']/train_seconds,
             'total_seconds':time.perf_counter()-start,'environment':env,'provenance':provenance}
    write_json(out/'summary.json',summary)
    write_json(out/'metrics.json',{'step':steps,**final})
    print(f"COMPLETE steps={steps} modes={final['modes']} hq={final['hq']:.4f} samples/s={summary['samples_per_second']:.1f}",flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/100gaussians/default.toml'))
    args=parser.parse_args();user=read_config(args.config)
    if not isinstance(user,dict) or set(user)-set(DEFAULTS):
        raise ValueError('config must be a mapping with known keys')
    train({**DEFAULTS,**user})

if __name__=='__main__':
    main()
