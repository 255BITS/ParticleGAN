#!/usr/bin/env python
"""TOML/YAML runner for the actual examples/100gaussians.py training loop."""
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
from experiments.train_denoising import json_safe, render, write_json
from experiments.run_grid import code_provenance
from lib.denoising_toy import GaussianGrid, grid_metrics
from particlegan.grad_regularizers import GradRegularizer

DEFAULTS = {
    **recipe_defaults('100gaussians'),
    'fourier': 2,
    'log_interval': 1000,
    'snapshot_interval': 1000000,
    'seed': 1234,
    'prior_kind': 'particles',
    'reg_fd_eps': 0.05,
    'reg_sync_stats': True,
    'fused_adam': False,
    'final_samples': 20000,
    'save_checkpoint': True,
    'out_dir': 'results/100gaussians/default',
}


def train(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment runner')
    for key in ('epochs','steps_per_epoch','batch_size','num_particles','log_interval','snapshot_interval','final_samples'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    GradRegularizer(cfg['reg_arm'],cfg['reg_coeff'],lazy_k=cfg['reg_every'],method=cfg['reg_method'],fd_eps=cfg['reg_fd_eps'])
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
    kwargs={k:v for k,v in cfg.items() if k not in ('final_samples','save_checkpoint')}
    start=time.perf_counter()
    result=example.train(**kwargs,device_str='cuda',return_details=True)
    g,prior=result['ema_G'].eval(),result['ema_prior'].eval()
    toy=GaussianGrid('cuda',.03,1)
    n=cfg['final_samples'];c=torch.zeros(n,device='cuda',dtype=torch.long)
    rng=torch.Generator('cuda').manual_seed(cfg['seed']+999)
    with torch.no_grad():
        x=g(prior.sample(n,generator=rng)[0])
        real=toy.sample(c,rng)
        final=grid_metrics(x,c,toy,real)
        floor=grid_metrics(toy.sample(c,rng),c,toy,toy.sample(c,rng))
        final['unique_outputs']=len(torch.unique(x,dim=0))
    np.savez_compressed(out/'final_samples.npz',x=x.cpu().numpy(),c=c.cpu().numpy())
    render(out,x.cpu().numpy(),c.cpu().numpy(),toy,None)
    if cfg['save_checkpoint']:
        torch.save({'config':cfg,'G':g.state_dict(),'prior':prior.state_dict()},out/'final.pt')
    steps=cfg['epochs']*cfg['steps_per_epoch']
    summary={'config':cfg,'final':final,'reference_floor':floor,'train_seconds':result['train_seconds'],
             'samples_per_second':steps*cfg['batch_size']/result['train_seconds'],
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
