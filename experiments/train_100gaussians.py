#!/usr/bin/env python
"""TOML/YAML runner for the examples/100gaussians.py problem on the shared toy runner.

Config recipe fields become a particlegan Recipe; ToyRun owns the update. This
driver only observes it: JSON log lines every log_interval, the MoG study
metrics (mog_metrics), and EMA sample snapshots (step 0, every
snapshot_interval updates and every epoch end)."""
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
from lib.toy_models import sample_100gaussians
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


@torch.no_grad()
def save_fake_scatter(generator, prior, filename, real, fixed_eps=None, n_fake=4096):
    """Real samples vs. fakes from a fixed first-n subset of particles (a consistent frame series)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_fake = min(n_fake, prior.num_particles)
    kwargs = {'eps': fixed_eps[:n_fake]} if fixed_eps is not None else {}
    device = real.device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        fake = generator(prior.sample(n_fake, fixed_first_n=True, **kwargs)[0]).cpu()
    real = real.cpu()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(real[:, 0], real[:, 1], s=4, alpha=.2, label='real')
    ax.scatter(fake[:, 0], fake[:, 1], s=4, alpha=.8, label='fake')
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect('equal', 'box')
    ax.legend(loc='upper right'); ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('100 Gaussians: real vs. model samples')
    fig.tight_layout(); fig.savefig(filename, dpi=150); plt.close(fig)


@torch.no_grad()
def critic_gap(toy, n, seed):
    """mean D(real) - mean D(fake): the live critic on a fixed-seed batch of real data and live-G
    fakes, without training noise. Own generators, so training RNG is untouched."""
    stream = torch.Generator(toy.device).manual_seed(seed + 2999)
    modules = [*toy.nets.generator_side(), *toy.critics.values()]
    flags = [(m, m.training) for root in modules for m in root.modules()]
    try:
        for module in modules:
            module.eval()
        real = toy.problem.real(n, stream)
        fake = toy.problem.fake(toy.nets, n, stream, None).x
        return float(sum(c(real).mean() - c(fake).mean() for c in toy.critics.values()))
    finally:
        for module, flag in flags:
            module.training = flag


def train(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment runner')
    for key in ('epochs','steps_per_epoch','batch_size','num_particles','log_interval','snapshot_interval','final_samples'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    removed=[k for k in ('reg_arm','loss_type','gan_mode','reg_method','reg_fd_eps') if k in cfg]
    if removed:
        raise ValueError(f'{removed} were removed: the objective and critic penalty are the recipe default')
    if cfg['fused_adam']:
        # The shared toy runner builds its optimizers from the recipe, which has no fused option.
        raise ValueError('fused_adam is not supported on the shared toy runner (examples/100gaussians.py)')
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
    toy=ToyRun(problem,recipe=training_recipe(cfg),seed=cfg['seed'],device='cuda')
    steps,seed=cfg['epochs']*cfg['steps_per_epoch'],cfg['seed']
    learnable=problem.prior_kind in ('particles','mog')
    initial_raw_std=float(toy.nets.prior.z.detach().std()) if learnable else None
    # Fixed visualization data (and, for a jittered MoG, fixed jitter) for a consistent frame series.
    real_viz=sample_100gaussians(8192,toy.device,generator=torch.Generator(toy.device).manual_seed(seed+1))
    fixed_eps=None
    if problem.prior_kind=='mog' and cfg['sigma_rel']>0:
        fixed_eps=torch.randn(min(4096,cfg['num_particles']),cfg['z_dim'],device=toy.device,
                              generator=torch.Generator(toy.device).manual_seed(seed+4))
    ema_g,ema_prior=toy.ema_nets.generator,toy.ema_nets.prior
    t_cover=d_gap=None
    if cfg['mog_metrics']:
        from lib.mog_metrics import evaluate
        metric_path=out/'metrics.jsonl';metric_path.write_text('')
    save_fake_scatter(ema_g,ema_prior,out/'samples_step_000000.png',real_viz,fixed_eps)
    start=time.perf_counter();maintenance=0.0
    for _ in range(steps):
        losses=toy.step()
        step=toy.completed_steps
        log_due=step%cfg['log_interval']==0 or step==steps
        snapshot_due=step%cfg['snapshot_interval']==0
        epoch_end=step%cfg['steps_per_epoch']==0
        if not (log_due or snapshot_due or epoch_end):
            continue
        torch.cuda.synchronize();paused=time.perf_counter()
        if log_due:
            row={'step':step,**toy.measure(ema=True),**{k:float(v) for k,v in losses.items() if k!='step'}}
            if cfg['mog_metrics']:
                d_gap=critic_gap(toy,cfg['batch_size'],seed)
                mog,_,_,_=evaluate(ema_g,ema_prior,20000,seed,initial_raw_std,pass_criteria=cfg['mog_pass_criteria'])
                if t_cover is None and mog['modes']==100 and mog['hq']>=.9:
                    t_cover=step
                mog.update(step=step,d_gap=d_gap,t_cover=t_cover,
                           raw_std_live=float(toy.nets.prior.z.detach().std()) if learnable else None)
                with metric_path.open('a') as stream:
                    stream.write(json.dumps(json_safe(mog),allow_nan=False)+'\n')
                row['mog']={k:mog[k] for k in ('hq','width_ratio','kl_balance','passed','d_gap','t_cover')}
            print(json.dumps(row,default=float),flush=True)
        if snapshot_due:
            save_fake_scatter(ema_g,ema_prior,out/f'samples_step_{step:06d}.png',real_viz,fixed_eps)
        if epoch_end:
            epoch=step//cfg['steps_per_epoch']-1
            save_fake_scatter(ema_g,ema_prior,out/f'samples_epoch_{epoch:04d}.png',real_viz,fixed_eps)
        torch.cuda.synchronize();maintenance+=time.perf_counter()-paused
    torch.cuda.synchronize()
    train_seconds=time.perf_counter()-start-maintenance
    g,prior=toy.ema_nets.generator.eval(),toy.ema_nets.prior.eval()
    grid=GaussianGrid('cuda',.03,1)
    n=cfg['final_samples'];c=torch.zeros(n,device='cuda',dtype=torch.long)
    rng=torch.Generator('cuda').manual_seed(cfg['seed']+999)
    if cfg['mog_metrics']:
        from lib.mog_metrics import evaluate, geometry
        final,components,x,real=evaluate(g,prior,n,seed,initial_raw_std,component_detail=True,pass_criteria=cfg['mog_pass_criteria'])
        final.update(t_cover=t_cover,d_gap=d_gap)
        live=geometry(toy.nets.prior)
        final['raw_std_live']=live['raw_std']
        final['raw_std_live_ratio']=live['raw_std']/initial_raw_std if live['raw_std'] is not None else None
        final['raw_std_drift_flag']=bool(final['raw_std_drift_flag'] or (final['raw_std_live_ratio'] is not None and not .5<=final['raw_std_live_ratio']<=2))
        write_json(out/'components.json',components)
        np.savez_compressed(out/'final_samples.npz',x=x.cpu().numpy(),real=real.cpu().numpy())
        floor={key[:-5]:value for key,value in final.items() if key.endswith('_real')}
    else:
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
