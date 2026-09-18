#!/usr/bin/env python
"""Read-only EMA sampling sensitivity and within-particle diversity diagnostic.

Pins the architecture dependency; original weights, prior sigma and checkpoint
are never mutated. Same standard evaluation RNG draws at each noise scale.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'experiments/train_cifar_ae_transgan.py'
BASE_SHA='76a481fe19b79450f89dd6ec2c592ddeb3f8a3095d6cddc35c8f26bc4609248e'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA
from experiments import train_cifar_ae_transgan as base
from experiments.run_grid import code_provenance
from lib.cifar_metrics import FIDEvaluator, uint8_images, PROTOCOL
from particlegan import MoGParticlePrior
from torch_fidelity.metric_fid import fid_statistics_to_metric

DEFAULTS={'checkpoint':'','checkpoint_sha256':'','out_dir':'','samples':50000,
          'noise_scales':[0.,1.,2.], 'batch_size':128,'particles':128,'draws_per_particle':16}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def train(cfg):
    assert set(cfg)==set(DEFAULTS)
    assert cfg['samples']>=2 and cfg['batch_size']>0 and cfg['particles']>1 and cfg['draws_per_particle']>1
    assert 1. in cfg['noise_scales'] and all(np.isfinite(v) and v>=0 for v in cfg['noise_scales'])
    start=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
    path=ROOT/cfg['checkpoint'];assert digest(path)==cfg['checkpoint_sha256']
    ck=torch.load(path,map_location='cpu',weights_only=False)
    for name,expected in ck['sources'].items():assert digest(ROOT/name)==expected,name
    parent={**base.DEFAULTS,**ck['config']}
    torch.manual_seed(parent['seed']);torch.cuda.manual_seed_all(parent['seed'])
    g,d,e=base.build_models(parent);del d,e
    g=g.cuda().eval().requires_grad_(False);g.load_state_dict(ck['ema_G'])
    prior=MoGParticlePrior(parent['num_particles'],parent['z_dim'],sigma_rel=parent['sigma_rel']).cuda().eval().requires_grad_(False)
    prior.load_state_dict(ck['ema_prior']);assert cfg['particles']<=prior.num_particles
    frozen=base.state_hash([g,prior])
    out=ROOT/cfg['out_dir'];out.mkdir(parents=True,exist_ok=True)
    assert not (out/'summary.json').exists(),'fresh directory required'
    provenance=code_provenance(__file__,sys.executable)
    provenance['sources'][str(BASE.relative_to(ROOT))]=BASE_SHA
    base.write_json(out/'provenance.json',provenance)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg))
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name,expected in provenance['sources'].items():
            assert digest(ROOT/name)==expected;archive.write(ROOT/name,name)
    real=torch.from_numpy(CIFAR10(parent['data_dir'],train=True,download=False).data).permute(0,3,1,2).contiguous()
    evaluator=FIDEvaluator(real,parent['fid_cache'],cfg['batch_size'])
    means=prior.means();results=[]
    for scale in cfg['noise_scales']:
        stream=base.rng(parent['seed']+10000);chunks=[]
        for lo in range(0,cfg['samples'],cfg['batch_size']):
            n=min(cfg['batch_size'],cfg['samples']-lo)
            state=stream.get_state()
            z,ids=prior.sample(n,stream)
            if scale==1.:
                # Exact original sampler path, including its float rounding.
                check=base.rng(parent['seed']);check.set_state(state)
                reference,reference_ids=prior.sample(n,check)
                assert torch.equal(z,reference) and torch.equal(ids,reference_ids)
            else:z=means[ids]+scale*(z-means[ids])
            chunks.append(uint8_images(g(z)).cpu())
        images=torch.cat(chunks)
        save_image(images[:100].float()/255,out/f'samples_noise_{scale:g}.png',nrow=10)
        stats=evaluator.statistics(images)
        fid=float(fid_statistics_to_metric(stats,evaluator.real,verbose=False)['frechet_inception_distance'])
        row={'noise_scale':scale,'samples':cfg['samples'],'fid':fid,
             'feature_mean_distance_squared':float(np.square(stats['mu']-evaluator.real['mu']).sum()),
             'feature_variance_ratio':float(np.trace(stats['sigma'])/np.trace(evaluator.real['sigma']))}
        results.append(row);print('SAMPLING',json.dumps(row),flush=True)
    # Balanced repeated draws: ANOVA sum-of-squares, not a class-coverage metric.
    stream=base.rng(parent['seed']+81000)
    ids=torch.randperm(len(means),device='cuda',generator=stream)[:cfg['particles']]
    eps=torch.randn(cfg['particles'],cfg['draws_per_particle'],parent['z_dim'],device='cuda',generator=stream)
    z=means[ids,None,:]+prior.sigma*eps
    pixels=torch.cat([uint8_images(g(x)).cpu() for x in z.flatten(0,1).split(cfg['batch_size'])])
    save_image(pixels.reshape(cfg['particles'],cfg['draws_per_particle'],3,32,32)[:16,:8].flatten(0,1).float()/255,out/'within_particle.png',nrow=8)
    features=[]
    matmul,cudnn=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for x in pixels.split(cfg['batch_size']):features.append(evaluator.model(x.cuda())[0].cpu())
    finally:
        torch.backends.cuda.matmul.allow_tf32=matmul;torch.backends.cudnn.allow_tf32=cudnn
    def decompose(values):
        x=values.double().reshape(cfg['particles'],cfg['draws_per_particle'],-1)
        centers=x.mean(1,keepdim=True);mean=x.mean((0,1),keepdim=True)
        within=float((x-centers).square().sum());between=float(((centers-mean).square().sum()*cfg['draws_per_particle']))
        total=float((x-mean).square().sum());assert abs(total-within-between)/total<1e-10
        return {'within_fraction':within/total,'between_fraction':between/total,'total_variance_trace':total/(x.shape[0]*x.shape[1]-1)}
    diversity={'particles':cfg['particles'],'draws_per_particle':cfg['draws_per_particle'],
               'pixel':decompose(pixels.float()/255),'inception':decompose(torch.cat(features))}
    assert frozen==base.state_hash([g,prior]) and digest(path)==cfg['checkpoint_sha256']
    summary={'final':{'sampling':results,'diversity':diversity},'config':cfg,'parent_step':ck['step'],'parent_unchanged':True,'frozen_state_unchanged':True,
             'protocol':PROTOCOL,'sampling':results,'diversity':diversity,'total_seconds':time.perf_counter()-start,
             'note':'Read-only EMA sampler intervention, not a trained model improvement. Scale 1 reproduces original sampler; other scales retain the same particle IDs and underlying noise. Within-particle ANOVA is descriptive, not a class-coverage metric.'}
    base.write_json(out/'summary.json',summary);print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=base.read_config(a.config);assert not set(cfg)-set(DEFAULTS);train({**DEFAULTS,**cfg})
