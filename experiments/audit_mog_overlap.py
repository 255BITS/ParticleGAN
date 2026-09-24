#!/usr/bin/env python
"""Post-training latent mixture ambiguity, grouped by observed output majority mode.

This is a diagnostic of the learned latent distributions, not a training loss.
All Gaussians have equal weights and the checkpoint's shared sigma. The Bayes
ambiguity is E_z[1 - max_label P(label | z)], estimated over fresh mixture draws.
Labels are either component IDs or each component's HQ-sample majority mode.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from particlegan.particle_prior import MoGParticlePrior
from experiments.analyze_mog_noise import load_rows
from experiments.analyze_mog_stage1 import save_csv
OUT=ROOT/'results/mog'


@torch.no_grad()
def audit(row,n=20000):
    folder=ROOT/row['out_dir']
    ckpt=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    cfg=ckpt['config']
    prior=MoGParticlePrior(cfg['num_particles'],cfg['z_dim'],sigma=0,standardize=cfg['standardize'])
    prior.load_state_dict(ckpt['prior'])
    sigma=float(prior.sigma)
    if sigma<=0:
        raise ValueError('Gaussian-overlap diagnostic requires positive sigma')
    component=json.loads((folder/'components.json').read_text())
    # Components without HQ observations have an explicit unknown label.
    majority=torch.tensor([100 if mode is None else mode for mode in component['majority_mode']])
    rng=torch.Generator().manual_seed(cfg['seed']+999)
    z,idx=prior.sample(n,generator=rng)
    means=prior.means().double()
    total_component=total_mode=total_error=total_known=0.
    for start in range(0,n,2048):
        batch=z[start:start+2048].double()
        squared=torch.cdist(batch,means).square()
        posterior=torch.softmax(-squared/(2*sigma*sigma),dim=1)
        mode_posterior=torch.zeros(len(batch),101,dtype=torch.float64)
        mode_posterior.scatter_add_(1,majority.expand(len(batch),-1),posterior)
        source=majority[idx[start:start+len(batch)]]
        known=source!=100
        total_component+=float((1-posterior.max(1).values).clamp_min(0).sum())
        total_mode+=float((1-mode_posterior.max(1).values).clamp_min(0).sum())
        total_error+=float(((mode_posterior.argmax(1)!=source)&known).sum())
        total_known+=float(known.sum())
    return dict(run=row['run'],steps=row['steps'],sigma_rel=cfg['sigma_rel'],seed=cfg['seed'],
                diagnostic_samples=n,diagnostic_rng='CPU seed+999; independent of GPU quality evaluation',
                component_bayes_ambiguity=total_component/n,
                output_mode_bayes_ambiguity=total_mode/n,
                output_mode_classification_error=total_error/total_known if total_known else None,
                known_source_fraction=total_known/n,components_without_hq=int((majority==100).sum()),
                generated_bridge=row['bridge'],generated_hq=row['hq'],r_eff=row['r_eff'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=20000)
    args=parser.parse_args()
    if args.samples<1:parser.error('samples must be positive')
    torch.set_num_threads(1)
    rows,_=load_rows()
    audits=[audit(r,args.samples) for r in rows if r['sigma_rel']>0]
    save_csv(OUT/'noise_check_overlap.csv',audits)
    print('r, steps, mean component ambiguity, mean output-mode ambiguity, mean generated bridge')
    for key in sorted({(r['sigma_rel'],r['steps']) for r in audits}):
        group=[r for r in audits if (r['sigma_rel'],r['steps'])==key]
        print(key,*[sum(r[k] for r in group)/len(group) for k in ('component_bayes_ambiguity','output_mode_bayes_ambiguity','generated_bridge')])


if __name__=='__main__':main()
