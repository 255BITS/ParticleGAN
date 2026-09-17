#!/usr/bin/env python
"""Generate MoG reference or optimizer-pilot configs for the shared grid runner."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from experiments.train_100gaussians import DEFAULTS

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = ROOT/'configs/mog/stage1_criteria.json'


def frozen_criteria():
    if CRITERIA.exists():
        return json.loads(CRITERIA.read_text())
    baseline_path = ROOT/'results/mog/results.csv'
    with baseline_path.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['arm'] == 'C0']
    if len(rows) != 3:
        raise ValueError('Expected the three completed Stage 0 C0 references')
    values = {key: [float(r[key]) for r in rows] for key in ('hq_ratio','width_ratio','kl_balance')}
    width = min(values['width_ratio'])
    mean_width = sum(values['width_ratio'])/3
    data = dict(version='c0_observed_envelope_v1',
                baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                baseline_seeds=[int(r['seed']) for r in rows],
                definition='100 modes; HQ at least C0 worst seed; width no farther from 1 than C0 worst seed; KL no worse than C0 worst seed. Frozen before Stage 1.',
                thresholds=dict(hq_ratio_min=min(values['hq_ratio']), width_ratio_min=width,
                                width_ratio_max=2-width, kl_balance_max=max(values['kl_balance'])),
                mean_thresholds=dict(hq_ratio_min=sum(values['hq_ratio'])/3, width_ratio_min=mean_width,
                                     width_ratio_max=2-mean_width, kl_balance_max=sum(values['kl_balance'])/3))
    CRITERIA.parent.mkdir(parents=True, exist_ok=True)
    CRITERIA.write_text(json.dumps(data, indent=2)+'\n')
    return data


def save(cfg, name, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'{name}.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(path.relative_to(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['stage0','stage1_lr','stage1_beta','noise_check','longer','refine_noise'], default='stage0')
    args=parser.parse_args()
    folder=ROOT/'configs/mog'/args.stage
    if args.stage == 'stage0':
        for arm, prior in [('C0','mog'),('C1','fresh_gaussian')]:
            for seed in (1,2,3):
                name=f'{arm}_s{seed}'
                cfg={**DEFAULTS,'prior_kind':prior,'seed':seed,'sigma_rel':0.,'standardize':False,
                     'mog_metrics':True,'log_interval':100,'final_samples':200000,
                     'out_dir':f'results/mog/stage0/{name}'}
                save(cfg,name,folder)
        return
    criteria=frozen_criteria()['thresholds']
    if args.stage == 'refine_noise':
        for seed in (1,2,3):
            name=f'n400_r1over40_14k_s{seed}'
            cfg={**DEFAULTS,'epochs':14,'prior_kind':'mog','num_particles':400,
                 'sigma_rel':1/40,'standardize':True,'seed':seed,
                 'particle_lr_multiplier':10.,'particle_beta1':.5,
                 'mog_metrics':True,'mog_pass_criteria':criteria,'log_interval':100,
                 'final_samples':200000,'out_dir':f'results/mog/refine_noise/{name}'}
            save(cfg,name,folder)
        return
    if args.stage == 'longer':
        selected=json.loads((ROOT/'results/mog/noise_check_winner.json').read_text())
        for seed in (1,2,3):
            name=f'n400_r{selected["sigma_rel"]:g}_14k_s{seed}'.replace('.', 'p')
            cfg={**DEFAULTS,'epochs':14,'prior_kind':'mog','num_particles':400,
                 'sigma_rel':selected['sigma_rel'],'standardize':True,'seed':seed,
                 'particle_lr_multiplier':10.,'particle_beta1':.5,
                 'mog_metrics':True,'mog_pass_criteria':criteria,'log_interval':100,
                 'final_samples':200000,'out_dir':f'results/mog/longer/{name}'}
            save(cfg,name,folder)
        return
    if args.stage == 'noise_check':
        selected=json.loads((ROOT/'results/mog/stage1_winners.json').read_text())['400']
        for tag,r in [('r0',0.),('r1over32',1/32),('r1over16',1/16)]:
            for seed in (1,2,3):
                name=f'n400_{tag}_s{seed}'
                cfg={**DEFAULTS,'prior_kind':'mog','num_particles':400,'sigma_rel':r,
                     'standardize':True,'seed':seed,
                     'particle_lr_multiplier':selected['particle_lr_multiplier'],
                     'particle_beta1':selected['particle_beta1'],
                     'mog_metrics':True,'mog_pass_criteria':criteria,'log_interval':100,
                     'final_samples':200000,'out_dir':f'results/mog/noise_check/{name}'}
                save(cfg,name,folder)
        return
    if args.stage == 'stage1_lr':
        for n in (100,400):
            for mult in (.1,.3,1.,3.,10.):
                for seed in (1,2,3):
                    name=f'lr_n{n}_m{str(mult).replace(".","p")}_s{seed}'
                    cfg={**DEFAULTS,'prior_kind':'mog','num_particles':n,'sigma_rel':.125,
                         'standardize':True,'seed':seed,'particle_lr_multiplier':mult,'particle_beta1':0.,
                         'mog_metrics':True,'mog_pass_criteria':criteria,'log_interval':100,
                         'final_samples':200000,'out_dir':f'results/mog/stage1/{name}'}
                    save(cfg,name,folder)
    else:
        winners=json.loads((ROOT/'results/mog/stage1_lr_winners.json').read_text())
        for n, selected in winners.items():
            for seed in (1,2,3):
                tag=str(float(selected['particle_lr_multiplier'])).replace('.','p')
                original=f'lr_n{n}_m{tag}_s{seed}'
                cfg=yaml.safe_load((ROOT/f'configs/mog/stage1_lr/{original}.yaml').read_text())
                # This exact config/output is reused by run_grid, not retrained.
                save(cfg, f'beta0_n{n}_s{seed}', folder)
                name=f'beta05_n{n}_m{tag}_s{seed}'
                save({**cfg,'particle_beta1':.5,'out_dir':f'results/mog/stage1/{name}'},name,folder)


if __name__=='__main__':
    main()
