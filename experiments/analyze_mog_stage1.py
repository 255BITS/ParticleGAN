#!/usr/bin/env python
"""Summarize the MoG optimizer pilot and select settings without launching runs."""
import argparse
import csv
import json
from pathlib import Path
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.mog_metrics import pass_metrics

OUT = ROOT/'results/mog'


def load_rows():
    criteria = json.loads((ROOT/'configs/mog/stage1_criteria.json').read_text())
    rows = []
    for folder in sorted((OUT/'stage1').glob('*')):
        if not (folder/'run_grid_complete.json').exists():
            continue
        s = json.loads((folder/'summary.json').read_text())
        cfg, metrics = s['config'], s['final']
        score = pass_metrics(metrics, criteria['thresholds'])
        assert metrics['passed'] == score['passed'], folder
        assert cfg['mog_pass_criteria'] == criteria['thresholds'], folder
        row = {'run':folder.name, 'phase':'lr' if folder.name.startswith('lr_') else 'beta',
               **cfg, **metrics, **score, 'git_sha':s['git_sha'], 'total_seconds':s['total_seconds']}
        row['passed_c0_mean'] = pass_metrics(metrics, criteria['mean_thresholds'])['passed']
        for key, passed in {
            'coverage':metrics['modes']==100,
            'hq':metrics['hq_ratio']>=criteria['thresholds']['hq_ratio_min'],
            'width':criteria['thresholds']['width_ratio_min']<=metrics['width_ratio']<=criteria['thresholds']['width_ratio_max'],
            'balance':metrics['kl_balance'] is not None and metrics['kl_balance']<=criteria['thresholds']['kl_balance_max'],
        }.items():
            row[f'pass_{key}'] = bool(passed)
        rows.append(row)
    return rows, criteria


def groups(rows):
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r['num_particles'],r['particle_lr_multiplier'],r['particle_beta1'])].append(r)
    cells=[]
    keys=['modes','hq','hq_ratio','width_ratio','kl_balance','r_eff','raw_std_live_ratio',
          'purity_mean','purity_below_09','alloc_empty','d_gap','bridge','bridge_heuristic']
    for (n,mult,beta),runs in grouped.items():
        cell=dict(num_particles=n,particle_lr_multiplier=mult,particle_beta1=beta,n_runs=len(runs),
                  passes=sum(r['passed'] for r in runs), strict_passes=sum(r['passed_strict'] for r in runs),
                  mean_passes=sum(r['passed_c0_mean'] for r in runs),
                  r_eff_flags=sum(r['r_eff_drift_flag'] for r in runs),
                  raw_std_flags=sum(r['raw_std_drift_flag'] for r in runs))
        for key in keys:
            vals=[r[key] for r in runs if r[key] is not None]
            cell[key]=float(np.mean(vals)) if vals else None
            cell[key+'_std']=float(np.std(vals)) if vals else None
        cells.append(cell)
    return cells


def rank(cell):
    return (-cell['passes']/cell['n_runs'], -cell['hq'], abs(cell['width_ratio']-1), cell['particle_lr_multiplier'], cell['particle_beta1'])


def table(cells):
    out=['| N | LR ×0.006 | β1 | Pass | Strict | Modes | HQ/real | Width/real | KL | r_eff | Purity | Empty allocations |',
         '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in sorted(cells,key=lambda c:(c['num_particles'],rank(c))):
        out.append(f"| {c['num_particles']} | {c['particle_lr_multiplier']:g} | {c['particle_beta1']:g} | {c['passes']}/{c['n_runs']} | {c['strict_passes']}/{c['n_runs']} | {c['modes']:.1f} | {c['hq_ratio']:.5f} | {c['width_ratio']:.3f} | {c['kl_balance']:.4f} | {c['r_eff']:.4f} | {c['purity_mean']:.3f} | {c['alloc_empty']:.1f} |")
    return '\n'.join(out)


def save_csv(path, rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def plot(rows, criteria):
    with (OUT/'results.csv').open() as stream:
        references=list(csv.DictReader(stream))
    metric_keys=['hq_ratio','width_ratio','kl_balance','passed','purity_mean','alloc_empty','r_eff','raw_std_live_ratio','d_gap']
    colors={100:'#347bb0',400:'#43a07c'}
    fig,axes=plt.subplots(3,3,figsize=(15,11))
    for ax,key in zip(axes.flat,metric_keys):
        for n in (100,400):
            for beta,style in ((0.,'-'),(.5,'--')):
                subset=[r for r in rows if r['num_particles']==n and r['particle_beta1']==beta]
                multipliers=sorted(set(r['particle_lr_multiplier'] for r in subset))
                means=[];low=[];high=[]
                for m in multipliers:
                    vals=[r[key] for r in subset if r['particle_lr_multiplier']==m and r[key] is not None]
                    means.append(np.mean(vals));low.append(min(vals));high.append(max(vals))
                    ax.scatter([m]*len(vals),vals,color=colors[n],alpha=.35,s=15,marker='s' if beta else 'o')
                if multipliers:
                    ax.plot(multipliers,means,style,marker='s' if beta else 'o',color=colors[n],label=f'N={n}, β1={beta:g}')
                    ax.fill_between(multipliers,low,high,color=colors[n],alpha=.08)
        for arm,color in [('C0','#555555'),('C1','#cc6255')]:
            ref=[r for r in references if r['arm']==arm]
            if key=='passed':
                vals=[pass_metrics({k:float(r[k]) if k!='modes' else int(r[k]) for k in ('modes','hq_ratio','width_ratio','kl_balance')},criteria['thresholds'])['passed'] for r in ref]
            else:
                vals=[float(r[key]) for r in ref if r.get(key)]
            if vals:
                ax.axhline(np.mean(vals),ls=':',color=color,lw=1,label=arm)
        if key=='hq_ratio':
            ax.axhline(criteria['thresholds']['hq_ratio_min'],color='black',lw=.7,ls='--',label='pass floor')
        if key=='width_ratio':
            ax.axhspan(criteria['thresholds']['width_ratio_min'],criteria['thresholds']['width_ratio_max'],color='gray',alpha=.12,label='pass band')
        if key=='raw_std_live_ratio':
            ax.axhline(2,color='black',ls='--',lw=.7,label='drift threshold')
        if key=='r_eff':
            ax.axhline(.125,color='black',ls='--',lw=.7,label='nominal r')
        if key=='kl_balance':
            ax.axhline(criteria['thresholds']['kl_balance_max'],color='black',ls='--',lw=.7,label='pass ceiling')
            null=json.loads((OUT/'allocation_null.json').read_text())
            for r in null:
                if r['n'] in colors:
                    ax.axhline(r['kl_balance'],ls='-.',color=colors[r['n']],lw=.7,label=f"allocation null N={r['n']}")
        ax.set(xscale='log',xlabel='Particle LR / shipped particle LR',title=key)
        ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle('Stage 1: points are seeds, lines are means, bands are observed ranges')
    fig.tight_layout();fig.savefig(OUT/'stage1_optimizer.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for n in (100,400):
        subset=[r for r in rows if r['num_particles']==n]
        ax.scatter([r['width_ratio'] for r in subset],[r['hq_ratio'] for r in subset],label=f'N={n}',color=colors[n],alpha=.7)
    for arm,color in [('C0','#555555'),('C1','#cc6255')]:
        subset=[r for r in references if r['arm']==arm]
        ax.scatter([float(r['width_ratio']) for r in subset],[float(r['hq_ratio']) for r in subset],marker='x',color=color,label=arm)
    t=criteria['thresholds']
    ax.axhline(t['hq_ratio_min'],ls='--',color='gray')
    ax.axvline(t['width_ratio_min'],ls='--',color='gray');ax.axvline(t['width_ratio_max'],ls='--',color='gray')
    ax.set(xlabel='width / real width',ylabel='HQ / real HQ',title='Stage 1: width versus HQ (PASS also requires coverage and KL)')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'stage1_width_hq.png',dpi=160);plt.close(fig)



def audit_component_centers(rows, output_path=None):
    """Explicit means-only diagnostic; primary metrics always keep noise on."""
    import torch
    from scipy.spatial.distance import cdist
    from particlegan.particle_prior import MoGParticlePrior
    from lib.toy_models import SimpleMLPGenerator
    torch.set_num_threads(1)
    audits=[]
    grid=torch.cartesian_prod(torch.arange(10)-4.5,torch.arange(10)-4.5)
    with torch.no_grad():
        for row in rows:
            ckpt=torch.load(ROOT/row['out_dir']/'final.pt',map_location='cpu',weights_only=False)
            cfg=ckpt['config']
            prior=MoGParticlePrior(cfg['num_particles'],cfg['z_dim'],sigma_rel=cfg['sigma_rel'],standardize=cfg['standardize'])
            prior.load_state_dict(ckpt['prior'])
            g=SimpleMLPGenerator(z_dim=cfg['z_dim']).eval();g.load_state_dict(ckpt['G'])
            distance,nearest=torch.cdist(g(prior.means()),grid).min(1)
            counts=torch.bincount(nearest[distance<=.09],minlength=100)
            # Audit whether d_med is dominated by neighbors assigned to the
            # same output mode. Keep prescribed r_eff unchanged in results.
            component=json.loads((ROOT/row['out_dir']/'components.json').read_text())
            majority=np.array([-1 if v is None else v for v in component['majority_mode']])
            points=prior.means().numpy().astype(np.float64)
            distances=cdist(points,points);np.fill_diagonal(distances,np.inf)
            nn=distances.argmin(1)
            valid=(majority>=0)&(majority[nn]>=0)
            same=float(np.mean(majority[valid]==majority[nn[valid]])) if valid.any() else None
            cross=(majority[:,None]!=majority[None,:])&(majority[:,None]>=0)&(majority[None,:]>=0)
            nearest_cross=np.where(cross,distances,np.inf).min(1)
            finite=nearest_cross[np.isfinite(nearest_cross)]
            cross_d=float(np.median(finite)) if len(finite) else None
            audits.append(dict(run=row['run'],num_particles=cfg['num_particles'],
                               particle_lr_multiplier=cfg['particle_lr_multiplier'],particle_beta1=cfg['particle_beta1'],
                               seed=cfg['seed'],component_center_hq=float((distance<=.09).double().mean()),
                               modes_with_hq_component_center=int((counts>0).sum()),
                               component_center_median_distance=float(distance.median()),
                               noisy_sample_hq=row['hq'], nearest_neighbor_same_majority_fraction=same,
                               nearest_different_majority_d_med=cross_d,
                               sigma_over_different_majority_d=float(prior.sigma)/cross_d if cross_d else None))
    save_csv(output_path or OUT/'stage1_component_centers.csv',audits)
    return audits

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['progress','lr','final'],default='progress')
    args=parser.parse_args()
    rows,criteria=load_rows()
    if not rows:
        print('No completed runs yet');return
    cells=groups(rows)
    save_csv(OUT/'stage1_results.csv',rows);save_csv(OUT/'stage1_leaderboard.csv',cells)
    print(table(cells))
    if args.phase=='progress':
        return
    lr_cells=[c for c in cells if c['particle_beta1']==0]
    if len(lr_cells)!=10 or any(c['n_runs']!=3 for c in lr_cells):
        raise RuntimeError('Need all 30 LR runs before selecting winners')
    lr_winners={str(n):min([c for c in lr_cells if c['num_particles']==n],key=rank) for n in (100,400)}
    (OUT/'stage1_lr_winners.json').write_text(json.dumps(lr_winners,indent=2)+'\n')
    if args.phase=='lr':
        return
    if len(rows)!=36 or any(c['n_runs']!=3 for c in cells):
        raise RuntimeError('Need 36 unique completed runs for the full pilot')
    winners={str(n):min([c for c in cells if c['num_particles']==n and c['particle_lr_multiplier']==lr_winners[str(n)]['particle_lr_multiplier']],key=rank) for n in (100,400)}
    (OUT/'stage1_winners.json').write_text(json.dumps(winners,indent=2)+'\n')
    plot(rows,criteria)
    center_audit=audit_component_centers(rows)
    t=criteria['thresholds']
    report=['# Fixed-sigma MoG: Stage 1 optimizer pilot','',
            'Stage 1 only: N=100/400, nominal r=1/8, 7k updates, three seeds per cell. Standardization on; fixed σ calibrated at initialization. LR multipliers are relative to the shipped particle LR 0.006. All other training settings retain the shipped recipe.', '',
            '## Revised pass criterion','',
            f"Frozen before the first pilot result: 100 modes; HQ/real ≥ {t['hq_ratio_min']:.8f}; width/real in [{t['width_ratio_min']:.8f}, {t['width_ratio_max']:.8f}]; KL ≤ {t['kl_balance_max']:.8f}.",
            'This accepts the observed three-seed C0 envelope (C0 passes 3/3), not statistical equivalence. It allows width to move toward real data without rewarding excessive broadening. The original strict pass and comparison with C0 means are also recorded. HQ remains high because the owner requested original-or-better quality; the width and KL requirements are relaxed.', '',
            '## Leaderboard','',table(cells),'',
            'Ranking: pass rate, then mean HQ, then distance of mean width/real from one. Cells use three runs; no best-seed selection. Seed standard deviations are in stage1_leaderboard.csv and every run is in stage1_results.csv.', '',
            '## Selected settings','']
    for n,c in winners.items():
        report.append(f"- N={n}: LR multiplier **{c['particle_lr_multiplier']:g}** (initial LR {c['particle_lr_multiplier']*.006:g}), particle β1 **{c['particle_beta1']:g}**, **{c['passes']}/3 baseline-relative passes**, {c['strict_passes']}/3 strict passes. HQ/real {c['hq_ratio']:.5f}, width/real {c['width_ratio']:.3f}, KL {c['kl_balance']:.5f}.")
    report+=['','## Interpretation and diagnostics','']
    for n in (100,400):
        subset=[r for r in rows if r['num_particles']==n]
        failures={k:sum(not r['pass_'+k] for r in subset) for k in ('coverage','hq','width','balance')}
        report.append(f"- N={n}: failure counts across {len(subset)} optimizer runs: {failures}. These overlap; they are not independent categories.")
        c=winners[str(n)]
        report.append(f"  Selected cell: purity {c['purity_mean']:.4f}, components below 0.9 purity {c['purity_below_09']:.1f}, empty majority allocations {c['alloc_empty']:.1f}, r_eff {c['r_eff']:.4f}; r_eff drift flags {c['r_eff_flags']}/3, raw-scale drift flags {c['raw_std_flags']}/3.")
    report+=['','The deterministic component-center audit is diagnostic only; it does not replace any noise-on quality metric. It evaluates G(means()) on CPU, using the saved EMA state.','']
    for n,c in winners.items():
        subset=[a for a in center_audit if a['num_particles']==int(n) and a['particle_lr_multiplier']==c['particle_lr_multiplier'] and a['particle_beta1']==c['particle_beta1']]
        report.append(f"- N={n}, selected cell: fraction of component centers mapping within a data mode's 3σ radius {np.mean([a['component_center_hq'] for a in subset]):.4f}; fraction of noisy samples HQ {np.mean([a['noisy_sample_hq'] for a in subset]):.4f}; nearest neighbor has the same majority mode for {np.mean([a['nearest_neighbor_same_majority_fraction'] for a in subset]):.3%} of evaluable components.")
    report+=['','## Recommendation','']
    if not any(r['passed'] for r in rows):
        report.append('No original-or-better result at nominal r=1/8 and 7k updates. The relaxed threshold is not the main issue: noisy outputs remain substantially too broad and HQ is far below C0. Do not launch the full Stage 2 grid on the assumption that this pilot succeeded.')
        report.append('Recommended next experiment, pending owner direction: prioritize the N=400 standardized atoms control (r=0), r=1/32, and r=1/16 at the selected optimizer setting. The r=1/32 cell would be an explicit addition to the original Stage 2 design. This isolates the small-table limitation from noise-induced spread before spending the full 120-run budget. No such runs were launched.')
        report.append('Both LR winners are at the tested upper boundary, and their raw-scale drift needs to remain visible. They are the best tested optimizer settings, not established optima. Same-mode component clumping also makes the prescribed median-NN r_eff hard to interpret as separation between different output modes; retain its drift flag and the additional diagnostic, without silently redefining the independent variable.')
    else:
        report.append('At least one baseline-relative pass exists; report the per-cell pass rate and carry the selected per-N settings to a separately authorized Stage 2.')
    report+=['','## Prediction status','',
             '| Prediction | Status after Stage 1 | Evidence / missing comparison |',
             '|---|---|---|',
             '| 1: N=100 atoms | Inconclusive | No r=0 small-N control was trained. |',
             f"| 2: N=400 plateau | Inconclusive | {sum(r['passed'] for r in rows if r['num_particles']==400)}/18 pilot runs pass at nominal r=1/8; other r values are untested and r_eff drifts. |",
             '| 3: bridge heuristic | Inconclusive as a wall-sharpness test | Mode widths are too broad to attribute non-HQ mass solely to walls. Per-run bridge and heuristic values are retained in the CSV. |',
             f"| 4: balance by tearing | Inconclusive overall; little HQ tearing in selected N=400 cell | Mean HQ-conditioned purity {winners['400']['purity_mean']:.5f}, KL {winners['400']['kl_balance']:.5f}; allocation-null KL 0.13208. |",
             f"| 5: N-dependent allocation | Inconclusive for pass rates | Selected empty-allocation means: N=100 {winners['100']['alloc_empty']:.2f}, N=400 {winners['400']['alloc_empty']:.2f}; no r sweep. |",
             '| 6: lower plateau edge | Inconclusive | No noise-radius sweep. |',
             '| 7: r=2 versus C1 | Inconclusive | No nominal r=2 run. A large clumping-driven r_eff is not that control. |']
    report+=['','## Execution and changes from the original design','',
             '- The owner explicitly revised the primary criterion to original-or-better. configs/mog/stage1_criteria.json freezes the baseline CSV hash, seed list, exact thresholds, and C0-mean thresholds.',
             '- 30 LR runs plus six new β1=0.5 runs; the six selected β1=0 comparisons reuse identical completed LR runs. That is 42 planned comparisons but 36 unique training runs.',
             '- The β1 round follows the winning LR independently for each N. No other hyperparameter was retuned.',
             '- All final metrics use 200k EMA samples with noise enabled and equal-size reals. Traces use 20k samples every 100 completed updates. Both pass definitions are saved without overwriting Stage 0 history.',
             '- An additional final-checkpoint diagnostic measures nearest neighbors with different majority output modes, to interpret clumping. It never replaces the prescribed r_eff or changes any selection criterion.',
             '- Component purity excludes components with no HQ samples, which are counted separately. Width uses the unchanged median-radius estimator. r_eff and raw scale are reported, including drift flags.',
             '- Optimizer plots use LR as the swept x-axis; r_eff is a diagnostic panel. The planned log-r_eff curves require Stage 2.',
             '- All seven preregistered predictions remain conditional on their specified comparisons; Stage 1 only probes N=100/400 at nominal r=1/8. It does not test the full plateau, N=100 atoms, r=2, or the no-standardization control.', '',
             '## Timing','',f"Measured mean {np.mean([r['total_seconds'] for r in rows]):.1f} seconds/run under the recorded GPU concurrency; total accumulated run time {sum(r['total_seconds'] for r in rows)/60:.1f} minutes. Runner logs contain actual elapsed wall time.", '',
             '## Artifacts','',
             '- stage1_results.csv: one row per unique run, all config and final scalar metrics, seed and git SHA.',
             '- stage1_leaderboard.csv, stage1_lr_winners.json, stage1_winners.json.',
             '- stage1_optimizer.png, stage1_width_hq.png, stage1_component_centers.csv.',
             '- stage1/<run>/metrics.jsonl, log.txt, components.json, final.pt, final_samples.npz, source.zip and provenance.json.', '']
    (OUT/'STAGE1.md').write_text('\n'.join(report))


if __name__=='__main__':
    main()
