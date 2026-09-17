#!/usr/bin/env python
"""Analyze the small-noise screen and its selected longer-budget follow-up."""
import argparse
import csv
import json
from pathlib import Path
import sys
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.mog_metrics import pass_metrics
from experiments.analyze_mog_stage1 import save_csv,audit_component_centers
OUT=ROOT/'results/mog'


def load_rows():
    criteria=json.loads((ROOT/'configs/mog/stage1_criteria.json').read_text())
    paths=[('noise_check',p) for p in (OUT/'noise_check').glob('*/summary.json')]
    paths += [('longer',p) for p in (OUT/'longer').glob('*/summary.json')]
    paths += [('refine_noise',p) for p in (OUT/'refine_noise').glob('*/summary.json')]
    paths += [('stage1_reference',OUT/f'stage1/beta05_n400_m10p0_s{s}/summary.json') for s in (1,2,3)]
    rows=[]
    for phase,path in sorted(paths):
        if not (path.parent/'run_grid_complete.json').exists():
            continue
        summary=json.loads(path.read_text());cfg=summary['config'];metrics=summary['final']
        assert cfg['num_particles']==400 and cfg['particle_lr_multiplier']==10 and cfg['particle_beta1']==.5
        assert cfg['mog_pass_criteria']==criteria['thresholds']
        score=pass_metrics(metrics,criteria['thresholds']);assert score['passed']==metrics['passed']
        rows.append({'run':path.parent.name,'phase':phase,**cfg,**metrics,
                     'steps':cfg['epochs']*cfg['steps_per_epoch'],
                     'passed_c0_mean':pass_metrics(metrics,criteria['mean_thresholds'])['passed'],
                     'git_sha':summary['git_sha'],'total_seconds':summary['total_seconds']})
    return rows,criteria


def grouped(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r['steps'],r['sigma_rel'])].append(r)
    cells=[]
    for (steps,sigma_rel),group in groups.items():
        cell=dict(steps=steps,sigma_rel=sigma_rel,n_runs=len(group),passes=sum(r['passed'] for r in group),
                  original_passes=sum(r['passed_strict'] for r in group),mean_passes=sum(r['passed_c0_mean'] for r in group),
                  r_eff_flags=sum(r['r_eff_drift_flag'] for r in group),raw_scale_flags=sum(r['raw_std_drift_flag'] for r in group))
        for key in ('modes','hq','hq_ratio','width_ratio','kl_balance','purity_mean','purity_below_09','alloc_empty','r_eff','raw_std_live_ratio','d_gap','bridge','bridge_heuristic'):
            vals=[r[key] for r in group if r[key] is not None]
            cell[key]=float(np.mean(vals)) if vals else None
            cell[key+'_std']=float(np.std(vals)) if vals else None
        cells.append(cell)
    return sorted(cells,key=lambda c:(c['steps'],c['sigma_rel']))


def rank(c):
    return (-c['passes']/c['n_runs'],-c['hq'],abs(c['width_ratio']-1),c['sigma_rel'])


def table(cells):
    lines=['| Steps | Nominal r | Pass | Original rule | Modes | HQ/real | Width/real | KL | r_eff |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        lines.append(f"| {c['steps']} | {c['sigma_rel']:g} | {c['passes']}/{c['n_runs']} | {c['original_passes']}/{c['n_runs']} | {c['modes']:.1f} | {c['hq_ratio']:.5f} | {c['width_ratio']:.3f} | {c['kl_balance']:.5f} | {c['r_eff']:.4f} |")
    return '\n'.join(lines)


def plot(rows,criteria):
    refs=list(csv.DictReader((OUT/'results.csv').open()))
    keys=['hq_ratio','width_ratio','kl_balance','purity_mean','passed','raw_std_live_ratio']
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    colors={.025:'#8c564b',.03125:'#347bb0',.0625:'#43a07c',.125:'#d49d29'}
    for ax,key in zip(axes.flat,keys):
        for r_nominal,color in colors.items():
            for steps,marker in [(7000,'o'),(14000,'s')]:
                group=[r for r in rows if r['sigma_rel']==r_nominal and r['steps']==steps]
                if group:
                    ax.scatter([r['r_eff'] for r in group],[r[key] for r in group],color=color,marker=marker,
                               label=f'r={r_nominal:g}, {steps//1000}k',s=35)
        for arm,color in [('C0','#555555'),('C1','#cc6255')]:
            group=[r for r in refs if r['arm']==arm]
            if key=='passed':
                vals=[pass_metrics({k:float(r[k]) if k!='modes' else int(r[k]) for k in ('modes','hq_ratio','width_ratio','kl_balance')},criteria['thresholds'])['passed'] for r in group]
            else:
                vals=[float(r[key]) for r in group if r.get(key)]
            if vals:
                ax.axhline(np.mean(vals),color=color,ls=':',lw=1,label=arm)
        atoms=[r[key] for r in rows if r['sigma_rel']==0 and r['steps']==7000 and r[key] is not None]
        if atoms:
            ax.axhline(np.mean(atoms),color='#9467bd',ls='--',label='N=400 atoms, 7k')
        if key=='width_ratio':
            ax.axhspan(criteria['thresholds']['width_ratio_min'],criteria['thresholds']['width_ratio_max'],alpha=.1,color='gray')
        if key=='kl_balance':
            null=next(r for r in json.loads((OUT/'allocation_null.json').read_text()) if r['n']==400)
            ax.axhline(null['kl_balance'],color='gray',ls='-.',label='allocation null N=400')
        ax.set(xscale='log',xlabel='Measured r_eff = σ / final median NN distance',title=key)
        ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle('N=400: each seed shown; large r_eff can reflect same-mode clumping')
    fig.tight_layout();fig.savefig(OUT/'noise_check_metrics.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for c in grouped(rows):
        group=[r for r in rows if r['sigma_rel']==c['sigma_rel'] and r['steps']==c['steps']]
        ax.scatter([r['width_ratio'] for r in group],[r['hq_ratio'] for r in group],marker='s' if c['steps']==14000 else 'o',label=f"r={c['sigma_rel']:g}, {c['steps']//1000}k")
    for arm in ('C0','C1'):
        group=[r for r in refs if r['arm']==arm]
        ax.scatter([float(r['width_ratio']) for r in group],[float(r['hq_ratio']) for r in group],marker='x',label=arm)
    t=criteria['thresholds'];ax.axhline(t['hq_ratio_min'],ls='--',color='gray')
    ax.axvline(t['width_ratio_min'],ls='--',color='gray');ax.axvline(t['width_ratio_max'],ls='--',color='gray')
    ax.set(xlabel='width / real width',ylabel='HQ / real HQ',title='Noise and training-budget follow-up')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'noise_check_width_hq.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['progress','select','final'],default='progress')
    args=parser.parse_args()
    rows,criteria=load_rows();cells=grouped(rows)
    save_csv(OUT/'noise_check_results.csv',rows);save_csv(OUT/'noise_check_leaderboard.csv',cells)
    print(table(cells))
    if args.phase=='progress':return
    if sum(r['phase']=='noise_check' for r in rows)!=9:
        raise RuntimeError('Need all nine small-noise runs before selection')
    winner=min([c for c in cells if c['sigma_rel']>0 and c['steps']==7000],key=rank)
    (OUT/'noise_check_winner.json').write_text(json.dumps(winner,indent=2)+'\n')
    if args.phase=='select':return
    longer=[c for c in cells if c['steps']==14000 and c['sigma_rel']==winner['sigma_rel']]
    if len(longer)!=1 or longer[0]['n_runs']!=3 or longer[0]['sigma_rel']!=winner['sigma_rel']:
        raise RuntimeError('Need the three selected 14k runs')
    if sum(r['phase']=='refine_noise' for r in rows)!=3:
        raise RuntimeError('Need the three targeted refinement runs')
    plot(rows,criteria)
    audits=audit_component_centers(rows,OUT/'noise_check_component_centers.csv')
    long=longer[0];atoms=next(c for c in cells if c['sigma_rel']==0)
    report=['# N=400 MoG: smaller noise and longer training','',
            'Three seeds per cell, the Stage 1 selected particle LR multiplier 10 (initial LR 0.06), particle beta1=0.5, standardized reads, fixed sigma calibrated at initialization. The C0-relative acceptance thresholds remain unchanged. All new runs use the existing trainer and grid runner.', '',
            '## Results','',table(cells),'',
            'The r=1/8, 7k row reuses the three selected Stage 1 runs. The other rows comprise nine new 7k runs, three selected 14k runs, and three targeted r=1/40 refinement runs at 14k. All final metrics use 200k noisy EMA samples with matched reals; traces use 20k samples every 100 updates. Per-seed results and population standard deviations are saved in the CSV artifacts.', '',
            'The r=0 control is not C0: it has 400 particles, standardized reads, and the selected particle optimizer. Its comparison to positive-r cells isolates the addition of noise at those settings.', '',
            '## Budget comparison','',
            f"The selected positive-noise setting was nominal r={winner['sigma_rel']:g}, selected across all three seeds by pass rate, then HQ, then width closeness to real. The atoms control was excluded from this selection because the requested longer run tests the MoG's continuous neighborhoods.",
            f"At 7k: {winner['passes']}/3 pass, HQ/real {winner['hq_ratio']:.5f}, width/real {winner['width_ratio']:.4f}, KL {winner['kl_balance']:.5f}. At 14k: {long['passes']}/3 pass, HQ/real {long['hq_ratio']:.5f}, width/real {long['width_ratio']:.4f}, KL {long['kl_balance']:.5f}.",
            'The 14k runs start from the same seeds and initialization. They are fresh training runs, not continuation from EMA checkpoints. The delayed cosine schedule scales with total budget (annealing starts at 8,400 instead of 4,200 updates), so this tests the longer-budget recipe, not extra updates at an unchanged LR schedule.', '',
            '## Targeted refinement','',
            'After inspecting the r=1/32 longer-budget result, we tested r=1/40 at 14k with the same optimizer and seeds. This is an exploratory follow-up chosen to reduce remaining spread; it was not part of the original grid. The fixed acceptance thresholds were not changed.', '',
            '## Geometry and component diagnostics','']
    for c in cells:
        group=[a for a in audits if any(r['run']==a['run'] and r['sigma_rel']==c['sigma_rel'] and r['steps']==c['steps'] for r in rows)]
        report.append(f"- r={c['sigma_rel']:g}, {c['steps']} steps: component-center HQ {np.mean([a['component_center_hq'] for a in group]):.4f}, noisy HQ {c['hq']:.4f}, HQ-conditioned purity {c['purity_mean']:.4f}, empty majority allocations {c['alloc_empty']:.2f}; r_eff {c['r_eff']:.4f}, r_eff flags {c['r_eff_flags']}/3, raw-scale flags {c['raw_scale_flags']}/3.")
    report+=['','The center-only diagnostic is explicit and supplementary; it never replaces noise-on evaluation. Large median-NN ratios can reflect neighboring components that map to the same output mode. Component-center CSV records that diagnostic separately. The r=0 deterministic all-table audit has only 400 points: its >=10 coverage and >=50 width thresholds are not suitable for judging that control; use its primary 200k-sample metrics.', '',
             '## Changes and validation','',
             '- r=1/32 was the explicitly proposed addition to the original noise grid. r=0 and r=1/16 were already planned values. Only N=400 is tested here.',
             '- The owner authorized further experiments and a longer-budget experiment. Budget doubled to 14k for the most promising positive-noise cell. A subsequent r=1/40 refinement changed only the fixed noise radius; model, optimizer, regularizer and evaluation settings remain fixed.',
             '- Pass thresholds are frozen from Stage 0, with the old design criterion retained in passed_strict. Stage 0 and Stage 1 reports are preserved as historical results.',
             '- Stage 1 references are reused rather than retrained. No seed-only search or best-seed selection.',
             '- For r=0, sigma consumes no noise RNG. Thus r=0 and r>0 training streams differ after sampling, as required by the zero-noise regression contract.', '',
             '## Artifacts','',
             '- noise_check_results.csv, noise_check_leaderboard.csv, noise_check_winner.json.',
             '- noise_check_metrics.png, noise_check_width_hq.png, noise_check_component_centers.csv.',
             '- noise_check/<run>/ and longer/<run>/: complete run configs, source archives, final checkpoints/samples, component diagnostics, JSONL traces, and logs.', '']
    (OUT/'NOISE_CHECK.md').write_text('\n'.join(report))


if __name__=='__main__':main()
