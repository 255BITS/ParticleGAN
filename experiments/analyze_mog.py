#!/usr/bin/env python
"""Audit and summarize the Stage 0 MoG gate without launching further stages."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from lib.mog_metrics import allocation_null, sample_metrics
from lib.toy_metrics import mode_recall_and_hists

OUT = ROOT/'results/mog'


def regression(seed):
    old = OUT/f'stage0/pre_s{seed}'
    new = OUT/f'stage0/C0_s{seed}'
    def lines(folder):
        return [line for line in (folder/'log.txt').read_text().splitlines() if line.startswith('[epoch ')]
    a, b = lines(old), lines(new)
    original = torch.load(old/'final.pt', map_location='cpu', weights_only=False)
    changed = torch.load(new/'final.pt', map_location='cpu', weights_only=False)
    tensors = {f'G.{k}': torch.equal(v, changed['G'][k]) for k, v in original['G'].items()}
    tensors['prior.z'] = torch.equal(original['prior']['z'], changed['prior']['z'])
    return dict(seed=seed, log_entries=len(a), logs_identical=a == b and len(a) == 70,
                ema_tensors_identical=all(tensors.values()), tensors=tensors)


def plots(rows):
    colors = {'C0':'#347bb0', 'C1':'#dc6255'}
    metrics = ['hq_ratio','width_ratio','kl_balance','purity_mean','purity_below_09','alloc_empty','r_eff','raw_std_ratio','d_gap']
    fig, axes = plt.subplots(3,3,figsize=(15,11))
    for ax, key in zip(axes.flat, metrics):
        for arm in ('C0','C1'):
            group = [r for r in rows if r['arm'] == arm]
            for r in group:
                trace = [json.loads(line) for line in (OUT/f"stage0/{arm}_s{r['seed']}/metrics.jsonl").read_text().splitlines()]
                pairs = [(p['step'],p.get(key)) for p in trace if p.get(key) is not None]
                if pairs:
                    ax.plot(*zip(*pairs),color=colors[arm],alpha=.45,lw=1)
            values = [r[key] for r in group if r.get(key) is not None]
            if values:
                ax.axhline(np.mean(values),color=colors[arm],ls='--',label=f'{arm} final mean')
        if key == 'kl_balance':
            for n, style in [(100,':'),(200,'-.'),(400,'--')]:
                ax.axhline(allocation_null(n)['kl_balance'],color='gray',ls=style,label=f'alloc null N={n}')
            ax.set_ylim(bottom=0,top=.7)
        if key in ('hq_ratio','width_ratio'):
            ax.axhline(1,color='black',lw=.5)
        ax.set(title=key,xlabel='Completed training steps')
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle('Stage 0: each seed shown; dashed lines are final 200k-sample references')
    fig.tight_layout();fig.savefig(OUT/'stage0_traces.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for arm in ('C0','C1'):
        group=[r for r in rows if r['arm']==arm]
        ax.scatter([r['width_ratio'] for r in group],[r['hq_ratio'] for r in group],label=arm,color=colors[arm])
        for r in group:
            ax.annotate(f"s{r['seed']}",(r['width_ratio'],r['hq_ratio']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.axhline(.98,color='gray',ls='--');ax.axvline(.9,color='gray',ls='--');ax.axvline(1.1,color='gray',ls='--')
    ax.set(xlabel='width / real width',ylabel='HQ / real HQ',title='Stage 0 references (pass also requires coverage and KL)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'stage0_width_hq.png',dpi=160);plt.close(fig)


def main():
    torch.set_num_threads(1)
    rows=[]
    for arm in ('C0','C1'):
        for seed in (1,2,3):
            folder=OUT/f'stage0/{arm}_s{seed}'
            if not (folder/'run_grid_complete.json').exists():
                raise RuntimeError(f'Run is not certified complete: {folder}')
            s=json.loads((folder/'summary.json').read_text())
            rows.append(dict(arm=arm,**s['config'],**s['final'],git_sha=s['git_sha'],
                             train_seconds=s['train_seconds'],total_seconds=s['total_seconds']))
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    audits=[regression(seed) for seed in (1,2,3)]
    (OUT/'regression.json').write_text(json.dumps(audits,indent=2)+'\n')
    nulls=[allocation_null(n) for n in (100,200,400)]
    (OUT/'allocation_null.json').write_text(json.dumps(nulls,indent=2)+'\n')
    original_kl = []
    for seed in (1, 2, 3):
        x = torch.from_numpy(np.load(OUT/f'stage0/pre_s{seed}/final_samples.npz')['x'])
        original_kl.append(dict(seed=seed, n=len(x),
            hist_kl_all_samples=mode_recall_and_hists(x)['hist_kl'],
            kl_balance_hq_only=sample_metrics(x)[0]['kl_balance']))
    (OUT/'original_kl.json').write_text(json.dumps(original_kl, indent=2)+'\n')
    plots(rows)
    board=['| Arm | Seed | Modes | HQ | HQ/real | Width/real | KL balance | Pass | t_cover |',
           '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    ordered=sorted(rows,key=lambda r:(-r['passed'],-r['hq_ratio'],abs(r['width_ratio']-1)))
    for r in ordered:
        board.append(f"| {r['arm']} | {r['seed']} | {r['modes']} | {r['hq']:.5f} | {r['hq_ratio']:.5f} | {r['width_ratio']:.5f} | {r['kl_balance']:.6f} | {'yes' if r['passed'] else 'no'} | {r['t_cover']} |")
    text=['# Fixed-sigma MoG prior: Stage 0 gate','',
          'Only Stage 0 has run. Stages 1–3 require the owner’s go/no-go. Reference code: `af1843a`; run code SHA is recorded per row in `results.csv`, together with per-run source hashes and archives.','',
          '## Regression','']
    for a in audits:
        text.append(f"- Seed {a['seed']}: {a['log_entries']} original log entries identical: **{a['logs_identical']}**; every final EMA generator tensor and the particle table identical: **{a['ema_tensors_identical']}**.")
    text += ['', '## Leaderboard','',*board,'','Cells are ranked by pass rate, then HQ, then closeness of width to real. Individual seeds above are shown for diagnosis, not model selection.','']
    for arm in ('C0','C1'):
        group=[r for r in rows if r['arm']==arm]
        values=lambda key: np.array([r[key] for r in group],dtype=float)
        text.append(f"- **{arm}: {sum(r['passed'] for r in group)}/3 pass**; HQ/real {values('hq_ratio').mean():.5f} ± {values('hq_ratio').std():.5f}; width/real {values('width_ratio').mean():.5f} ± {values('width_ratio').std():.5f}; KL {values('kl_balance').mean():.6f} ± {values('kl_balance').std():.6f}. Spread is population std across seeds.")
    text += ['','## C0e deterministic mass balance','', '| Seed | All-table HQ | All-table KL | Min share ×100 | Max share ×100 |','|---|---:|---:|---:|---:|']
    for r in rows[:3]:
        text.append(f"| {r['seed']} | {r['deterministic_hq']:.5f} | {r['deterministic_kl_balance']:.6f} | {r['deterministic_share_min']:.4f} | {r['deterministic_share_max']:.4f} |")
    text+=['','This audit evaluates each of the 20,000 atoms once, so repeated sampling cannot explain its imbalance. The ordinary final metrics use 200,000 draws, matched to 200,000 independent real samples.',
           '', 'Historical hist_kl uses all nearest-assigned samples, including bridge samples. Applying both estimators directly to the saved pre-change 20k-sample outputs:', '',
           '| Seed | Historical definition | HQ-only definition |', '|---|---:|---:|']
    for row in original_kl:
        text.append(f"| {row['seed']} | {row['hist_kl_all_samples']:.6f} | {row['kl_balance_hq_only']:.6f} |")
    text+=['', 'These are measurements of freshly reproduced pre-change outputs; they are not recovered scores from the old five-seed FINDINGS study, whose raw summaries were not present in the checked result locations.', '', '## Allocation null (1,000 simulated draws each)', '']
    for n in nulls:
        text.append(f"- N={n['n']}: mean KL **{n['kl_balance']:.5f}**, mean empty modes **{n['empty_modes']:.3f}**.")
    text+=['','## Timing','']
    for arm in ('C0','C1'):
        durations=[r['total_seconds'] for r in rows if r['arm']==arm]
        text.append(f'- {arm}: {np.mean(durations):.1f} seconds/run including evaluation and artifacts (range {min(durations):.1f}–{max(durations):.1f}).')
    text+=['','## Predictions and next decision','',
           'All seven preregistered predictions remain **inconclusive**: no small-N or positive-r 7k-step experiment has run. Specifically: (1) N=100 atoms, (2) N=400 plateau, (3) bridge versus heuristic, (4) tearing versus migration, (5) N-dependent allocation failures, (6) low-r conditioning, and (7) r=2 versus C1 all await their specified comparisons. C0 and C1 are references, not tests of those predictions.','',
           'Recommendation: proceed to the Stage 1 optimizer pilot, subject to owner approval, while retaining the preregistered pass threshold. C0 has a reproducible mass-balance problem and compressed width; matching its HQ alone would not establish success. C1 supplies a clearly separated failure reference. No positive-noise performance claim is supported yet. At roughly 70 seconds/run and two GPUs, budget about 25 minutes for the 42-run pilot, 70 minutes for the 120-run main grid, and 3–4 minutes per control block, before re-timing a small-N run.','',
           '## Definitions and deviations','',
           '- Accepted correction: pilot multipliers are relative to shipped particle LR 0.006; G LR remains 0.0006. The exposed particle beta1 is independent of G/D beta1.',
           '- Accepted correction: mode_coverage already uses sample(); no direct-indexing fix was needed there. Fixed scatters cap the sample count at N and reuse a seeded epsilon buffer.',
           '- Accepted correction: raw VICReg is scale-dependent and its variance term is not redundant. Standardization removes the adversarial scale escape approximately, not full-loss scale dependence.',
           '- Accepted correction: bridge_heuristic is not a rigorous bound. It is recorded but does not decide PASS.',
           '- Width exactly reuses median-centered radial core sigma, averaged equally over nearest-assigned modes with at least 50 samples; no HQ truncation. Width audited-mode counts accompany it.',
           '- The wrapper’s older final coverage threshold differed from >=10 HQ samples; the new suite uses >=10 throughout. Existing training log lines are preserved for the regression comparison.',
           '- Metrics run at completed updates 100, 200, …, 7000; t_cover has 100-step resolution. Legacy console logs retain their original zero-based labels.',
           '- Reference samples are seeded independently of latent samples. Noise remains enabled at evaluation for positive-r runs. C0e separately evaluates every atom exactly once.',
           '- Prior geometry and component identity have no counterpart for real data or a fresh Gaussian prior: those comparisons are not defined. C1 component/geometry fields are null, and those C1 reference lines are omitted from plots.',
           '- Components with zero HQ samples have undefined purity and no majority allocation; they are counted separately as components_no_hq. Purity mean excludes them. components_unsampled records missing observations.',
           '- Per-run components.json contains purity_i, bridge_i, majority_mode and alloc. For C0 these use the exact all-atom audit; final aggregate component statistics use the 200k sample.',
           '- Geometry is measured on the EMA table; raw_std_live and its ratio also track the live table. Drift flags are symmetric for increases/decreases beyond the specified factor.',
           '- Final eval is batched in chunks of 20k after drawing all latent codes, bounding generator memory. The C0 checkpoint audit establishes exact training preservation.',
           '- Stage 0 plots use training step, since C0 has r_eff=0 and C1 has no particle-distance ratio. Log-r_eff plots, pass-rate curves and the Stage 2 scatter await Stage 2; no placeholder results are claimed.',
           '- Three explicitly requested seeds override the repository’s general no-seed-experiments preference for this study. C0e reuses C0; no separate training runs.',
           '- The optional rotated-grid control remains unimplemented and unrun at this gate. FINDINGS.md does not already resolve the lattice confound.','',
           '## Artifacts','',
           '- results.csv: one row per C0/C1 run, configuration, seed, SHA and all scalar final metrics.',
           '- stage0/<arm>_s<seed>/metrics.jsonl: 70 metric rows per run; log.txt is tail-friendly.',
           '- regression.json, allocation_null.json, stage0_traces.png, stage0_width_hq.png.',
           '- Each run retains source.zip, provenance.json, final.pt, final_samples.npz, and components.json.','']
    (OUT/'STAGE0.md').write_text('\n'.join(text))
    print('\n'.join(board))
    if not all(a['logs_identical'] and a['ema_tensors_identical'] for a in audits):
        raise RuntimeError('C0 regression gate failed; see regression.json')


if __name__=='__main__':
    main()
