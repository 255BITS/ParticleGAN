#!/usr/bin/env python
"""Export a reviewable summary of completed CIFAR baseline runs."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.run_grid import has_valid_summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',nargs='+',default=['results/cifar_ddgan/baseline'])
    p.add_argument('--out',default='reports/cifar-ddgan/baseline')
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted(p for root in args.root for p in Path(root).glob('*/summary.json')):
        s=json.loads(path.read_text());run=path.parent
        if not has_valid_summary(str(run),s['config'],s['provenance']):
            raise ValueError(f'Uncertified run: {run}')
        rows.append((run.name,s))
        for name in ('config.yaml','summary.json','fid_protocol.json','metrics.jsonl','samples.png','environment.json','provenance.json','run_grid_complete.json'):
            dst=out/run.name;dst.mkdir(exist_ok=True)
            shutil.copy2(run/name,dst/name)
    if not rows:
        raise ValueError('No completed runs')
    sources=[s['provenance'] for _,s in rows]
    if any(s!=sources[0] for s in sources):
        raise ValueError('Source provenance differs between runs')
    rows.sort(key=lambda item: item[1]['final']['fid'])
    text=['# CIFAR-10 particle DDGAN comparison','',
          '| Run | Updates | Final samples | Final FID | Training min | Total min | G params | D trainable | D total |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in rows:
        text.append(f"| {name} | {s['config']['steps']:,} | {s['final']['samples']:,} | {s['final']['fid']:.3f} | {s['train_seconds']/60:.2f} | {s['total_seconds']/60:.2f} | {s['parameters']['G']:,} | {s.get('trainable_parameters',s['parameters'])['D']:,} | {s['parameters']['D']:,} |")
    seeds=', '.join(str(v) for v in sorted({s['config']['seed'] for _,s in rows}))
    text+=['',f'Seeds: {seeds}. See each full config for the controlled differences.',
           'Progress and final FIDs with different sample counts have different sample-count bias.',
           'Labels are used in training. Global FID does not verify class fidelity.',
           'See ../README.md for architecture, full metric protocol and reproduction commands.','']
    (out/'TABLE.md').write_text('\n'.join(text))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for name,s in rows:
        metrics=[json.loads(l) for l in (out/name/'metrics.jsonl').read_text().splitlines()]
        m=[v for v in metrics if 'fid' in v]
        label=f"{name} ({s['config']['eval_samples']:,} samples)"
        ax[0].plot([v['step'] for v in m],[v['fid'] for v in m],'o-',label=label)
        ax[1].plot([v['train_seconds']/60 for v in m],[v['fid'] for v in m],'o-',label=label)
    for a in ax:
        a.set_ylabel('Diagnostic FID (50,000 real)');a.legend();a.grid(alpha=.2)
    ax[0].set_xlabel('Training updates');ax[1].set_xlabel('Training minutes (evaluation excluded)')
    fig.tight_layout();fig.savefig(out/'fid_curves.png',dpi=140);plt.close(fig)
    print((out/'TABLE.md').read_text())


if __name__=='__main__':
    main()
