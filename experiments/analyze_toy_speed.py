#!/usr/bin/env python
"""Certified toy throughput, mode coverage and within-mode shape comparisons."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    rows=[]
    for root in args.roots:
        for cert_path in sorted(Path(root).glob('*/run_grid_complete.json')):
            cert=json.loads(cert_path.read_text());path=cert_path.parent/'summary.json';s=json.loads(path.read_text());c=s['config'];f=s['final']
            if cert['summary_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest() or cert['config']!=c or cert['provenance']!=s['provenance']:
                raise RuntimeError(f'Invalid certificate: {path}')
            steps=c.get('steps',c.get('epochs',1)*c.get('steps_per_epoch',1))
            row={'root':str(path.parent),'run':path.parent.name,'task':'one-shot' if 'epochs' in c else 'DDGAN/joint UCD',
                 'steps':steps,'samples_per_second':s['samples_per_second'],'train_seconds':s['train_seconds'],
                 'modes':f['modes'],'joint_hq':f['joint_hq'],'hq':f['hq'],'cond_acc':f['cond_acc'],
                 'mode_tv':f['mode_tv'],'conditional_sw1':f['conditional_sw1'],'core_ratio':f['per_mode_core_ratio'],
                 'cov_eig_min_ratio':f['per_mode_cov_eig_min_ratio'],'cov_eig_max_ratio':f['per_mode_cov_eig_max_ratio'],
                 'cov_audited_modes':f['per_mode_cov_audited_modes'],'fd_eps':c['reg_fd_eps']}
            rows.append(row)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'leaderboard.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['| Task | Steps | Variant | Samples/s | HQ + correct class | Modes | Mode TV ↓ | Core std ratio | Cov eigenvalue ratios |', '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['task'],r['steps'],r['run'])):
        def fmt(v):return '—' if v is None else f'{v:.3f}'
        lines.append(f"| {r['task']} | {r['steps']} | {r['run']} | {r['samples_per_second']:.0f} | {r['joint_hq']*100:.2f}% | {r['modes']} | {r['mode_tv']:.3f} | {fmt(r['core_ratio'])} | {fmt(r['cov_eig_min_ratio'])}, {fmt(r['cov_eig_max_ratio'])} |")
    (out/'TABLE.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
