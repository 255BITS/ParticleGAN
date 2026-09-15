#!/usr/bin/env python
"""Summarize certified CIFAR speed scouts, excluding startup from throughput."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    rows = []
    for root in args.roots:
        for path in sorted(Path(root).glob('*/summary.json')):
            cert_path = path.parent / 'run_grid_complete.json'
            if not cert_path.exists():
                raise RuntimeError(f'missing certificate: {path}')
            cert = json.loads(cert_path.read_text())
            if cert['summary_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise RuntimeError(f'summary hash mismatch: {path}')
            s = json.loads(path.read_text()); cfg = s['config']
            if cert['config'] != cfg or cert['provenance'] != s['provenance']:
                raise RuntimeError(f'config/provenance mismatch: {path}')
            logs = [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text().splitlines()]
            # Match discarded warmup exposure even for larger-batch scouts.
            first = next(m for m in logs if m['step'] * cfg['batch_size'] >= 12800)
            last = logs[-1]
            steady = ((last['step'] - first['step']) * cfg['batch_size'] /
                      (last['train_seconds'] - first['train_seconds']))
            rows.append({'run':path.parent.name, 'root':str(path.parent), 'gpu':s['environment']['visible_devices'],
                         'architecture':cfg['architecture'], 'steps':cfg['steps'], 'batch':cfg['batch_size'],
                         'samples_per_optimizer':cfg['steps']*cfg['batch_size'],
                         'steady_samples_s':steady, 'all_train_samples_s':s['samples_per_second'],
                         'train_s':s['train_seconds'], 'total_s':s['total_seconds'],
                         'peak_train_gb':last['peak_memory_gb'], 'fid':s['final']['fid'], 'fid_samples':s['final']['samples'],
                         'cache':cfg['cache_condition'], 'channels_last':cfg['channels_last'],
                         'fused_adam':cfg['fused_adam'], 'reg_method':cfg['reg_method'], 'reg_every':cfg['reg_every']})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'leaderboard.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['| Run | GPU | Steps × batch | Steady samples/s | Train min | Peak train GiB | FID | FID samples |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['samples_per_optimizer'],r['architecture'],-r['steady_samples_s'])):
        lines.append(f"| {r['run']} | {r['gpu']} | {r['steps']} × {r['batch']} | {r['steady_samples_s']:.1f} | {r['train_s']/60:.2f} | {r['peak_train_gb']:.2f} | {r['fid']:.3f} | {r['fid_samples']} |")
    (out/'TABLE.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
