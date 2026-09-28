#!/usr/bin/env python
"""Compare a harness run's observations with archived evidence metrics (exact float equality).

usage: compare.py RUN_DIR EVIDENCE_METRICS(.jsonl|.jsonl.gz) [--rates EVIDENCE_RATES] [--steps N]
"""
import argparse
import gzip
import json
from pathlib import Path


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def numeric(d, prefix=''):
    out = {}
    for k, v in d.items():
        if k in ('seconds', 'diag', 'lr', 'pass', 'learning_rates', 'penalty', 'policy', 'losses', 'precision',
                 'game', 'trainer_stats', 'input_noise', 'output_noise', 'evaluation_output_noise', 'frozen'):
            continue
        if isinstance(v, dict):
            out.update(numeric(v, prefix + k + '.'))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            out[prefix + k] = v
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            for i, x in enumerate(v):
                out[f'{prefix}{k}[{i}]'] = x
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('evidence')
    p.add_argument('--rates', help='evidence learning-rates.jsonl(.gz) for per-step lr comparison')
    args = p.parse_args()
    mine = {r['step']: r for r in rows(Path(args.run) / 'metrics.jsonl')}
    ref = {r['step']: r for r in rows(args.evidence) if 'step' in r}
    steps = sorted(set(mine) & set(ref))
    compared = exact = 0
    worst = (0.0, None)
    first_diff = None
    for s in steps:
        a, b = numeric(mine[s]), numeric(ref[s])
        for k in set(a) & set(b):
            compared += 1
            if a[k] == b[k]:
                exact += 1
            else:
                d = abs(a[k] - b[k])
                if first_diff is None:
                    first_diff = (s, k, a[k], b[k])
                if d > worst[0]:
                    worst = (d, (s, k))
    out = dict(steps_run=len(mine), steps_evidence=len(ref), steps_compared=len(steps), values_compared=compared,
               values_exact=exact, bitwise=compared > 0 and exact == compared and len(mine) == len(ref),
               max_abs_diff=worst[0], worst=worst[1], first_diff=first_diff)
    if args.rates:
        mr = {r['step']: r for r in rows(Path(args.run) / 'rates.jsonl')}
        er = rows(args.rates)
        same = diff = 0
        for r in er:
            lr = r.get('applied_group_rates') or r.get('rates')
            if lr and isinstance(lr[0][0], dict):
                lr = [[g['lr'] for g in groups] for groups in lr]
            m = mr.get(r['step'])
            if m is None:
                continue
            mine = m['lr']
            if lr is None:  # ring worker: flat {role_i: lr} in optimizer order
                flat = [v for k, v in r.items() if k.rsplit('_', 1)[-1].isdigit() and isinstance(v, float)]
                if not flat:
                    continue
                lr, mine = flat, [x for group in m['lr'] for x in group]
            if mine == lr:
                same += 1
            else:
                diff += 1
        out['lr_steps_exact'] = same
        out['lr_steps_diff'] = diff
    print(json.dumps(out))


if __name__ == '__main__':
    main()
