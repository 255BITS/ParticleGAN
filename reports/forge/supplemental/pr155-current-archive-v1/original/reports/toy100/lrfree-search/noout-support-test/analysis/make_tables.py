"""Markdown cells for the RESULTS tables straight from the run directories (no transcription): 'P|F prec ctr (hold ctr)'.
usage: python make_tables.py LABEL [LABEL ...]   where LABEL is a run-label prefix such as E19a, E19lr075 (reads runs/<LABEL>-{grid100,rotated100,staggered100})"""
import sys, json, math
def cell(run):
    try:
        v = json.load(open(f'{run}/native-noisy/verdict.json'))['accuracy']; res = json.load(open(f'{run}/result.json'))
    except Exception:
        return 'n/a'
    m = [t['metrics'] for t in v['terminal_checks'][-5:]]
    prec = min(x['precision'] for x in m); ctr = max(x['center_rms_sigma'] for x in m); h = v['holdout_metrics']
    eig = None
    try:
        L = [json.loads(l) for l in open(f'{run}/metrics.jsonl')]
        eig = max(x['max_cov_eig_ratio'] for x in L[-5:])
    except Exception:
        pass
    tag = 'P' if res['status'] == 'PASS' else 'F'
    return f"**{tag}** prec {prec:.4f} ctr {ctr:.3f} (hold {h['precision']:.4f} / {h['center_rms_sigma']:.3f}) eig {eig:.2f}" if eig else f"**{tag}** prec {prec:.4f} ctr {ctr:.3f}"
for lab in sys.argv[1:]:
    cells = [cell(f'runs/{lab}-{t}') for t in ('grid100', 'rotated100', 'staggered100')]
    npass = sum(c.startswith('**P') for c in cells)
    print(f"| {lab} | " + ' | '.join(cells) + f" | {npass}/3 |")
