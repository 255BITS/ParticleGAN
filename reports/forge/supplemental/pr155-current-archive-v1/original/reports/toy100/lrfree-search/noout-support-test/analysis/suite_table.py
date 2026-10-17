"""Markdown table of the 13-gate rows from the shared ledger for a few candidates. usage: python suite_table.py"""
import json
rows = [json.loads(l) for l in open('/ml2/hypergan/lrfree-20260926/ledger.jsonl')]
cands = [('seq-C (base)', ['seqC-control-noout']), ('E4 (codex run)', ['e4-noout-fullsuite']), ('E10', ['E10-noout-13gates']), ('E11', ['E11-noout-13gates']),
         ('E12nr', ['E12nr-gates', 'E12nr-ring']), ('E13s', ['E13s-gates']), ('E14s', ['E14s-gates', 'E14s-va']), ('E17b', ['E17b-gates']), ('E17a', ['E17a-gates']), ('E19a', ['E19a-gates']), ('E20a', ['E20a-gates'])]   # a column may merge several candidate names (same config, split submissions)
tasks = ['mode_hold', 'img_bars4', 'img_blobs4', 'img_intensity2', 'img_stripes2', 'vector_anisotropic', 'vector_overlap', 'vector_spiral', 'vector_two_broad', 'vector_unequal_mass', 'vector_unequal_width', 'ring_shift', 'stationary']
by = {}
for r in rows:
    for name, cs in cands:
        if r['cand'] in cs: by.setdefault(name, {})[r['task']] = r
print('| task | ' + ' | '.join(n for n, _ in cands) + ' |'); print('|---|' + '---|' * len(cands))
tot = {n: [0, 0] for n, _ in cands}
for t in tasks:
    cells = []
    for c, _ in cands:
        r = by.get(c, {}).get(t)
        if r is None: cells.append('-'); continue
        cells.append(('PASS ' if r['status'] == 'PASS' else ('ERROR ' if r['status'] == 'ERROR' else 'FAIL ')) + f"{r.get('passing_checks')}/{r.get('observations')}")
        tot[c][1] += 1; tot[c][0] += r['status'] == 'PASS'
    print(f'| {t} | ' + ' | '.join(cells) + ' |')
print('| **passed** | ' + ' | '.join(f'{tot[n][0]}/{tot[n][1]}' for n, _ in cands) + ' |')
