"""Churn analysis of the support-test re-draws from the dump files written by pkg-E16dbg (ISO_DUMP=dir). Evaluation side: distances use the true lattice centres.
usage: python iso_churn.py DUMP_DIR RUN_DIR TASK"""
import sys, glob, numpy as np, torch
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/analysis/forensics')
from lib import centers, assign
d, run, task = sys.argv[1], sys.argv[2], sys.argv[3]
tr = torch.load(f'{run}/final-state.pt', map_location='cpu', weights_only=False)['trainer']['models']
W = tr['G']['weight'].double().numpy(); b = tr['G']['bias'].double().numpy()
ce = centers(task)
def dist(z): return np.linalg.norm(assign(z.double().numpy() @ W.T + b, ce)[1], axis=1)      # sigma units, approx G = final G
files = sorted(glob.glob(f'{d}/iso-*.pt'))
ev = [torch.load(f) for f in files]
print(len(ev), 'acted evaluations; rows re-drawn:', sum(len(e['dead']) for e in ev), '| steps', ev[0]['step'], '..', ev[-1]['step'])
allrad = np.concatenate([dist(e['z_dead']) for e in ev]); parrad = np.concatenate([dist(e['z_parent']) for e in ev])
bins = [0, 2, 3, 4, 6, 10, 16, 1e9]
h = np.histogram(allrad, bins)[0]; print('distance of the re-drawn rows to the nearest centre (sigma):', {f'{bins[i]}-{bins[i+1]}': int(h[i]) for i in range(len(h))})
hp = np.histogram(parrad, bins)[0]; print('distance of the parents:', {f'{bins[i]}-{bins[i+1]}': int(hp[i]) for i in range(len(hp))}, 'mean', float(parrad.mean()))
cnt = {}; last = {}; gaps = []
for e in ev:
    for r in e['dead'].tolist():
        cnt[r] = cnt.get(r, 0) + 1
    for r, p in zip(e['dead'].tolist(), e['parent'].tolist()):
        pass
mult = np.bincount(list(cnt.values()))
print('unique rows re-drawn:', len(cnt), 'of 20000 | multiplicity histogram (times re-drawn: rows):', {i: int(m) for i, m in enumerate(mult) if i > 0 and m})
# time since a row was last re-drawn (as a child), when it is re-drawn again
lastmove = {}
for e in ev:
    for r in e['dead'].tolist():
        if r in lastmove: gaps.append(e['step'] - lastmove[r])
        lastmove[r] = e['step']
if gaps:
    g = np.array(gaps); print('re-draws of a row that had been re-drawn before:', len(g), '| steps since its last re-draw: median', float(np.median(g)), 'p10', float(np.percentile(g, 10)), 'p90', float(np.percentile(g, 90)))
# how far into the run: re-draws per 1000 steps
steps = np.array([e['step'] for e in ev]); n = np.array([len(e['dead']) for e in ev])
for lo in range(0, int(steps.max()) + 1, 1000):
    sel = (steps > lo) & (steps <= lo + 1000)
    if sel.any(): print(f'  steps {lo:5d}-{lo+1000:5d}: {int(sel.sum()):3d} acted evals, {int(n[sel].sum()):5d} rows re-drawn, median distance {float(np.median(np.concatenate([dist(ev[i]["z_dead"]) for i in np.where(sel)[0]]))):.1f} sigma')
