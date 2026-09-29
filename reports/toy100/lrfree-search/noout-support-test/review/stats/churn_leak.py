"""Q6: is the churn (the same rows re-drawn many times) a leak of the bulk or a false-positive problem? Uses the dumps of the logging replicate (iso-*.pt: step, dead rows,
parents, their z). Distances to the nearest lattice centre in sigma units use the FINAL G (G is linear in the native tasks; earlier evaluations are approximate; steps >= 4000
are the reliable window). Evaluation-side analysis only.
Per re-draw (row r at step t1, parent p): parent distance class -> was r flagged again later (dumps hold only ACTED evaluations, so re-flags are a lower bound), how soon, and where."""
import sys, glob, numpy as np, torch
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/analysis/forensics')
from lib import centers, assign
D = '/tmp/claude-1000/-ml2-hypergan/ab03afcf-b45c-4c74-953c-966a514e3934/scratchpad/isodump17'
RUN = '/ml2/hypergan/gan-attempts/noout-20260928/runs/E17dbg-rotated100'
tr = torch.load(f'{RUN}/final-state.pt', map_location='cpu', weights_only=False)['trainer']['models']
Wg = tr['G']['weight'].double().numpy(); bg = tr['G']['bias'].double().numpy(); ce = centers('rotated100')
def dist(z): return np.linalg.norm(assign(z.double().numpy() @ Wg.T + bg, ce)[1], axis=1)
ev = [torch.load(f) for f in sorted(glob.glob(f'{D}/iso-*.pt'))]
steps = np.array([e['step'] for e in ev]); nfl = np.array([e['n_flag'] for e in ev]); ndead = np.array([len(e['dead']) for e in ev])
print(f'{len(ev)} acted evaluations, steps {steps.min()}..{steps.max()}; flagged per acted evaluation: min {nfl.min()} p10 {np.percentile(nfl, 10):.0f} median {np.median(nfl):.0f} p90 {np.percentile(nfl, 90):.0f} max {nfl.max()}; evaluations with < 40 flagged: {(nfl < 40).sum()}')
print(f'rows re-drawn per acted evaluation (flagged minus rows the ordinary moves took): min {ndead.min()} median {np.median(ndead):.0f} max {ndead.max()}; acted evaluations per 1000 steps: ' + ', '.join(f'{lo // 1000}k:{int(((steps > lo) & (steps <= lo + 1000)).sum())}' for lo in range(1000, 7000, 1000)))
# the dumps at the unit level
rows = []      # (step, row, parent, d_dead_now, d_parent, z_dead, z_parent)
for e in ev:
    dd, dp = dist(e['z_dead']), dist(e['z_parent'])
    for i, (r, p) in enumerate(zip(e['dead'].tolist(), e['parent'].tolist())): rows.append((e['step'], r, p, dd[i], dp[i], e['z_dead'][i].numpy(), e['z_parent'][i].numpy()))
by_row = {}
for j, t in enumerate(rows): by_row.setdefault(t[1], []).append(j)
classes = [(0, 2), (2, 3), (3, 4), (4, 6), (6, 99)]
print('\nparent distance class (sigma) | re-draws | share | child flagged again within 50 / 100 / 300 / 1000 steps (acted evaluations only; censored at the end of the run, steps <= 6000 only) | median steps to re-flag')
for lo, hi in classes:
    sel = [j for j, t in enumerate(rows) if lo <= t[4] < hi and t[0] <= 6000]
    tot_all = sum(1 for t in rows if lo <= t[4] < hi)
    gaps = []; near = 0
    for j in sel:
        t = rows[j]; nxt = [k for k in by_row[t[1]] if rows[k][0] > t[0]]
        if nxt:
            k = nxt[0]; gaps.append(rows[k][0] - t[0])
    gaps = np.array(gaps); n = max(len(sel), 1)
    fr = ' / '.join(f'{(gaps <= h).sum() / n:.3f}' for h in (50, 100, 300, 1000))
    print(f'  {lo:2d}-{hi:2d} | {tot_all:6d} | {tot_all / len(rows):.3f} | {fr} | {np.median(gaps) if len(gaps) else float("nan"):.0f}')
# distance of the re-flagged child from the parent's location, in the latent metric normalised by the lattice noise: use data-space via final G
print('\nwhen a child is flagged again <= 300 steps later: its new distance to the nearest centre (sigma) vs the distance of the parent it was cloned from')
pairs = []
for j, t in enumerate(rows):
    if t[0] > 6700: continue
    nxt = [k for k in by_row[t[1]] if 0 < rows[k][0] - t[0] <= 300]
    if nxt: pairs.append((t[4], rows[nxt[0]][3]))
pairs = np.array(pairs)
if len(pairs):
    print(f'  {len(pairs)} re-flags within 300 steps; corr(parent distance, new distance) {np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]:.2f}')
    for lo, hi in classes:
        s = (pairs[:, 0] >= lo) & (pairs[:, 0] < hi)
        if s.any(): print(f'  parent {lo}-{hi} sigma: {int(s.sum())} re-flags, median new distance {np.median(pairs[s, 1]):.1f} sigma, share of these re-flags with new distance >= 3 sigma {np.mean(pairs[s, 1] >= 3):.3f}')
# rows re-drawn from core parents (< 2 sigma) and flagged again: leak from the bulk?
core = [j for j, t in enumerate(rows) if t[4] < 2 and t[0] <= 6000]
core_re = sum(1 for j in core if any(0 < rows[k][0] - rows[j][0] <= 300 for k in by_row[rows[j][1]]))
print(f'\nchildren cloned from a CORE parent (< 2 sigma) and re-flagged within 300 steps: {core_re} of {len(core)} = {core_re / max(len(core), 1):.4f}')
shell = [j for j, t in enumerate(rows) if t[4] >= 3 and t[0] <= 6000]
shell_re = sum(1 for j in shell if any(0 < rows[k][0] - rows[j][0] <= 300 for k in by_row[rows[j][1]]))
print(f'children cloned from a SHELL parent (>= 3 sigma) and re-flagged within 300 steps: {shell_re} of {len(shell)} = {shell_re / max(len(shell), 1):.4f}')
# the late window: parent classes and dead classes, steps >= 4000
late = [t for t in rows if t[0] >= 4000]
print(f'\nsteps >= 4000: {len(late)} re-draws; dead rows: <3 sigma {np.mean([t[3] < 3 for t in late]):.4f}, 3-4 {np.mean([3 <= t[3] < 4 for t in late]):.3f}, 4-6 {np.mean([4 <= t[3] < 6 for t in late]):.3f}, >=6 {np.mean([t[3] >= 6 for t in late]):.3f}; parents: <2 {np.mean([t[4] < 2 for t in late]):.3f}, 2-3 {np.mean([2 <= t[4] < 3 for t in late]):.3f}, 3-4 {np.mean([3 <= t[4] < 4 for t in late]):.3f}, 4-6 {np.mean([4 <= t[4] < 6 for t in late]):.3f}')
