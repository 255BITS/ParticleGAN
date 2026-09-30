"""Compact side-by-side of two native runs from their native100-diagnostics.jsonl (checkpoints every 250 steps): clean live outlier fraction, noisy live precision,
mode mass TV. usage: python compare_series.py RUN_A RUN_B [step ...]   (run directories; default steps 250 500 1000 1500 2000 3000 4000 5000 6000 7000)"""
import json, sys
def load(d):
    out = {}
    for l in open(f'{d}/native100-diagnostics.jsonl'):
        r = json.loads(l)
        try:
            cl, nz = r['clouds']['clean']['live'], r['clouds'].get('noisy', {}).get('live', {})
            out[r['step']] = (cl['outlier_fraction'], nz.get('precision'), nz.get('mass_tv') or cl.get('mass_tv'), r.get('birth_death', {}).get('counters', {}))
        except Exception:
            pass
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
steps = [int(s) for s in sys.argv[3:]] or [250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000]
print(f"{'step':>5} | {'outlier A':>9} {'outlier B':>9} | {'prec A':>7} {'prec B':>7} | {'tv A':>6} {'tv B':>6} | B iso moves")
f = lambda x, n=4: '-' if x is None else f'{x:.{n}f}'
for s in steps:
    if s in a and b:
        sb = b.get(s)
        print(f"{s:>5} | {f(a[s][0]):>9} {f(sb[0] if sb else None):>9} | {f(a[s][1]):>7} {f(sb[1] if sb else None):>7} | {f(a[s][2]):>6} {f(sb[2] if sb else None):>6} | {sb[3].get('iso_moves', '-') if sb else '-'}")
