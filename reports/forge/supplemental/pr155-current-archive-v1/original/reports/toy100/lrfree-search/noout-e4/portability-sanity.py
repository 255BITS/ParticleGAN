"""CPU-only sanity of the row-evidence gate under data changes (not the S3 regression; no training): (1) invariance of the flags under g -> a R g (units, rotation),
(2) null false-flag fraction and planted-drift recall for other table sizes N and dimensions d. usage: python portability_sanity.py"""
import sys, math, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
from particlegan.row_evidence import RowEvidence
torch.manual_seed(0)
def run(N, d, steps=600, a=1.0, R=None, drift_rows=0, rho=1.0, p=0.1, seed=1):
    g = torch.Generator().manual_seed(seed)
    ev = RowEvidence(torch.zeros(N, d, dtype=torch.float64), window=50, level=0.05)
    mu = torch.zeros(N, d, dtype=torch.float64); mu[:drift_rows, 0] = rho
    for t in range(steps):
        touched = torch.rand(N, generator=g) < p
        noise = torch.randn(N, d, generator=g, dtype=torch.float64)
        gr = (mu + noise) * touched.unsqueeze(1)
        if R is not None: gr = gr @ R.T
        ev.update(gr * a)
    return ev
def rot(d, seed=3):
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)); return q
print('(1) invariance: same gradient stream, transformed by a*R (d=2, N=2000, 30 planted drifting rows)')
base = run(2000, 2, drift_rows=30)
for a in (0.5, 2.0, 1e-3):
    t = run(2000, 2, a=a, R=rot(2), drift_rows=30)
    print(f'  a={a:g}: flags identical {bool((t.flag == base.flag).all())}, flagged {int(t.flag.sum())} vs {int(base.flag.sum())}')
print('(2) other sizes / dimensions: null false flags (no drift) and recall of 3% planted rows with rho=1')
print('    N      d   null flagged fraction   planted recall (rho=1)  [d=32: the test needs n_eff >= 3d = 96 and the window caps n_eff at 99 (touches: 60 in 600 steps), so it never runs]  (BH Q=.05; the hold rule engages when the flagged fraction exceeds Q, so for N < 1/Q = 20 a single flagged row already exceeds it)')
for N, d in ((12, 4), (60, 2), (200, 2), (2000, 2), (20000, 2), (2000, 3), (2000, 8), (2000, 32)):
    n0 = run(N, d)
    k = max(1, int(.03 * N)); n1 = run(N, d, drift_rows=k)
    rec = float(n1.flag[:k].float().mean())
    print(f'  {N:6d} {d:3d}   {float(n0.flag.float().mean()):.4f}                  {rec:.2f}     (a single flagged row exceeds the hold budget)' if N < 20 else f'  {N:6d} {d:3d}   {float(n0.flag.float().mean()):.4f}                  {rec:.2f}')
