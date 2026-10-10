"""Finite-atom oracle for the protocol v5 particle-resolution floor.

A perfect particle sampler: n atoms drawn i.i.d. from the true component Gaussian. ParticlePrior.sample
is pure indexing into a fixed particle table (no per-sample latent noise) and G is deterministic, so the
generated component is exactly n atoms. The suite draws EVAL_SAMPLES=4096 indices uniformly over
particles=256, so each atom is repeated Binomial(4096, 1/256) times (~16); the oracle draws these
multiplicities from the same multinomial. The component is scored in isolation (the corners are 16.7
sigma apart, so nearest-mean assignment never moves an atom), with the suite's exact per-component
core/spill statistics, for every distinct covariance shape declared by a core/spill task.

Predeclared before running:
  grid   n in (3, 5, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)
  rule   N_MIN = smallest grid n such that, at n and at every larger grid n, each gated per-component
         statistic's per-observation false-fail rate (max over covariance shapes) is <= 5%.
  gated  core covariance error <= .5, core min whitened eigenvalue >= .15, spill <= .05.
Sustained columns: P(no 5-observation passing suffix at the end) if the 5 final observations were
independent draws, 1-(1-p)^5; frozen atoms (no motion) give p itself. Real runs lie in between.

python oracle.py  ->  oracle.log (the table) and oracle.jsonl; one line per (shape, n).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

GRID = (3, 5, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)
TRIALS = 20000
BOUND = .05
STABLE = 5


def shapes():
    from benchmarks.transfer_suite import vector_tasks as vt
    core = {k for k, _, _ in vt.CORE_SPILL_BOUNDS}
    out = {}
    for t in vt.TASKS:
        if core <= {k for k, _, _ in t["thresholds"]}:
            for k, c in enumerate(t["covariances"]):
                c = np.asarray(c, float)
                key = "isotropic" if np.allclose(c, np.eye(2) * c[0, 0]) else f"{t['name'].split('_', 1)[1]}[{k}]"
                out.setdefault(key, c)
    return out, vt


def statistics(u, counts, cov, vt):
    """u: (T, n, 2) whitened atoms; counts: (T, n) multiplicities. Returns core error, core eig, spill."""
    chol = np.linalg.cholesky(cov)
    m2 = (u ** 2).sum(-1)
    total = counts.sum(1)
    spill = np.where(total >= 10, (counts * (m2 > vt.SPILL_MAHALANOBIS2)).sum(1) / np.maximum(total, 1), 1.)
    w = counts * (m2 <= vt.CORE_MAHALANOBIS2)
    wt = w.sum(1)
    mean = (w[..., None] * u).sum(1) / np.maximum(wt, 1)[:, None]
    x = u - mean[:, None]
    s = np.einsum("tn,tni,tnj->tij", w, x, x) / np.maximum(wt, 1)[:, None, None]  # whitened core covariance
    emp = chol @ s @ chol.T
    err = np.linalg.norm(emp - cov, axis=(1, 2)) / np.linalg.norm(cov)
    eig = np.linalg.eigvalsh(s)[:, 0]
    ok = wt >= 10
    return np.where(ok, err, 1.), np.where(ok, eig, 0.), spill


def draw(rng, trials, n, particles=256, samples=4096):
    counts = rng.multinomial(samples, np.full(particles, 1 / particles), size=trials)[:, :n].astype(float)
    return rng.standard_normal((trials, n, 2)), counts


def validate(vt, rng):
    """Check the vectorized replica against vt.score_samples on unequal_mass, rare component = n atoms."""
    import torch
    spec = next(t for t in vt.TASKS if t["name"] == "vector_unequal_mass")
    cov = np.asarray(spec["covariances"][3])
    for n in (3, 5, 12):
        u, counts = draw(rng, 1, n)
        atoms = np.asarray(spec["means"][3]) + u[0] @ np.linalg.cholesky(cov).T
        rare = np.repeat(atoms, counts[0].astype(int), axis=0)
        # The other components: exact-shape filler so their statistics are irrelevant to the check.
        rest = vt.sample_target(dict(spec, masses=[.55/.98, .30/.98, .13/.98, 0.]), 4096 - len(rare),
                                torch.Generator().manual_seed(n), spec["steps"])
        m = vt.score_samples(torch.tensor(np.concatenate([rest.numpy(), rare]), dtype=torch.float32), spec, spec["steps"])
        err, eig, spill = (float(a[0]) for a in statistics(u, counts, cov, vt))
        got = m["component_core_covariance_errors"][3], m["component_core_eigen_ratios"][3], m["component_spill"][3]
        assert np.allclose((err, eig, spill), got, atol=2e-4), (n, (err, eig, spill), got)


def main():
    cases, vt = shapes()
    rng = np.random.default_rng(0)
    validate(vt, rng)
    rows = []
    with open(HERE / "oracle.log", "w") as log, open(HERE / "oracle.jsonl", "w") as out:
        log.write("# per-observation false-fail rate of a perfect n-atom sampler (sustained, independent obs)\n")
        for name, cov in cases.items():
            for n in GRID:
                err, eig, spill = statistics(*draw(rng, TRIALS, n), cov, vt)
                p = dict(core_err=float((err > .5).mean()), core_eig=float((eig < .15).mean()),
                         spill=float((spill > .05).mean()))
                p["any"] = float(((err > .5) | (eig < .15) | (spill > .05)).mean())
                row = dict(shape=name, n=n, trials=TRIALS, fail=p,
                           sustained_independent={k: 1 - (1 - v) ** STABLE for k, v in p.items()})
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                log.write(f"{name:<18} n={n:3d} core_err={p['core_err']:.3f} core_eig={p['core_eig']:.3f} "
                          f"spill={p['spill']:.3f} any={p['any']:.3f} "
                          f"sust_any={row['sustained_independent']['any']:.3f}\n")
                log.flush()
        worst = {n: max(max(r["fail"][k] for k in ("core_err", "core_eig", "spill")) for r in rows if r["n"] == n)
                 for n in GRID}
        n_min = next(n for n in GRID if all(worst[m] <= BOUND for m in GRID if m >= n))
        log.write(f"# worst per-metric rate by n: {json.dumps({n: round(v, 4) for n, v in worst.items()})}\n")
        log.write(f"# N_MIN = {n_min}\n")
    print(open(HERE / "oracle.log").read())


if __name__ == "__main__":
    main()
