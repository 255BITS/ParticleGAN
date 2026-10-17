"""Best-case per-row mean-test power for a proposed prospective detector.

This is an oracle bound, not a candidate or a benchmark run. It assumes a
known gradient direction, known innovation covariance and known AR(1)
correlation. An implementable vector test that estimates these from the row's
own touches cannot have better power under the same family.

The generic multiplicity scenarios are one signal, 1% of rows, and 5% of rows;
they are not inferred from any native task. The tested touch budgets are the
dyadic block grid and an optimistic 700-touch upper budget. No benchmark
geometry, gradients, output samples, scores or seeds enter this calculation.
"""
import json
import math
from pathlib import Path
from statistics import NormalDist


N = 20_000
Q = 0.05
POWER = 0.90
RHO = (0.0, 0.5, 0.9)
M = (4, 8, 16, 32, 64, 128, 700)
K = (1, N // 100, N // 20)
H_N = sum(1.0 / j for j in range(1, N + 1))
NORMAL = NormalDist()


def information(m: int, rho: float) -> float:
    # 1' Sigma^{-1} 1 for AR(1) unit marginal variance.
    return (m * (1 - rho) + 2 * rho) / (1 + rho)


def min_effect(m: int, rho: float, alpha: float) -> float:
    # One-sided oracle z-test, standardized mean shift for >= 90% power.
    return (NORMAL.inv_cdf(1 - alpha) + NORMAL.inv_cdf(POWER)) / math.sqrt(information(m, rho))


rows = []
for method in ('BH', 'BY'):
    for k in K:
        # k-th threshold is optimistic: it assumes the other k-1 true signals
        # are also ranked above this row by the multiple-testing procedure.
        alpha = Q * k / (N * (H_N if method == 'BY' else 1.0))
        for rho in RHO:
            for m in M:
                rows.append(dict(method=method, n=N, q=Q, signals=k,
                                 alpha_k=alpha, rho=rho, touches=m,
                                 effective_information=information(m, rho),
                                 minimum_standardized_effect=min_effect(m, rho, alpha),
                                 assumption='oracle known direction/covariance/correlation; one-sided'))

out = Path(__file__).with_name('oracle-feasibility.json')
out.write_text(json.dumps(dict(harmonic_n=H_N, power=POWER, rows=rows), indent=2) + '\n')
print(f'N={N} q={Q} H_N={H_N:.4f} oracle power={POWER}')
for method in ('BH', 'BY'):
    for k in K:
        vals = [r for r in rows if r['method'] == method and r['signals'] == k]
        print(method, f'k={k}', f'alpha_k={vals[0]["alpha_k"]:.3g}',
              ' | '.join(f'rho={rho}: ' + ', '.join(f'm{m}={next(r for r in vals if r["rho"] == rho and r["touches"] == m)["minimum_standardized_effect"]:.2f}' for m in (32, 128, 700))
                         for rho in RHO))
