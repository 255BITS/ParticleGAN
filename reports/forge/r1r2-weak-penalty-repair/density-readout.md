# Saved density audit for weaker positive R1/R2

**Reject this weaker-penalty repair.** Constant gamma .01 on exact
non-saturating AMSGrad-m110 passes two of six ordinary tasks plus its own clock;
Gaussian, pole, Ring and words fail. Its measured gamma .1 parent passes five
ordinary tasks plus clock, with Ring failing. This is a frozen comparison, not
a new leaderboard or qualification.

The [audit](density-audit.json) reproduces all **96 new Ring panels and six
original gate metrics exactly**, with maximum difference zero. Covariance passes
0/96, full passes 0/96, terminal passing suffix 0/5. The exact cached parent
retains its [original 96-panel audit](../r1r2-score-law-repair/density-audit.json)
and grades; this audit verifies its saved endpoint and optimizer without
regrading it.

| Whole recipe | Best covariance (step) | Endpoint covariance ≤.85 | Minimum eigen ratio ≥.15 | HQ ≥.85 | Modes ≥16 | Mass TV ≤.15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact NS AMSGrad parent, gamma .1 | 1.501881 (1584) | 1.556546 | .094075 | .888672 | 16 | .049805 |
| Same recipe, gamma .01 | 4.198951 (1484) | 4.625256 | .189267 | .899902 | 15 | .180664 |

Every new terminal panel at updates 1534/1550/1567/1584/1600 fails covariance,
mode coverage and mass balance; the first two also fail minimum spread. Covariance
is 4.429584/4.521804/4.605873/4.802590/4.625256. Across all 96 panels, covariance
and modes fail 96 times each, mass 94, HQ 65 and minimum spread 5; sample count
passes every panel. Passing endpoint HQ and minimum spread do not establish the
joint sustained gate.

## Saved shape diagnosis

Geometry uses original float32 assignments and float64 population moments. All
16 nearest-center components have available moments, despite only 15 qualifying
HQ modes. Ratios divide by target component variance; target trace is 2.

| Saved endpoint diagnostic | Gamma .1 parent | Gamma .01 |
| --- | ---: | ---: |
| Mean radial variance (range) | .297 (.108–.679) | 5.288 (.193–67.239) |
| Mean tangent variance (range) | 3.053 (2.379–4.240) | 3.405 (1.430–12.951) |
| Mean between-center covariance trace | .342 | 6.336 |
| Mean local Jacobian noise trace | 2.739 | 2.859 |
| Mapped prior means outside 3 / 4 sigma | 1 / 0 | 4 / 3 |

The radial mean is an unweighted component average with a large extreme; some
directions remain compressed. These saved outputs reject a uniform radial
repair and show poorer density and allocation. The local `sigma_z² J Jᵀ` proxy
and center trace are not an exact additive covariance partition or proof of
historical forces, sampled-row ownership or a unique causal mechanism.

## Other failed gates

Pole's endpoint passes movement (1.033388 ≥.30) and gradient (.592705 ≤1), but
its original sustained gate fails: only four passing observations total and
terminal suffix **2/5**. Gradients at the last five checks are
2.061161/1.753642/1.296387/.941948/.592705, while movement passes throughout
those five. Endpoint-only interpretation would hide the instability.

Words records **0/24 confirmed joint passes**, versus the parent's 24/24. It
ends with three modes, quality 1, failed inverse reconstruction and mass TV
.401758. Gaussian likewise records zero confirmed passes. Token and AE retain
their original sustained passes; clock passes all four comparisons. See the
[certified publication](publication/results.json) for exact task evidence.

## Provenance and reproduction

Execution is `3cb46b91121f42f8b171a77f17776365aebad1d8`; new Ring attempt is
`dc9341994b694c8bad549b1c09192c82`, exact parent
`fca1f1c665354d3089066abe90f291ad`. Source remains
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`;
runtime remains `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
The audit verifies all source files, captured actual initial-tensor and ordered
target-batch identities, checkpoint digests and every saved named stream. Native
AMSGrad groups have G/D betas(.5,.99), prior(0,.99), 1,600 steps, LR .0085 and
epsilon 1e-8. Both positive penalties use gamma .01 every update.

No live model, optimizer or evaluator executes. The audit adds zero draws and
updates; Python/NumPy/Torch CPU RNG states remain unchanged. Captured receipts
are checked without reconstructing initial tensors. Exact CLI stdout and stderr
are retained in `/mnt/ml7tb/ParticleGAN-forge/r1r2-weak-penalty-repair-20261012/`
as `density-audit.stdout.log` and `density-audit.stderr.log`.

```sh
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-weak-penalty-repair/density_audit.py --execution-commit 3cb46b91121f42f8b171a77f17776365aebad1d8 > reports/forge/r1r2-weak-penalty-repair/density-audit.json
```
