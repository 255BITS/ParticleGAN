# Saved Ring density audit

**The native non-saturating contrast fails the unchanged Ring gate.** The
[frozen audit](density_audit.py) reproduced all six gate metrics exactly on all
96 saved panels: maximum absolute discrepancy **0**. The
[receipt](density-audit.json) binds execution
`517257ff3879516ae2da64009c375ffffe9b001a` and new attempt
`fca1f1c665354d3089066abe90f291ad`; the exact cached AMSGrad parent retains
attempt `6403eb2c98224c1f8b788c1d26d5c1d7` and its original source identity.

| Endpoint metric, update 1,600 | Paired AMS parent | Non-saturating | Original bound |
| --- | ---: | ---: | --- |
| Component covariance error | 1.679120 | 1.556546 | ≤ .85 |
| Minimum eigen ratio | .159521 | .094075 | ≥ .15 |
| HQ | .879150 | .888672 | ≥ .85 |
| Mass TV | .046875 | .049805 | ≤ .15 |
| Modes | 16 | 16 | 16 |
| Samples | 4,096 | 4,096 | ≥ 4,096 |

Covariance fails **96/96** checks; the best value is **1.501881** at update
1,584. There are **0 full passing observations** and **0/5 terminal passes**.
All five terminal checks fail covariance and minimum spread; the first also
fails HQ. Across all 96 checks, minimum spread fails 31, HQ 73, mass TV 10 and
coverage 18. A modest endpoint covariance improvement does not repair density.

Observed component shape remains anisotropic. Mean radial variance falls from
**.413 to .297 times target**, while mean tangential variance falls from
**3.251 to 3.053 times target**. Every new component has radial variance below
target and tangential variance above target. The score-law change therefore
coincides with stronger radial compression. It does not establish a sufficient
anchoring repair or isolate anchoring from the changed G/D loss weights.
Functional Jacobian geometry remains a first-order proxy, not an exact additive
partition of empirical covariance.

The audit verifies the actual loss-only delta, unchanged initial tensors and
ordered target batches against the cached parent, complete named-stream
checkpoints and the full frozen source manifest. Saved G/D optimizers retain
native Adam/AMSGrad betas `(.5,.99)`; the learned prior retains `(0,.99)`.
Every parameter has 1,600 updates, LR .0085 and epsilon 1e-8, with saved AMSGrad
maxima at least the current second moment. Both positive R1/R2 terms remain
coefficient .1 every update. No model construction, live evaluation, update or
sampling draw was added; Python/NumPy/Torch CPU RNG states are unchanged.

Reproduce in the declared environment:

```sh
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-score-law-repair/density_audit.py --execution-commit 517257ff3879516ae2da64009c375ffffe9b001a
```

Exact CLI stdout and stderr remain in the external score-law archive as
`density-audit.stdout.log` and `density-audit.stderr.log`. This readout retains
the failure and registers no further arm.
