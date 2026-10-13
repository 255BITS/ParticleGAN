# Saved critic regularizer balance

This separate deterministic fixture probes saved paired and non-saturating
AMSGrad critics. It uses their own 4,096 saved endpoint fake outputs and the
16 declared Ring target centers. Centers omit target noise; these are not saved
training pairs. It constructs no models, draws no samples, takes no optimizer
steps and changes no gate. See the [receipt](diagnosis.json) and [reproducer](probe.py).

| Saved critic | Classification parameter-gradient norm | Gamma-weighted R1+R2 norm | Ratio | Gradient cosine |
| --- | ---: | ---: | ---: | ---: |
| Paired AMSGrad | .087878 | .033347 | .379473 | -.642384 |
| Non-saturating AMSGrad | .104903 | .031985 | .304900 | -.745931 |

Both critics retain gamma .1. The positive penalty opposes classification in
this fixture, as regularization is intended to do. Its parameter gradient is
material relative to classification even though its scalar loss is small.
This makes a lower-strength trajectory contrast intelligible; it does not
establish excessive smoothing, an optimal coefficient or a future density pass.
Native AMSGrad histories/preconditioning and earlier training inputs are not
reconstructed. The paired classification is the exact expectation across all
center/fake pairs, rather than a realized same-size batch pairing.

Functional analytic input Jacobians are differentiated with respect to saved D
weights. Sixty-six central-difference derivative checks across the three
objectives have maximum absolute error **1.49e-9**, below the explicit **5e-7**
software bound. All 1,214 scientific files and original checkpoint/panel hashes
are verified. Python, NumPy and Torch CPU RNG states are unchanged. Tiny parameter
perturbations are numerical derivative fixtures, not altered training initializers.

The separately [registered weaker-penalty comparison](../../../r1r2-weak-penalty-repair/PLAN.md)
selects one historical coefficient as a tenfold effect-size contrast, with its
own full-suite falsifier. This diagnostic supplies no qualification or loss grid.

```sh
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-score-law-repair/research/regularizer-balance/probe.py
```
