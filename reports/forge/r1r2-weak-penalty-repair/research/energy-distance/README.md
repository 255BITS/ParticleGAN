# Signed energy-U software prerequisite

**Numerical and resource prerequisites pass; no training repair is established.**
The [probe](probe.py) implements one nonsquared Euclidean statistic:
`2 mean d(fake,real) − off-diagonal mean d(fake,fake) − off-diagonal mean d(real,real)`.
Targets are detached; both fake arguments differentiate. Diagonal indices are
excluded, duplicate-valued off-diagonal rows remain, and signed values are never
clamped. This prerequisite selects **no coefficient, normalization, candidate or
paid comparison**. The [primary-source mechanism review](../next-density-mechanism.md)
owns the population theory and possible host-binding requirements.

## Numerical checks

The [receipt](diagnosis.json) checks eight float64 coordinates away from ties by
central differences: maximum error **2.83e-10 < 2e-9**. Target detachment,
unequal batch sizes, permutation value/gradient equivariance, rigid-motion
invariance and linear scaling all pass. Equal empirical panels can give signed
U **−.064516**; that value is retained.

The native [PyTorch `cdist` direct-distance mode](https://docs.pytorch.org/docs/2.14/generated/torch.cdist.html)
avoids a Python `n×m×d` difference tensor and the matrix-multiplication distance
formula. `where(distance>0, distance, distance.detach())` explicitly chooses
zero contribution at collisions, with no jitter or denominator floor. Exact
duplicate-row gradients are zero on CPU and CUDA. The symmetric zero pole
fixture has gradient zero on CUDA float32 and at most 6.94e-18 on CPU float64;
this term cannot separate its identical particles. The repeated token fixture
pulls the trained concept toward its target, but shared parameters can still
move the unused output despite its slot having no direct gradient. Keep hold.

Deterministic narrow/wide fixtures give opposing scale derivatives. A compressed
radial / expanded tangent circular fixture gives derivatives **−.095155 / .283430**,
so descent expands one direction and contracts the other. These atoms are not
Gaussian laws. A labeled squared-distance population-V control is zero for
equal means with different widths, while nonsquared U distinguishes that fixture.
No squared-distance arm is selected.

## Saved Ring and word limits

Both Ring endpoints retain their original attempt/checkpoint/panel hashes,
96-panel archives and FAIL grades. The probe uses only the **first 128 ordered
endpoint evaluation outputs**, against 128 deterministic atoms: eight equally
spaced circle offsets at each declared center. Those atoms match declared
component moments but are **neither Gaussian samples nor actual G-phase targets**.

| Existing endpoint; separate fixture | Signed U | Input-gradient norm | Radial expanding / available | Tangent contracting / available |
| --- | ---: | ---: | ---: | ---: |
| Exact NS AMSGrad gamma .1 | −.033981 | .011540 | 12/16 | 9/16 |
| Closed weaker gamma .01 | .232335 | .038483 | 9/16 | 8/16 |

Directions are mixed. The parent fixture's mean radial variance derivative is
6.29e-6 and tangent derivative −1.75e-7 under raw output-gradient flow. These
are local directions on an ordered prefix, not G/prior/AMS updates, historical
forces or a qualification metric. No actual Gaussian target minibatch is
reconstructed or substituted.

A separate canonical-word softmax fixture shows logit-gradient norm falling
from **.029118** at gap 2 to **2.31e-35** at gap 80 despite output-gradient norm
.292789. Marginal output matching can therefore lose parameter force at
saturation and supplies no paired inverse-encoder objective. Finite signed U
also has minibatch variance: IID unbiasedness does not apply to these coupled,
deterministic fixtures or guarantee shape repair. Raw energy scales with output
length; bandwidth-free does not determine global strength.

## Native-shape resource check and reproduction

RTX A6000, Torch 2.14.0+cu130, float32: two warmups and five synchronized
forward-plus-input-backward timings per native shape, including real/real cost.

| Host shape | Median ms | Incremental peak allocated bytes |
| --- | ---: | ---: |
| Pole 12×1 | .465 | 9,728 |
| Token 8×2 | .457 | 6,144 |
| Clock 8×2 | .452 | 6,144 |
| AE 64×2 | .455 | 137,216 |
| Gaussian 128×1 | .454 | 477,184 |
| Ring 128×2 | .454 | 543,744 |
| Words 256×168 | 1.118 | 46,023,680 |

Native backward has additional workspace; three pair matrices alone do not
describe peak memory. These synthetic array timings establish practical size;
integrated training time, budget compliance and API support remain unmeasured. Signed-U
checkpoint accounting still requires an explicit compatible consumer; existing
nonnegative transport summaries cannot be reused silently.

All 1,214 frozen scientific files, queue state and Python/NumPy/Torch CPU/all-CUDA
RNG states remain unchanged. Constructors for models, recipes and optimizers,
and random-draw entry points, are forbidden during the probe. No training,
sampling, task evaluator, grade or reservation is added. Exact final stdout
matches `diagnosis.json`; stderr is empty. Logs, including superseded software
drafts, remain under the current archive outside Git.

```sh
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-weak-penalty-repair/research/energy-distance/probe.py > /mnt/ml7tb/ParticleGAN-forge/r1r2-weak-penalty-repair-20261012/energy-distance-probe.stdout.log
```
