# Saved density audit for optimizer-role repair

All seven new global recipes fail sustained Ring16. The [audit](density-audit.json)
verifies **672 saved panels**, 96 per recipe; all six original gate metrics
reproduce **exactly**, with maximum absolute difference zero. Every recipe has
**0/96 covariance-bound passes**, **0/96 full passes** and terminal suffix
**0/5**. Original grades, thresholds and qualification records remain unchanged.

The [read-only reproducer](density_audit.py) imports the frozen
[five-track equations](../r1r2-five-track-repair/density_readout.py) for gate
metrics, passing windows and functional endpoint geometry. It constructs no
models, draws no samples and performs no optimizer updates. Python and Torch
CPU RNG states remain unchanged; no reference-target/global-moment/SW1 branch
is called. This is a consistency audit of saved evidence, not a new experiment.

## Original density requirements

At least 4,096 samples, all 16 HQ modes, mass TV <=0.15, HQ >=0.85, mean
component covariance error <=0.85 and minimum eigen ratio >=0.15 must hold
for five terminal observations within the original 1,600-update allowance.
Bits below describe D / G+E / sampled-prior beta1: 1 means .5, 0 means 0.
Beta2=.99, paired R1/R2 coefficient .1 every update and global LR=.0085 remain
constant. `ams_m111` adds native AMSGrad to all-.5 momentum.

| Global role policy | Best covariance error (step) | Endpoint covariance | HQ | HQ modes | Mass TV | Min eigen ratio | Endpoint failed bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| m100 | 1.976478 (1367) | 4.705740 | 0.501465 | 16 | 0.104248 | 0.752855 | HQ, covariance |
| m010 | 1.695596 (1534) | 2.403444 | 0.786377 | 16 | 0.064209 | 0.243074 | HQ, covariance |
| m001 | 2.717876 (1534) | 6.708268 | 0.124756 | 2 | 0.177734 | 0.387864 | modes, mass TV, HQ, covariance |
| m110 | 0.904193 (1584) | 1.083997 | 0.923340 | 16 | 0.068115 | 0.111723 | covariance, minimum spread |
| m101 | 1.644668 (1400) | 2.819689 | 0.832520 | 16 | 0.035400 | 0.642583 | HQ, covariance |
| m011 | 1.539979 (1534) | 1.963361 | 0.902832 | 16 | 0.060303 | 0.094217 | covariance, minimum spread |
| ams_m111 | 1.816302 (1584) | 2.009387 | 0.867432 | 16 | 0.057373 | 0.176131 | covariance |

## What m110 supports

Keeping D/G+E momentum with zero sampled-prior momentum is the strongest
observed density lead in this finite comparison. **m110** reaches covariance
error **0.904193 at update 1,584**: its HQ 0.927734, modes 16, mass TV
0.057373 and minimum eigen ratio 0.154614 pass, but covariance still misses
0.85. This single near-pass supplies no sustained qualification. Its last five
covariance errors are **1.050390 / 1.011790 / 1.040245 / 0.904193 / 1.083997**;
all exceed the bound. At the endpoint, minimum eigen ratio also fails at
**0.111723**.

The exact reused m111 control had best covariance 1.098351, endpoint
1.185580 and minimum eigen ratio 0.052811. Compared with that effective
all-.5 policy, removing sampled-prior momentum improves these density
measurements. D-only m100 and G-only m010 have best errors 1.976478 and
1.695596, so neither alone reproduces m110's improvement. These are matched
fixed-protocol interventions, not evidence of a universal optimizer rule.

**m110 still fails Gaussian and word acquisition**, as the
[whole-recipe numerical publication](publication/results.json) records.
Its Gaussian endpoint mean error 0.351987 sigma and CDF KS 0.179686 fail their
bounds. Its word endpoint has three modes, mass TV 0.401758 and exact inverse
reconstruction 0. Pole, token/AE holds and its own clock pass. Preserving
D/G momentum with P=0 is a justified density hypothesis for any separately
registered global repair; it is not a complete Tier 1 replacement or permission
to adopt a different recipe for Gaussian or words.

## Remaining shape failure

At m110's endpoint, every component still has broad tangent variance:
**1.600–3.466 times target**. Radial variance ranges from **0.133–2.496**,
with mean **0.575**. The density problem combines tangential spread and an
under-spread direction; an isotropic contraction cannot be inferred to fix
both. Only **1/256** mapped prior means lies outside a 3-sigma target region.
The local Jacobian noise-trace estimate averages **1.976** target-variance
units, near the desired total trace 2, while between-mean trace averages
**0.721**. These averages mask component anisotropy. They are separate
first-order diagnostics, not an exact additive variance decomposition or
proof that one optimizer role caused the failure.

The single-role prior-momentum recipe m001 instead collapses late to two HQ
modes and fails mass balance. D-only m100 and D/prior m101 also show late
precision/mode excursions. AMSGrad plus all-.5 momentum passes the endpoint
minimum-spread and precision bounds but has covariance error 2.009387; its
previously separate acquisition and momentum density improvements cannot be
combined as existing qualification.

## Verified provenance and limits

All seven new runs belong to execution commit
`6f634cec5c3ec6e0dcda7c1508a2eafaf9fb2bfd`, scientific source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`,
and runtime hash
`640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
The audit verifies admitted candidate revisions and job keys, frozen source
files, request/raw/graded/sample/checkpoint file hashes, complete saved-state
hashes, actual saved G/prior/D beta groups and AMSGrad flags, and every named
RNG checkpoint against its recorded state hash.

Captured actual initial-tensor metadata/hashes and ordered consumed target-batch
receipts match the exact m000 control and numerical publication. This validates
the retained instrumentation receipts; the audit does not reconstruct fresh
initialization. Functional saved-model geometry retains unavailable fields
explicitly and does not substitute center-only sampling for the positive-width
MoG task. Cached m000/m111 controls keep their original execution identities.
No training, draws, paid cost, rescored qualification or default change follows.

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-role-repair/density_audit.py > reports/forge/r1r2-role-repair/density-audit.json
```
