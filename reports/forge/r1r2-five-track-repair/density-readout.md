# Saved Ring16 diagnosis for the five repair tracks

None of the five new recipes passes sustained Ring16. Across **480 saved
observations**, every recipe has **0/96 covariance-bound passes**, **0/96 full
passes** and **0/5 terminal suffix**. All six gate metrics reproduce exactly
from the saved samples: maximum absolute difference is zero for every metric
and recipe. Momentum improves the best earlier covariance endpoint, but retains
both excessive covariance error and a failed minimum-spread bound.

The [compact source-bound audit](density-readout.json) records exact attempts,
recipe revisions, request/raw/graded/sample/state hashes, unchanged thresholds,
selected trajectory points and all endpoint component diagnostics. The
[read-only reproduction source](density_readout.py) constructs no models,
draws no new samples and performs no updates; Python and Torch RNG states remain
unchanged. Results remain provisional and grant no qualification or adoption.

## Numerical failures

All tracks use the same constant beta2=.99/gamma=.1/LR=.0085 parent, each with
one declared recipe delta. The table reports those whole recipes in track order,
without selecting a different policy for each task. Original bounds remain
covariance error ≤0.85, HQ ≥0.85, all 16 modes, mass TV ≤0.15, minimum eigen ratio
≥0.15 and at least five terminal passing checks at update 1,600.

| Recipe delta | Best covariance error (step) | Endpoint covariance error | Endpoint HQ | HQ modes | Mass TV | Min eigen ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Critic LR multiplier 1→2 | 2.873421 (1,534) | 3.609186 | 0.636719 | 16 | 0.061523 | 0.197063 |
| Prior LR multiplier 1→0.25 | 2.912019 (1,534) | 3.175129 | 0.475342 | 12 | 0.030518 | 0.098492 |
| AMSGrad enabled | 2.416893 (1,434) | 2.960424 | 0.839355 | 16 | 0.053223 | 0.310603 |
| Local transport weight 0→0.1 | 3.268189 (1,584) | 4.653696 | 0.557617 | 12 | 0.041504 | 0.777843 |
| Adam beta1 0→0.5 | 1.098351 (1,584) | 1.185580 | 0.936523 | 16 | 0.045898 | 0.052811 |

The [previous constant parent](../r1r2-tier1-search/README.md) had endpoint
covariance error **2.386427**, HQ **0.842285**, and zero full passing observations.
All four non-momentum children worsen its covariance endpoint. Momentum reaches
**1.098351** at 1,584, improving the earlier search's best **1.409079**, while
still exceeding 0.85. Its last five covariance errors are
**1.293452 / 1.595524 / 1.145399 / 1.098351 / 1.185580**; none passes the bound.
At the endpoint, modes, HQ and mass allocation pass, but covariance and minimum
eigen ratio both fail. Higher precision alone does not solve local density.

AMSGrad never reaches HQ 0.85 in any saved observation, despite all modes at
the endpoint. Slower prior and local transport lose HQ mode coverage at their
endpoints. Every arm balances endpoint mass within the original tolerance;
mass balance does not establish the required component shape.

## What the saved geometry shows

Population moments use the original scorer's float32 nearest-component sample
assignments and float64 arithmetic. Radial points from the origin to the target
mean; tangent is perpendicular. Every component has enough assigned samples and
mapped prior means for this diagnostic. That availability does not mean the
task's stricter HQ mode gate passes. The script preserves explicit UNAVAILABLE
geometry for insufficient component counts, with no substituted measurement.

| Recipe delta | Mean radial variance / target | Tangent variance / target (range) | Mapped means outside 3σ /256 | Local-noise trace | Between-mean trace |
| --- | ---: | ---: | ---: | ---: | ---: |
| Critic LR multiplier 1→2 | 4.037484 | 1.910–9.424 | 49 | 3.882009 | 4.625130 |
| Prior LR multiplier 1→0.25 | 1.035302 | 2.699–7.777 | 104 | 5.522423 | 2.517367 |
| AMSGrad enabled | 2.588217 | 1.621–5.149 | 3 | 3.622355 | 1.925475 |
| Local transport weight 0→0.1 | 5.220207 | 1.952–8.735 | 86 | 3.826891 | 4.935850 |
| Adam beta1 0→0.5 | 0.937245 | 1.364–4.176 | 3 | 2.098440 | 1.038392 |

Trace columns are unweighted means over the 16 components, in target-variance
units; the desired total component trace is 2. The local-noise estimate is
`0.1² × J_G(z) × J_G(z).T` at each saved learned prior mean. The between-mean
term is the covariance of G evaluated at those saved means. Neither is the
task's sampling law, and their sum is not an exact variance decomposition.
Noise can cross activation boundaries and mode assignments; sampled row IDs
are unavailable. The audit does not establish one causal failure mechanism.

Slowing prior motion did not reduce tangent stretch in this recipe: every
component remains broad along the tangent, and 104 mapped means lie outside a
3-sigma target region. Faster critic and local transport also introduce large
radial and between-mean spread. AMSGrad leaves only three means outside 3 sigma,
but its worst radial component has **20.754923** times target variance; a few
outlying centers can strongly affect component moments.

Momentum has the smallest tangent range among these five endpoints, yet every
component still exceeds target tangent variance. Radial variance ranges from
**0.054115 to 2.842621** times target; its low minimum eigen ratio confirms
over-compression in at least one direction. The mean local-noise trace is near
2, but component anisotropy and between-mean dispersion remain. Aggregate
traces or an isotropic contraction cannot substitute for all original shape
and spread bounds.

## Provenance and stopping

All five Ring16 runs belong to execution commit
`9ad31157b8327044f9364b5fc458659cced0c8f0`, common scientific source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`,
and the recorded runtime. The [registration](registration.json) and
[manifest](manifest.json) retain their exact pre-run identities. The audit checks
the frozen scorer source hashes and original full sample/gate equations,
including insufficient-component and nonfinite-panel behavior. It avoids the
reference-target and SW1 branches because these are outside the six Ring16
gates and would consume additional deterministic draws.

This is saved-state analysis of the declared 1,600-update runs, with no new
training cost or qualification. Full-suite results own the final recipe
comparison; a failed Ring16 gate already rules out a complete Tier 1 repair.
Retain these negatives and the recorded baseline. No automatic continuation,
seed change, sampler contraction, task/gate revision or additional rate point
is authorized by this diagnosis.

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-five-track-repair/density_readout.py > reports/forge/r1r2-five-track-repair/density-readout.json
```
