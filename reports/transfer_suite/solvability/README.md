# Solvability: useful fixes, no universal winner

**The practical failures are worth fixing.** Target and representation controls
show that every healthy vector/ring task has jointly attainable metrics with
the existing generator and 256 particles. The three failing image targets are
also representable by the original generator under supervised training.
These controls establish feasibility; they do not count as GAN wins.

Actual GAN training now supplies sustained solutions for **16/16 practical
tests**, using different configurations.
This is a per-task solvability result, **not one configuration passing 16/16**.
We did not find a shared configuration passing every behavioral metric.

## Working reference leaderboard

The useful immediate improvement is **residual nearest-neighbor upsampling,
width 16**, for the healthy image tasks, retaining the original RpGAN logistic
loss, b_cap coefficient 3 / κ1.25, Adam, particle settings and cosine schedule.
All four image tasks sustain success in their original 600 updates.

| Reference profile | Required live | Data | Dynamics | Images | Practical total | Balanced score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original cosine / transpose images | 9/9 | 3/6 | 0/6 | 1/4 | 4/16 | 25.0% |
| **Cosine / residual16 images** | **9/9** | **3/6** | **0/6** | **4/4** | **7/16** | **50.0%** |
| Cap10 / residual16 images | 8/9; **ineligible** | 4/6 | 0/6 | 3/4 | 7/16 | 47.2% |

This compares explicit host configurations, including the changed image
architecture. It is not a learned-controller gain or an equal-parameter-count
comparison. The residual image reference reuses the 29 unchanged required,
vector/dynamics and diagnostic episodes from the original cosine study; its
four changed image tasks were rerun in the native host. Every reused episode
and hash is identified in [the profile](reference_profile.json).
[Complete profile curves](reference_profile_results.json.gz).

The cap10 row changes the cap coefficient across hosts and uses residual16 for
images. It preserves the R1+R2 stress test's declared arm and coefficient.
It loses the required ring badly: **4/8 modes, HQ57.6%**. Its image bars case
also falls to HQ81.25%, below 90%. More practical passes cannot rescue a required
regression. Production defaults remain unchanged; the residual image reference
is a better starting point for subsequent controller comparisons.

## Which failures are passable?

All PASS entries below require live weights, a complete 24-check curve and at
least five final passing checks, with the original metric thresholds. EMA is
retained separately. These rows deliberately show individual solver witnesses.

| Practical task | Original cosine | Sustained GAN witness | Resource/setup change |
| --- | --- | --- | --- |
| Two broad modes | PASS | Original cosine | None |
| Unequal mode masses, including 2% mode | FAIL | Density witness | 4,096 particles / batch 256 / 6,000 updates; other settings below |
| Unequal component widths | FAIL | Cap coefficient 10 | Same architecture, particles and 1,200-update budget |
| Anisotropic components | PASS | Original cosine | None |
| Overlapping Gaussians | FAIL | Half LR | Same architecture/particles/budget; confirms at 1,150/1,200 |
| Continuous spiral | PASS | Original cosine | None |
| Fast critic | FAIL | Prior LR multiplier 30 | Same architecture/data/budget; original multiplier 10 |
| Slow critic | FAIL | Two D updates per G update | 6,000 D / 3,000 G updates; original reduced D LR retained |
| Small batch | FAIL | Original settings, 3× updates | 1,200→3,600 updates |
| Larger critic | FAIL | Original settings, 3× updates | 1,200→3,600 updates |
| Long horizon | FAIL | Original settings, 3× updates | 2,400→7,200 updates |
| R1+R2 dynamics | FAIL | Extended stock-style card | More capacity/support and 1,200→3,600 updates; R1+R2 remains .1 |
| Image stripes | FAIL | Residual16, original b_cap | Architecture changed; same 600 updates |
| Image bars | FAIL | Residual16, original b_cap | Architecture changed; same 600 updates |
| Image blobs | PASS | Original cosine or residual16 | Residual16 retains success |
| Image intensity | FAIL | Residual16, original b_cap | Architecture changed; same 600 updates |

More updates establish solvability at that larger budget. They are not a
same-budget fix, early-stopping result or speedup. The former reserved cadence
case is now inspected development data and remains unresolved; the previously
solved annulus and residual bars are not new independent transfer evidence.

The slow-critic solution retains its original D learning-rate multiplier .75,
data, architecture, particles and penalty. It changes the update ratio and
budget: D updates every outer step, G every second step, for 6,000 outer steps.
Eleven final observations pass; final HQ is 100%, covariance error .2384,
minimum covariance eigen ratio .6403 and mode-mass TV .1392. The same ratio at
3,600 outer steps only passes its final two checks and remains a failure.
[Dynamics matrix, all failures and exact configurations](dynamics/MATRIX.md).

## What the search explains

**Image optimization depends on architecture.** Residual16 reaches final HQ
100% / 93.75% / 100% / 100% on stripes / bars / blobs / intensity and confirms
at steps 225 / 550 / 550 / 575. Residual12, with the original discriminator width
and fewer generator parameters, solves stripes and intensity. Transpose16 uses
the same larger discriminator as residual16 but solves 0/4. Greater width alone
does not explain the result. Doubling the original transpose budget to 1,200
also fails to produce a shared solution.
[All 15 shared image cards, controls and diagnostics](images/README.md).

**Vector failures can hide behind high HQ.** In an exact baseline reproduction,
the rare mode has six actual particle outputs; two lie far outside its target
spread. The narrowest unequal-width component has three bad particle outputs.
Those tails dominate covariance error even though most samples look good.
Replacing just one of 4,096 target samples by a distant outlier changes the
unequal-width covariance error from .0416 to 3.008 while HQ changes only from
98.828% to98.804%. Both metrics are doing different, intended jobs.
[Actual particle coordinates and MoG results](mog/README.md).

The rare 2% mode is only about five of 256 uniform particles. One empirical
target support fails its minimum-spread bound, while 512/1,024 supports pass.
A deterministic moment-balanced 256-particle construction passes too, using
the exact original generator architecture. This is sensitivity to finite
support and optimization, not proof that the original task is impossible.
The construction matches these coarse metrics, not an exact Gaussian density.
[Target, representation and tail controls](controls/README.md).

The successful rare-mode GAN uses 4,096 particles, batch 256, 6,000 updates,
LR .0006, Adam (0,.999), prior regularization 1, cap coefficient 1 / κ1 and prior
LR multiplier 10. It confirms at 4,250, with HQ 99.58%, covariance error .1044 and
minimum covariance eigen ratio .7811. Reducing particles to 256 at 6,000 updates,
or retaining 4,096 particles at 1,200 updates, fails. Neither ablation alone
establishes a minimum resource requirement. The same larger card fails the
unequal-width and overlapping tasks, so it is not a shared winner.

## Shared data configurations

These rows each use one setting across all six data tasks. The three initially
passing cases were explicitly rerun for each promising single-knob candidate.

| Shared setting | Sustained data tests /6 |
| --- | ---: |
| Original cosine | 3/6 |
| Cap coefficient 10 | 4/6 |
| Adam β2=.999 | 4/6 |
| Half LR | 2/6 |
| Cap10 + half LR | 3/6 |
| Cap10 + β2=.999 | 2/6 |
| Cap10 + prior LR multiplier 30 | 3/6 |

Combining individual fixes does not reliably retain their wins. Alternate
interpolated penalties, Wasserstein/eikonal variants and sixteen MoG-prior
attempts also fail to supply a shared solution. Every attempt is retained:
[single-knob screen](vectors/screen/README.md),
[combinations](vectors/combinations/README.md),
[critic geometry](vectors/geometry/README.md),
[regressions](vectors/regressions/README.md),
[larger density card](vectors/density/README.md),
[resource reductions](vectors/reduction/README.md).

## Reproduction and validation

Reproduce the four-task image winner without the research shim:

```bash
python -u -m benchmarks.transfer_suite.solvability_search \
  --plan benchmarks/transfer_suite/plans/residual16.json \
  --output /tmp/residual-image-reference > /tmp/residual-image-reference.log 2>&1
tail -f /tmp/residual-image-reference.log
```

Use `plans/rare_mode_witness.json` for the larger rare-mode experiment. Each
run records the original and effective configuration, full live/EMA curves,
actions, source/runtime hashes, failures and a continuously updated leaderboard.
The plan runner rejects changed targets/thresholds, unsupported options and
untranslated legacy-host overrides. These are input checks, not leaderboard
gates. All GAN initializations use seed 0; no seed sweeps were run.

**71 focused tests pass.** Native-host replay of the four residual16 image
runs exactly matches every live/EMA checkpoint, loss checkpoint, action and
convergence result from the scoped search, excluding time fields.
[Parity evidence](native_image_parity.json). The original image baseline also
reproduces the earlier study exactly. Source archives retain the precise
executed code, including versions preceding the plan-input validation changes.

The new search retains **254 complete GAN episodes and four supervised
controls**, including all failures, plus separate target/representation controls.
Every training curve has all 24 observations; no numerical errors occurred.
[Archive and verdict verification](validation.json) can be reproduced with
`python -m reports.transfer_suite.solvability.verify`.

These are inspected toy tasks with one initialization. Individual solvability
does not establish a universal recipe, natural-image transfer or the predictive
importance of each test. Diagnostics retain zero selection weight. The
pre-existing [CI failure](../../ci_status.md) is separate and unchanged.
