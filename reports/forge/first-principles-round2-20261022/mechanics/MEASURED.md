# Mechanics: one retained-window gain, two control regressions

**Do not replace the energy parent with this global recipe.** Mechanics passes
**5/15 tasks versus 6/15** for the fresh energy control. It gains overlap,
retains anisotropic vectors and blobs, but loses ring acquisition and mode hold.
Only two of the four original regression controls survive. Rare shape, unequal
width, Gaussian stability and all three native density gates remain failed.
This completes the bounded revision; no additional paid arm or parameter sweep
was launched.

The [preregistered mechanism and negative controls](README.md) remain unchanged.
The sole delta from `bispf-output-energy-v1` is
`bispf_generator_step=secant_extragradient`: replay the existing G-phase batch
and perturbations through a frozen critic, certify the unprotected normalized
G/prior field, then apply original anchor direction protection once. Learned
prior locations, SBNP .01, energy weight 1 and all task laws, initial capacity,
architectures, sampling, numerical gates and budgets remain fixed.

The [compact receipt](receipt.json) retains final metrics, phase/terminal
counts, bound summaries, aggregate movement diagnostics and original artifact
hashes. The parent [numerical appendix](../numerical-results.md) owns the full
four-arm comparison and all 60 actual-training GIFs. This report adds no
qualification or separate leaderboard.

| Task | Energy | Mechanics | Mechanics's recorded gate evidence |
| --- | --- | --- | --- |
| Gaussian smoke | PASS | PASS | Confirmed acquisition at 125; endpoint also passes |
| Gaussian stability | FAIL | FAIL | 40/72 stationary, 12/24 shifted hold; deadline fails |
| Unequal mass | FAIL | FAIL | 0/24 full checks; suffix 0 |
| Unequal width | FAIL | FAIL | 2/24 full checks; suffix 2 |
| Anisotropic | PASS | PASS | 18/24 checks; suffix 18 |
| Overlap | FAIL | PASS | 22/24 checks; suffix exactly 5 |
| Bars | FAIL | FAIL | 0/24 checks |
| Blobs | PASS | PASS | 7/24 checks; suffix 7 |
| Intensity | FAIL | FAIL | 1/24 checks; suffix 0 |
| Grid100 | FAIL | FAIL | 0/5 final checks; independent holdout fails |
| Rotated100 | FAIL | FAIL | 0/5 final checks; independent holdout fails |
| Staggered100 | FAIL | FAIL | 0/5 final checks; independent holdout fails |
| Ring16 acquisition | PASS | FAIL | 0/96 checks; suffix 0 |
| Mode hold | PASS | FAIL | 3/24 checks; suffix 3 |
| Stripes | PASS | PASS | 11/24 checks; suffix 5 |

## What improved, and what the endpoints conceal

Overlap gains the existing five-final-check window at updates 1000–1200;
energy's suffix was one. Earlier failures at 550 and 950 remain. Mean-error
bound failures decrease from four to two, but mechanics also has one sliced-W1
failure and one covariance failure, where energy had none. This is a scoped
terminal-window repair, not uniform quality over every observation. Anisotropic
passing checks improve from 8/24 to 18/24 and its suffix from 6 to 18; this
retains an existing parent pass rather than creating another repaired task.

Gaussian acquisition and stability are distinct. Energy passes smoke through
independent confirmation at update 209 despite a failed endpoint KS of .29279;
mechanics confirms at 125 and ends at .03612. Its continued stationary checks
improve from 12/72 to 40/72, shifted recovery from 4/24 to 11/24, and shifted
hold from 0/24 to 12/24. Both methods pass only one of the five deadline checks
and fail reacquisition. Mechanics's final shifted endpoint KS is .03869 versus
energy's .06514, but its shifted passing suffix is only three. The required
retention gates remain failed. The separate stored `frozen_passes` diagnostic
is not the shifted-recovery count.

The unequal-mass endpoint's **only failed bound** is minimum component
eigenvalue ratio `.062121 < .15`; covariance error .402759, mass TV .084570 and
minimum mass ratio .413161 pass. Its 2% component has 48/4096 saved samples,
versus the parent's 81. The minimum mass ratio concerns the underrepresented
13% component, which has 220 samples; it must not be mistaken for the endpoint
failure or the rare component's mass ratio.

Hash-verified saved samples reproduce the rare component's whitened covariance
eigenvalues `[.062121, .209100]`, with minimum-axis direction
`±[.946919, .321471]`. Its target covariance is `.0324 I`, so both axes are
contracted. This uses the exact stored scored samples and evaluator algebra,
with no target sampler or model call. Across the scheduled trajectory, rare
support is present at 14/24 checks versus energy's 16/24; covariance fails
22/24 versus 19/24. Despite fewer minimum-eigenvalue failures (15 versus 19),
full passing checks fall from two to zero. Aggregated high quality does not
establish a rare-density repair.

Width also has a better endpoint: covariance error .464316 versus .716421 and
minimum eigenvalue ratio .649668 versus .180936. All endpoint bounds pass, but
only the last two complete checks pass; covariance still fails 22/24 scheduled
checks. The suffix improves from one to two without meeting five.

The control regressions decide global rejection. Ring retains all 16 modes,
HQ .911621 and mass TV .054688, but component covariance error
`1.087105 > .85` fails at the endpoint and at all 96 scheduled checks. Energy
passed 50/96 with a terminal suffix of 47. Mode hold's endpoint has eight modes
and HQ .904297, both passing; acquisition arrives only at 1100, leaving three
final passing checks instead of five. Energy had 14/24 passes and suffix seven.
Neither regression can be rescued by endpoint selection or extending its budget.

## Native precision improves while density accuracy fails

These are the independent 100,000-sample holdouts, separate from each final
20,000-sample training observation. Each mechanics native run completes all 7,000
updates; mechanics passes none of the five required terminal checks.

| Native task | Precision, energy → mechanics | Mass TV ≤ .06 | Center RMS/σ ≤ .20 | Absolute covariance-trace bias ≤ .10 | Radial KS ≤ .04 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grid | .32545 → .62864 | .10209 FAIL | .26611 FAIL | .74577 FAIL | .26204 FAIL |
| Rotated | .24826 → .60551 | .08264 FAIL | .30766 FAIL | .74066 FAIL | .26227 FAIL |
| Staggered | .29757 → .63868 | .08418 FAIL | .17678 PASS | .78897 FAIL | .26737 FAIL |

Each of the four accuracy bounds fails all 33 postinitial live observations on
every native task; staggered's separate, larger holdout passes centering.
Coverage also fails. Covariance Frobenius RMS values
1.02872/1.05377/.97611 are retained diagnostics: the declared accuracy limits
contain **no independent Frobenius threshold**. Higher precision alone is not
distribution fidelity.

## What the saved mechanics traces establish

All 15 returned traces verify their original hashes, row counts and consecutive
clocks: **37,000 committed updates**, counting Gaussian's 1,000-update smoke
prefix once and its 5,000-update continuation separately. There are 183,518
field evaluations, a mean 3.95995 trials per update, no rejected searches, no
zero applied moves and no accepted secant-bound violations. Final direction
protection changes 1,747 proposals. Replay reports no additional sampling
draws; consumed non-evaluation streams, initialization, prior/sampling laws and
update counts match the energy parent for all 15 tasks.

The intervention is substantial. Median actual movement divided by the raw
anchor proposal is .22877 on Gaussian continuation, .47279 on overlap, .24787
on ring, .12510 on mode hold and about .0613–.0617 on the native tasks. Native
mean trial counts are 4.8754, 5.1154 and 5.0980. Alpha is an interpolation
factor on the corrected protected proposal; it is **not** the actual/base
movement ratio. Raw correction, final protection and rounding also change that
ratio. Alpha one still evaluates the predictor gradient and need not reproduce
ordinary energy training.

Cooling is not globally inactive. Gaussian and overlap traces have thermal
scale one throughout. Mechanics grid/rotated/staggered minima are
.748861/.580873/.693764 with means .985477/.842450/.898907; mode hold reaches
.936036 with mean .998366. The unchanged thermal policy responds to different
trajectories, so native or mode outcomes cannot be attributed to alpha alone.

Rare mass, width, ring, mode hold and all three native tasks have **zero final
direction-protection interventions**. Their remaining failures therefore do not
require protection to mask an accepted raw field. The certificate controls a
combined Euclidean G/prior parameter secant; it does not certify individual
roles, output-space travel, local covariance, rare populations or finite loss
descent. Saved global norms cannot establish which role dominates.

One plausible explanation for the acquisition regressions is changed relative
player kinetics: G/prior movement is corrected and often reduced while D keeps
its ordinary update rule. Equal clocks do not imply equivalent game dynamics,
and D's later trajectory is not held identical. This remains a hypothesis.
The fixed frozen-opponent bilinear control already shows that this G-only
method cannot test or damp joint bilinear circulation. No Lyapunov descent or
nonconservative-game convergence claim follows from the quadratic fixture or
the accepted trained secants.

**Recommendation:** stop this exact global revision. Preserve overlap's scoped
repair, Gaussian improvements and native precision gains as evidence, together
with both control regressions and all failed density gates. Keep the energy
parent as the stronger global partial result in this comparison. A subsequent
joint-dynamics question would need its own falsifiers and full fixed panel;
do not tune alpha constants, relax gates, extend budgets or add another paid
arm to rescue these results.

## Evidence and reproduction

The common execution source is `a03373da3421303b89c99a39898b091e22910134`,
manifest `668914920afda7bb6fc53d9c918aac92a9c9cd5ae582884fb9f068ff1b8249e8`.
Final energy and mechanics readouts, immutable attempt/trace/native-event
hashes and saved-sample identities are bound in [receipt.json](receipt.json).
Live queue bytes are monitoring inputs only and are not bound as permanent
evidence. Parent certification verifies the actual saved receipts/source;
this reducer checks pairing metadata and retained artifact hashes without
rerunning training or graders.

Raw reduction stays at
`/mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/mechanics-analysis.json`;
its hash is in the compact receipt. Per-observation values, JSONL and tensors
stay outside Git. Reduction used 1.56 CPU seconds, within the 180-second
analysis allowance, with no new forwards, updates or samples. Native paid
attempts remain within their original 3,600-second caps; the parent unique
attempt ledger owns total cost. Timing was potentially contended and is not a
speed comparison.

```sh
OMP_NUM_THREADS=1 python reports/forge/first-principles-round2-20261022/mechanics/reduce.py \
  --control /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/full-readouts/bispf-output-energy-v1.json \
  --candidate /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/full-readouts/bispf-secant-extragradient-v1.json \
  --output /tmp/bispf-mechanics-analysis.json \
  --compact-output /tmp/bispf-mechanics-receipt.json
```
