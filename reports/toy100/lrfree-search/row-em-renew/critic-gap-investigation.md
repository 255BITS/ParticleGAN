# Diagnostic follow-up: why does training miss the sampler correction?

## Decision

Retire fresh-reservoir row EM as a proposed scalable solution. Keep it as an
expensive diagnostic calibration. It needed 486–615 fits and roughly 4.4–4.8x
the critic-floor runtime, and each fit evaluated five generator outputs for
every prior row. The all22 replay scored 10 PASS, 4 FAIL, and 8 custom-host
parity errors; on the native100 cases, all three calibrated samplers passed.

The calibration changed public sampling weights and width while the training
state, applied rates, and training random streams matched the critic-floor
baseline exactly. Therefore, the native gains show that a sampling-only
correction can improve these measured distributions; they do not show that
GAN learning improved. Preserve the frozen checkpoints, results, source and
package hashes, fit counts/timings, and audit as evidence. Do not continue
optimizing this sampler.

## Question

Does the trained critic detect the sample-quality gap that the offline
calibration corrects? If it detects the gap but training does not close it,
the issue may lie in how the prior or generator responds. If the critic does
not detect it, the critic signal or objective may be missing relevant
distribution structure.

## Astra review and revised experiment

Astra reviewed the protocol before measurement. It confirmed frozen-critic
scoring is useful as an **endpoint sensitivity** diagnostic, and warned that
absolute critic scores and `sigmoid(D(x))` are meaningless here. The native
critic returns unrestricted scalars; the actual relativistic losses pair real
and fake scores:

\[
L_D=\mathbb E\,\mathrm{softplus}(D(f)-D(r)),\quad
L_G=\mathbb E\,\mathrm{softplus}(D(r)-D(f)).
\]

Score-level mass-weight correlations do not establish a realizable
generator/prior update. The revised work separates endpoint sensitivity from
local update-direction diagnostics.

### Part 1: endpoint sensitivity of the frozen objective

Use the already-frozen `grid100`, `rotated100`, and `staggered100` final states
for the critic-floor baseline and row-EM calibration. The audit establishes
that the trained model states and prior locations match task by task, so this
are identical, but existing sample clouds are independent unless their random
draws match. Generate fresh diagnostic samples with common random numbers and
the same real references. Compare four laws: uniform weights/training width,
calibrated weights/training width, uniform weights/calibrated width, and both
calibrated. Match latent jitter, preprocessing, and noise rules. Report mean
real–fake score difference, rank AUC, actual paired \(L_D\)/\(L_G\), and
checkpoint-conditional confidence intervals. Include real–real and
baseline–baseline null controls. For mass alone, estimate the frozen-objective
change

\[
\Delta_{mass}=\sum_i(w_i^{cal}-1/N)\,
\mathbb E_{r,\epsilon}[\mathrm{softplus}(D(r)-D(f_i(\epsilon)))].
\]

A negative change means this frozen objective favors the mass reweighting.
It does not establish that the particle optimizer can realize it. Width
comparisons identify the objective's preference for that width change.

### Part 2: independent discrepancy power check

Before measuring update directions, test whether a characteristic Gaussian
MMD detects the baseline/calibrated difference. Bandwidth was selected from a
separate 4,096-sample real reference using its median eighth-neighbor distance
(about .02735); the fixed scales were half, one, and two times this value. The
MMD estimates used 4,096 real and generated samples, summarized over sixteen
256-sample blocks, with real–real and baseline–baseline null comparisons.

The check was inconclusive: block confidence intervals were wide, the MMD
differences between baseline and calibrated output overlapped zero, and the
null estimates were noisy. Its directional derivatives therefore cannot
support a claim about which learner update improves distribution fidelity.
Per Astra's recommendation, no replacement discrepancy was selected after
seeing that result, and no MMD directional derivatives are reported. The
useful follow-up stops at Part 1: endpoint sensitivity of the frozen GAN
objective. A final checkpoint cannot explain historical optimization or
whether particle motion can realize a reweighting.

An initial exploratory code run also computed local MMD derivatives, but these
were discarded because MMD did not demonstrate adequate power for this
correction. The reproducible script below does not compute them.

## Results

The corrected script loads the calibrated final checkpoint for each task;
the audit confirms that G, D, and prior positions match the critic-floor
baseline. It generates 20,000 samples per law with shared uniform draws,
latent jitter, and output noise. It uses the same independent real batch for
each law, reports relativistic paired losses and rank separation, and gives
95% intervals across 40 blocks. All weights and widths are post hoc sampling
conditions; no parameters were trained or changed.

| Task | Uniform mass, training width: \(L_G\) | Calibrated mass, training width: \(L_G\) | Mass-only \(\Delta L_G\) (95% CI) | Width-only \(\Delta L_G\) (95% CI) |
|---|---:|---:|---:|---:|
| grid100 | .693567 | .693243 | −.000324 [−.000551, −.000098] | +.00000035 [+.00000031, +.00000039] |
| rotated100 | .696578 | .693439 | −.003139 [−.003681, −.002598] | +.00001060 [+.00000962, +.00001157] |
| staggered100 | .693604 | .693145 | −.000459 [−.000699, −.000218] | +.00000119 [+.00000096, +.00000142] |

Here \(L_G=\mathrm{softplus}(D(r)-D(f))\); a lower value means the frozen
relativistic objective favors that fake law. Reweighting improves this
objective on all three endpoints, with the largest effect on rotated100.
Changing width alone barely changes it and slightly raises the loss in each
case. The score gap \(D(r)-D(f)\) moves from +.000635/.005469/.000622 for
uniform mass to −.000013/−.000777/−.000291 for calibrated mass on
grid/rotated/staggered. The endpoint critic is therefore sensitive to the
mass correction; this does not establish that the online learner can realize
it through particle motion.

The MMD point estimates do not give a consistent result: calibrated mass
lowers MMD on grid100 but raises it on rotated100 and staggered100, with broad
overlapping block intervals and noisy null controls. The full values and
intervals are in [`critic-gap-results.json`](critic-gap-results.json).

## Revisit of the underlying problem

The evidence makes a critic blind spot less likely at these final checkpoints:
the relativistic generator objective assigns a better loss to the corrected
row mixture. It does **not** show that training failed to use an available
gradient. Training samples rows uniformly, so each particle has equal
sampling mass; row EM changes those masses directly after training. The
calibrated change may not be a direction that the existing G/prior optimizer
can express efficiently by moving rows. Distinguishing that representation
limit from historical critic or optimizer behavior needs an independently
validated differentiable discrepancy and, after identifying a specific
mechanism, a matched continuation from a saved checkpoint.

Do not resume work on row EM as the fix. Preserve this endpoint result as
evidence. If pursuing the issue, investigate how a scalable training-time
prior can allocate unequal particle mass under the adversarial objective, or
how particle movement reallocates support under the existing uniform-mass
prior. Any proposed optimizer intervention needs a separate causal test.

The reviewed, reproducible frozen-checkpoint experiment is
[`critic_gap.py`](critic_gap.py). **No training or candidate changes were
made.** Astra reviewed the protocol before measurements and the interpretation
after the MMD power check.

## Frozen artifacts

- [Native100 report](README.md)
- [All22 result matrix](all22-leaderboard.md)
- [Machine-readable all22 summary](all22-summary.json)
- Calibrated states: `/ml2/hypergan/gan-attempts/row-em-renew-20260928/runs/`
- Critic-floor states: `/ml2/hypergan/gan-attempts/combined-h1-h2-20260928/runs/h2-handoff-critic-floor/`
