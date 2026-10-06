# BCAP-pure dualnorm: hyperparameter and Tier 1 gate audit

**Recommendation:** tune player pacing before changing the optimizer rule. The
completed full-dualnorm sweep tested four network step sizes and three momentum
values, but never independently varied D/G pace or the absolute prior step.
The selected starter remains **3/6 recorded Tier 1 passes**. Two-pole is close
to its sustained gate; Gaussian distribution fit and ring component covariance
are substantive remaining failures. No existing result demonstrates that
optimizer tuning alone can clear all six provisional tasks.

This report is a zero-training audit on `report/dualnorm-hyperparameter-audit`,
based on `origin/develop` at
`9cd0544bc43b1ace5bce3b7d67816bef424419c8` after PR #303. It proposes follow-ups;
it does not register, reserve, execute, regrade or promote them. BCAP means
**BCAP-pure** throughout: keep relativistic logistic loss, BCAP coefficient/cap
1, task auxiliaries, architectures, initialization, data/prior/sampling laws,
batch sequence, schedules, update budgets and evaluation unchanged. Repairs to
the existing tasks belong to the separate investigation.

The [current technique leaderboard](../technique-inventory.md) remains the
single leaderboard. The tables below audit parameters and gate conjuncts;
they do not create a second ranking. Read alongside
[EXPERIMENTATION](../../../EXPERIMENTATION.md),
[compiled memory](../EXPERIMENT_MEMORY.md), [original readout](README.md), and
the [machine-readable audit](hyperparameter-audit.json).

## Evidence and starting recipe

The completed screen has 41 whole configurations, including 12 full dualnorm
configurations, at protocol seed 0. Training used commit
`15eb7cb0911905e401bdfcd7e264945a7ea64d97`, source digest
`c5c60a8018e144c447325f3705dd398d04069a91bb294eaaa76d37084ade67cc`.
Its [results](results.json), [analysis](analysis.json),
[starter selection](starter-selection.json), and
[archive inventory](artifact-inventory.json) retain that identity.

The user-selected experimental starter is candidate `78aa66ae…`:

```python
get_recipe("bcap", optimizer_family="dualnorm", lr=.01,
           d_lr_mult=1.5, prior_lr_mult=3., optimizer_momentum=0.)
```

This means etaG=etaE=.01, etaD=.015, and absolute sampled-prior eta=.03,
before the existing schedule multipliers. The pure preset's two LR floors
are 1, so those multipliers are constant at 1; its inherited cosine label does
not imply annealing. The historical .0006/EMA/cosine example is a different
cohort. The public `bcap` preset still defaults to Adam; the explicit dualnorm
starter is an experimental choice among ties, not a qualified new default.

Saved grades belong to their frozen source. The selected starter's seven task
execution/evaluation fingerprints and declared evaluator-source hashes match
the current task cards, including words. Some earlier baseline/search cohorts
have historical word bindings; that difference does not apply to this starter.
Later optimizer/observer/API fixes still mean the merged trainer source differs
from the executed source. Future work must bind its actual source and evaluator,
and cannot assume old evidence is reusable.

## Available parameters versus the search we actually ran

The public [Recipe factory](../../../particlegan/recipes.py) and
[optimizer](../../../particlegan/optim/dualnorm.py) expose seven fields that
affect full dualnorm. Their effects are not seven interchangeable rate knobs.

| Field | Effect in full dualnorm | Completed sweep | Audit conclusion |
| --- | --- | --- | --- |
| `lr` | G/E matrix and vector step size | .003, .01, .03, .1 | Coarse only; .01–.03 remains unresolved |
| `d_lr_mult` | etaD = `lr * d_lr_mult` | Always 1.5 | Important unswept player-pace axis |
| `prior_lr_mult` | etaPrior = `lr * prior_lr_mult`, sampled rows only | 10, 3, 1, .3 paired with the four rates | Absolute etaPrior was always .03; this was not a prior sweep |
| `optimizer_momentum` | One shared mu for D/G/E; none on prior | 0, .5, .9 at all four rates | Explored only at fixed D/G and absolute prior pace |
| `eps` | G/E vector denominator and whole-matrix gradient/momentum skip threshold; default for D/prior | 1e-8 | Fixed; meaningful damping only when comparable to gradient norms |
| `d_eps` | Optional D-specific version of `eps` | No override | Available after develop integration, untested |
| `prior_eps` | Optional epsilon in sampled-row normalization | No override | Available after develop integration, untested |

The [zero-momentum grid](../../../configs/forge/searches/bcap-optim-dualnorm-zero-tier1-v1.json)
and [positive-momentum grid](../../../configs/forge/searches/bcap-optim-dualnorm-momentum-tier1-v1.json)
prove the paired rate assignments. The three prior-only isolation rates did
**not** test different prior paces inside full dualnorm: their other players
used Adam. D-only dualnorm similarly does not substitute for a D/G ratio sweep
in the all-dualnorm arm.

`betas`, `d_betas`, `prior_betas` and `amsgrad` do not enter full-dualnorm
updates, even though common APIs carry or validate them. `optimizer_adam_lr`
and any non-None `beta2_end` are rejected for full dualnorm. TensorFlow-style
Adam is irrelevant here. Sweeping these fields would waste configurations.
Schedule choices remain fixed by this experiment's scope.

There is no public independent E rate, D/G momentum split, intermediate mu
such as .1/.25, or matrix/vector/output-head multiplier. Low-level groups can
express different rates/epsilons, but a Forge comparison needs a declared
global Recipe extension, rather than task-specific optimizer construction.
Mu=0 versus positive mu is a structural category in Forge; keep their existing
separate base ideas when declaring searches.

The aspect-ratio multiplier `sqrt(max(1, fan_out/fan_in))`, polar rule, and
absence of prior momentum are part of the declared technique. SVD below or at
side length 1024, the larger-matrix 30-iteration Newton–Schulz attempt and
1e-3 residual fallback are implementation safeguards, not useful quality axes.
All polar-processed Tier 1 matrices are below that cutoff. Nonzero weight decay and
the optional fused/foreach/maximize modes are unsupported here. A softened
polar rule, decay or trust bound would be a new optimizer variant.

**Host applicability matters.** Ordinary two-pole directly optimizes generated
coordinates through the generator matrix rule; it has no sampled latent prior.
Changing `prior_lr_mult` or `prior_eps` cannot fix it. Unused-token likewise has
no sampled prior. See the [field boundaries](../../../experiments/forge/boundaries.py).
G/E share a pace in the joint-word and AE hosts; sampled prior rows use their
own pace. Fixed stored-critic/zero-coordinate initialization in two-pole
remains its explicitly declared cohort, not a substituted initialization.

## Exact Tier 1 thresholds and starter results

Each of these six required tasks records 24 observations and needs **at least
five consecutive joint passes ending at its final observation**. An earlier
passing streak or a passing final point alone is insufficient. Finite-state,
intended-update, mechanism and RNG guards also apply; the starter's recorded
guards pass. All metrics below are saved live results under each task's law,
including AE's explicit scheduled-noise exception; they are not EMA grades.

### Gaussian acquisition: FAIL, 1/24 joint passes, terminal suffix 0

[Task](../../../configs/forge/tasks/gaussian1d_acquisition.json): 1,000 updates,
4096 evaluation samples, learned MoG sigma=.025, target N(2, .5²).

| Metric | Threshold | Final value | Final conjunct |
| --- | --- | --- | --- |
| `sample_count` | >=4096 | 4096 | PASS |
| `finite_fraction` | ==1 | 1 | PASS |
| `mean_error_sigma` | <=.2 | .035695893 | PASS |
| `std_ratio` | >=.8 and <=1.2 | 1.231132560 | **FAIL upper bound** |
| `cdf_ks` | <=.05 | .083278201 | **FAIL** |

Width exceeds the upper bound by .031132560; KS exceeds its bound by
.033278201 (1.6656 times the allowed value). The only joint passing observation
was step 125. This is not a location failure. Matching mean/variance alone
cannot satisfy the [analytic CDF scorer](../../../benchmarks/toy_audit/gaussian1d_quality.py).

### Two-pole: FAIL, 14/24 joint passes, terminal suffix 4

[Task](../../../configs/forge/tasks/two_pole.json): 80 updates; fixed stored
critic and zero direct coordinates.

| Metric | Threshold | Final value | Final conjunct |
| --- | --- | --- | --- |
| `mean_abs` | >=.3 | .791628540 | PASS |
| `grad_med` | <=1 | .984050214 | PASS |

The last five observed steps are 67, 70, 74, 77 and 80. Step 67 has
`grad_med=1.000958920`, exceeding the limit by .000958920; the remaining four
pass. Terminal slope has only .015949786 of headroom. This is a critic-slope
stability failure, not insufficient endpoint travel. Lower D pace is a
testable hypothesis, not a guarantee: the GAN trajectory can change its slope
non-monotonically. Extra updates or a relaxed suffix would change the task.

### Ring acquisition: FAIL, 0/24 joint passes, terminal suffix 0

[Task](../../../configs/forge/tasks/ring16_acquisition.json): 400 updates,
16 two-dimensional Gaussians on radius 3, sigma=.1, learned MoG prior sigma=.025.

| Metric | Threshold | Final value | Final conjunct |
| --- | --- | --- | --- |
| `sample_count` | >=4096 | 4096 | PASS |
| `modes` | >=16 | 16 | PASS |
| `mass_tv` | <=.15 | .082763672 | PASS |
| `hq` | >=.85 | .930175781 | PASS |
| `component_covariance_error` | <=.85 | 13.164045306 | **FAIL** |
| `component_min_eigen_ratio` | >=.15 | .172928542 | PASS |

The failed covariance metric is **15.487 times** its allowed value, exceeding
the limit by 12.314045306. In the
[scorer](../../../benchmarks/transfer_suite/vector_tasks.py), each sample is
assigned to its nearest target mean; the metric averages the 16 relative
Frobenius covariance errors using *all* samples assigned to each component.
It is neither the overall covariance error (.056550268) nor a core-only error.

Saved core covariance error is .567985335 versus full error 13.164045306;
maximum per-component spill is .285171092. This supports investigating tails
and bridges between modes, rather than treating HQ as proof of shape quality.
Core minimum eigen ratio .172928542 is already above the Tier 1 lower bound,
but below Adam's .241633520 diagnostic. Core/resolved metrics cannot replace
this task's full covariance gate or introduce finite-atom exemptions.

### Unused-token hold: PASS, terminal suffix 16

[Task](../../../configs/forge/tasks/unused_token_hold.json): 200 updates.
`unused_hold=.989279812 >= .85` and `concept_move=.991806589 >= .85`.
The margins are .139279812 and .141806589. Retain both when tuning pace.

### AE/GAN hold: PASS, terminal suffix 22

[Task](../../../configs/forge/tasks/ae_gan_hold.json): 250 updates.
`recon_mse=.002923731 <= .05` and `hold=.006327168 <= .35`.
Their upper-bound headroom is .047076269 and .343672832. This is the frozen
behavioral reconstruction/hold question, not a new auxiliary-loss ablation.

### Five-word joint acquisition: recorded PASS, terminal suffix 6

[Task](../../../configs/forge/tasks/five_word_joint_acquisition.json): 20,001
updates, 5 learned two-dimensional particle rows, fixed G/E/joint-D geometry.

| Metric | Threshold | Final value | Final conjunct |
| --- | --- | --- | --- |
| `sample_count` | >=1024 | 1024 | PASS |
| `quality_fraction` | >=.95 | 1 | PASS |
| `modes` | ==5 | 5 | PASS |
| `mass_tv` | <=.1 | .018945313 | PASS |
| `reconstruction_exact` | ==1 | 1 | PASS |
| `minimum_reconstruction_token_probability` | >=.9 | 1 | PASS |

The final passing streak starts at step 15,835; 18/24 observations pass.
Suffix 6 gives only one extra passing observation beyond the requirement.
Preserving this inverse/coverage success is essential; increasing the base
rate to .03 loses it. The starter matches the current word task/evaluator
binding; a future campaign must additionally bind its actual trainer source.

The separate `clockfree_audit_measurement_v1` diagnostic passes all four parity
comparisons (step label, horizon, evaluation cadence, restart). It is outside
the required denominator and cannot turn 3/6 into 4/7 or unlock Tier 2.

## What neighboring configurations tell us

These are conditional sweep observations, not another selected leaderboard.
Every row retains D/G=1.5 and absolute etaPrior=.03.

| etaG | mu | Ring HQ | Ring full component covariance error | Two-pole suffix | Word suffix |
| --- | --- | --- | --- | --- | --- |
| .003 | 0 | .660156 | 56.482052 | 0 | 7 |
| .01 | 0 | .930176 | 13.164045 | 4 | 6 |
| .03 | 0 | .818359 | .509620 | 21 | 0 |
| .1 | 0 | .156250 | 31.789172 | 22 | 0 |
| .01 | .5 | .943359 | 5.069370 | 0 | 0 |
| .01 | .9 | .245605 | 66.734020 | 15 | 0 |

At .03/mu0 the ring's **only failed final conjunct is HQ**; its modes, mass,
covariance and minimum eigen ratio pass. At .01/mu0 only its full component
covariance fails. Intermediate rates are therefore a useful missing contrast,
but would still need five terminal joint passes and must retain word success.
None of all 41 tested configurations passes the full Gaussian or ring task.
Mu0 reaches 3/6; positive-momentum full dualnorm reaches at most 2/6. That
disfavors starting with momentum, without proving every positively smoothed
player configuration must fail.

The separately published
[BCAP repair branch](https://github.com/255BITS/ParticleGAN/blob/d5a8c73a1a60483900d736ed73da746446cc300e/reports/forge/bcap-tier1-repair/README.md)
is a different cohort: its 12 Adam configurations and two duration diagnostics
use a seven-required-task view, including a schedule audit. Its best 4/7
cannot be pooled with this study's 3/6 plus separate clock diagnostic. Preserve
that investigation and avoid repeating its Adam/penalty/horizon work here.

Saved, hash-verified prior diagnostics show nearly constant sampled-row
displacement .03 at the logged steps in Gaussian, ring and word; measured
unsampled raw-row displacement is zero. This makes prior pace worth isolating.
Aggregate table gradient norms are not per-row norm distributions, so they
cannot by themselves select a meaningful `prior_eps`.

The original input-gradient observer had a phase-routing defect. Its real/fake
input-gradient traces are excluded. Actual updates, weights and matrix spectral
products remain usable. The latter excludes Fourier maps/nonlinearities and
is not the whole critic's Lipschitz bound. No universal spectral smoothing or
width/depth transfer success follows from these receipts.

## Proposed path toward six complete passes

**First: independent player pacing with the existing implementation.** Hold
etaG=.01 and mu=0. Compare D/G in {1, 1.5, 2} and absolute etaPrior in
{.01, .03, .1}; encode the prior as `prior_lr_mult=etaPrior/.01`. The center
configuration is the existing starter, leaving eight substantive new recipes.
The slower prior tests reduced fixed-step transport; the faster prior is a
directional control. D/G=1 tests lower critic pace near two-pole's slope
boundary; D/G=2 tests whether stronger discriminator tracking improves law
fit. Their opposite outcomes can distinguish pace effects from simply lowering
every rate. Each recipe must run all six independent Tier 1 peers.

Falsifiers: a slower prior does not reduce Gaussian KS/ring covariance while
preserving coverage and word reconstruction; lower D pace does not produce
five terminal two-pole passes; improvements disappear through a hold/word
regression. Do not attribute two-pole changes to prior pace, which is inactive
there. Do not merge each task's best settings into a fictional whole winner.

**Second: resolve the .01–.03 rate tradeoff.** Use round 1's whole-recipe
required-PASS-count/configuration-hash winner. Proceed only if it reaches
at least 4/6 required passes and still passes unused-token, AE hold and words;
otherwise stop this proposed sequence. This criterion requires a new complete
sustained gate, rather than an undefined endpoint improvement. Freeze that
winner's D/G and absolute prior pace, then test etaG in {.012, .016, .022} at mu=0. Recompute
`prior_lr_mult` at each rate to keep the chosen absolute prior step fixed.
The hypothesis is that a rate between the two endpoint tradeoffs can meet
ring HQ and full covariance simultaneously without losing the word task.
Reject it if no complete terminal window meets both, or word/hold performance
regresses. This staged design prioritizes gaps; it is not an exhaustive
Cartesian search of all rate interactions.

**Third, only if pace tuning is insufficient: separate causes before expanding
the optimizer.** Use saved samples and valid update traces to separate ring
core width, centroid error and distant assigned tails, and examine Gaussian
CDF residuals and late update magnitudes. No new training is needed if the
certified saved states/samples contain those measurements. Do not use broken
input-gradient traces. Epsilon sensitivity is available now, but first measure
per-row and per-vector norms: larger epsilon attenuates vectors/rows, while a
matrix still takes a full polar step until the entire gradient is skipped.
It cannot smoothly attenuate small singular directions in a nonzero matrix.

If those diagnostics support a structural optimizer change, the most plausible
declared variants are (a) D-only directional momentum with G/E mu0 and no prior
momentum, (b) a separate global bias/vector pace, and (c) a softened polar factor
that suppresses weak singular directions instead of mapping all of them to
unit size. Each requires its own public configuration/support and matched
step-size search. Positive global momentum already failed to improve the
whole gate count, so these are lower-priority hypotheses, not known fixes.
Particle momentum, decay and adaptive controllers would change the current
definition; do not disguise them as already searched hyperparameters.

The public rate fields accept values beyond the first proposed grid. D/G
below 1 (for example .5/.75) is a plausible further critic-slope contrast;
absolute prior steps below .01 are also available. The first round prioritizes
the originally requested ratios/prior steps. An improving edge would motivate
a separately bounded extension, not establish an optimum or trigger automatic
widening. The audit therefore finds meaningful omissions, rather than claiming
that every likely value has been covered.

## Finite follow-up contract

These are proposed ceilings, **not an active or reserved campaign**. The
existing six required tasks reserve 2,220 seconds per whole recipe; retaining
the 300-second clock diagnostic makes 2,520. Eight new pacing recipes therefore
need at most **20,160 seconds**. If a matched current-source starter control is
scientifically necessary because exact reuse fails, explicitly justify and
reserve one more recipe: **22,680 seconds for nine**. Do not rerun the center
merely because develop was merged. The conditional three-rate refinement
would require a separate **7,560-second** declaration after reviewing round 1.
The proposed sequence totals **27,720 seconds for 11 new recipes**, or
**30,240 seconds including one justified source-control recipe**. These are
worst-case reservations including the clock diagnostic, not expected runtimes.
No numeric retries, edge widening or structural expansion is automatic.

Before execution, freeze the study ID, actual recipe/source/runtime bindings,
protocol seed 0, shared public deterministic initializer with task exceptions,
named streams, complete current-tier policy, objective and stopping rule.
Keep Forge's required-PASS-count/configuration-hash selection objective;
report gate margins as diagnostics without silently changing the tie-break.
Every compared setting is one global recipe across tasks. Missing or failed
cells cannot be pooled away. Tier 2 stays blocked until all six required gates
pass under one compatible candidate; that remains a provisional screen, not
scientific default adoption. Preserve historical failures and report new ones
under their actual source, particularly the word evaluator binding.

Put execution stdout/events/checkpoints in an ignored new queue, with one
tail-able driver log and the existing event stream. Commit only compact
metrics/provenance and actual-training GIFs. Update the existing leaderboard
only when new measured evidence warrants it. This audit changes no code,
task, result, selection, leaderboard or training budget.

The compact audit independently verifies seven certificate bindings, recomputes
the six required terminal suffixes, checks current task/evaluator fingerprints
and source hashes, and verifies the three diagnostic trace hashes. It stores
final metrics and selected failure examples; bulk observations remain in the
original ignored queue/archive. The local audit log can be read with
`tail -F runs/forge/dualnorm-hyperparameter-audit/audit.log`.
