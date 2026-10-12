# Prospective spectral optimizer comparison

**A bounded full-versus-critic-only comparison is justified as a different
optimizer question, with limited evidence of success.** After the current
direction/floor campaign closes, compare exactly two global family-only changes
from the original constant m000 parent: `optimizer_family='dualnorm'` and
`optimizer_family='dualnorm_D_only'`. Each keeps LR .0085, paired R1/R2 .1 on
every update, declared betas (0,.99), role multipliers 1, epsilon 1e-8, learned
prior, zero extra regularization, zero momentum/smoothing and constant schedules.
Keep `optimizer_adam_lr`, `d_eps` and `prior_eps` null. No additional rate point,
new declaration, admission or training is created by this note.

The [receipt](spectral-note.json) contains certified current negative results,
saved-gradient spectra and Jacobian geometry, source hashes, prior BCAP evidence
and **14 passing prospective public preflight bindings** for the original six
ordinary tasks plus clock. Existing failures remain provisional under their
original source; central closure, registration, budgets and source freeze are
still required before any execution.

## What Ada failure changes

These are individually terminal certified Ring16 queue results, with final
whole-campaign collection still pending when this note was written:

| Current delta | Component covariance error | HQ | Modes | Minimum eigen ratio | Original gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Adam critic epsilon 1e-4 | 2.635454 | .846191 | 16 | .297133 | FAIL |
| AdaNSGDA, original epsilon | 4.570227 | .432861 | 14 | .697502 | FAIL |
| AdaNSGDA, critic epsilon 1e-4 | 7.055577 | .258057 | 6 | 1.129127 | FAIL |

Removing within-tensor Adam rotation while retaining its tensor norm did not
repair density. That result weakens the original direction-only hypothesis.
It does not identify the cause or demonstrate that a spectral alternative will
work. The spectral alternative also changes magnitude in its own step units.

## Distinct spectral mechanism

Ada multiplies an entire raw gradient matrix by one scalar, preserving its
singular-value ratios. The existing zero-momentum polar rule replaces retained
singular values by one. Its matrix update operator norm is
`lr * sqrt(max(1,out/in))`, within numerical rounding. The actual helper discards
singular values at or below `max(shape) * dtype_epsilon * largest_singular_value`.
Biases use an L2-normalized gradient. These are update rules; they impose no
constraint on existing weight matrices or the learned network Jacobian.

The same saved m000 Ring16 gradients show a concrete remaining distinction:

| Hidden 64×64 matrix | Leading gradient squared-norm fraction | Retained gradient rank | Native update operator norm | Ada operator norm | Polar operator norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | .593636 | 44 | .073231 | .079755 | .008500 |
| D | .887084 | 23 | .265644 | .268424 | .008500 |

Ada retains those leading fractions. Polar spreads its squared norm equally
over the 44/23 retained directions, giving leading fractions 1/44 and 1/23.
This tests whether an update dominated by a few directions contributes to
uneven density adaptation. It can also amplify weak or noisy singular directions
and regress coverage; the retained-gradient condition ratios are approximately
118,713 for G and 41,478 for D. No moment or gradient tensor is committed.

The saved generator's *own* Jacobian at its 256 learned prior means has median
condition number **1.665**, maximum **3.089**. Its original-width local noise has
mean tangent/radial target-variance ratios **2.315/.848**. This describes moderate
anisotropy with tangent stretch, rather than a severely ill-conditioned learned
Jacobian. A polar update could alter how that geometry evolves; it does not
directly bound or equalize it. Local linear noise is descriptive at saved row
centers, not an exact covariance decomposition or a task gate.

## Role ownership and controls

Full DualNorm uses polar G/E and D matrix updates and normalized vector updates.
Its learned prior uses `-lr * g_row/(||g_row||+eps)` on each **actual sampled
generator-side unique row**, with no prior momentum. Original latent index and
width-noise streams, prior initialization/capacity and sampling distributions
are held fixed. Unsampled rows remain fixed even when auxiliary losses produce
their gradients. This prior update-law change is explicit in the family delta.

`dualnorm_D_only` uses the same polar D law, retaining native Adam for G/E and
prior at .0085 because `optimizer_adam_lr` stays null. It retains the parent's
moment adaptation on those roles. In full DualNorm, their declared beta2 .99 is
inactive metadata; the raw-gradient polar/row laws have no Adam second moments.
The pair holds the critic law fixed while jointly changing G/E **and** prior
laws. A difference between the arms cannot isolate generator normalization from
prior normalization; their future critic states can also diverge through feedback.

The word host uses its existing dense G/E/joint-D matrices, original joint and
inverse objectives and five-row learned table. `policy.generate(..., rows=rows)`
records sampled IDs only during the generator phase. Five rows are neither
silently enumerated nor assumed all visited. Behavioral AE sampling likewise
supplies its actual G-side row IDs. Token parameters and all ordinary network
weights are supported vectors/matrices; no convolution adaptation is required.
Both families retain checkpointed optimizer/row-ownership state and the same
clock host, whose actual gate must still be measured.

## Exact direct-pole bound

The original twelve zero-initialized direct coordinates are generator-owned,
even though the behavioral fixture annotates their input group `forge_role=prior`.
Full DualNorm treats this 12×1 tensor as a matrix. Its nonzero step L2 is
`.0085 * sqrt(12) = .02944486`. Since `L1/12 <= L2/sqrt(12)`, mean absolute
coordinate displacement is at most **.0085 per update** and **.68 after 80**.
The earliest update at which this upper bound permits .3 is 36. At the last five
original observations, updates **67/70/74/77/80**, bounds are
**.5695/.5950/.6290/.6545/.6800**. Thus no fixed travel ceiling rules out the .3
sustained gate; the bound proves neither outward motion nor the required critic
gradient median ≤1. D-only retains the original native Adam coordinate rule,
while its changed critic can still change the trained trajectory.

## Prior negatives and stopping

The original [unsmoothed full BCAP screen](../../configuration-search/bcap-optim-dualnorm-zero-tier1-v1.json)
failed Ring16 and Gaussian at all four points. Its .01 recipe acquired words but
failed pole. The [BCAP critic-only screen](../../configuration-search/bcap-optim-dualnorm-d-only-tier1-v1.json)
also failed Ring16, Gaussian and words at all four points; it used momentum .5,
pinned Adam LR .00425, D multiplier 1.5 and prior multiplier 2. These are negative
evidence under source `c5c60a8018e144c447325f3705dd398d04069a91bb294eaaa76d37084ade67cc`,
with different formulation and original task budgets. Later BCAP pacing improved
its whole recipe to 4/6 while still failing both density tasks; its later selected
smoothing repair is another explicit trainer delta. None supports transferring
a pass or choosing a new R1/R2 rate/smoothing grid.

The proposed pair is one fixed-rate question, with only `optimizer_family`
changed in each arm from m000. If centrally registered, it requires two complete
2,520-second task allowances, 14 new jobs and all original gates; cache the exact
parent rather than rerunning it. Any original gate failure falsifies a complete
Tier 1 repair. Better spectral diagnostics, a better endpoint or a per-task
mixture grants no qualification. Close the current campaign first; retain every
failure and stop this finite comparison after its declared jobs.

All saved arithmetic uses source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`.
The receipt verifies its 1,214 current file hashes, exact parent checkpoint and
current Ring16 certificates. No networks, priors, trainers or optimizers were
constructed, no module forward was called, and no update, sampling or rescoring
occurred. Only Recipe's parameter-free inactive Coulomb option validator was
allowed during public preflight. Python/Torch RNG stayed unchanged. CPU float32
SVD rounding and cutoff sensitivity limit transfer of these conditional spectra
to accelerator trajectories.
