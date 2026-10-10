# Prospective mean transport prototype

Status: private CPU prototype selected; no production implementation or next
quality candidate selected. Source and numerical inputs must be frozen before
measurement. No threshold or parameter search is authorized.

## Evidence and scope

RA9 passes the original final CUDA toy and the complete grid holdout, but all
five terminal grid clouds fail center RMS. Their other original gates pass.
The saved clean cloud follows EMA anchors closely. A fixed learned chart has
100 pure observed groups, and the existing paired clouds have no clean/noisy
group switches. Split-reference mean noise is substantial; the aggregate
direction is reproducible across most groups. This supports testing a mean
repair, without claiming every local direction is certain.

The final affine generator and latent table largely compensate each other.
That decomposition does not establish an adverse generator step, so the
generator-only guard remains reserved. The proposed reuse of already
certified copy slots was rejected because the final reaction has none.
Direct covariance correction is unsupported by the saved covariance budgets.

## Fixed prototype

Use even-fit real groups to center and scale the existing learned projected
features. Clip feature vectors to norm at most `sqrt(effective_rank / Q)`.
Construct unit mean-residual directions from current EMA anchors and the
even reference only. Use the scalar
`X_j = u[group_j] · (clipped_features(odd_real_j) - EMA_mean[group_j])`
with unit-or-zero directions from even mean minus EMA mean, and the known
range `4 * sqrt(effective_rank / Q)`. Its empirical variance enters one fixed
empirical Bernstein formula. This witness naturally weights the odd real
mixture; the action objective separately uses even empirical group masses.
Do not treat these as the same weighting or select groups from the score.
Insufficient rows, missing groups, zero scales or nonfinite values veto the
prototype conservatively.

One added global hypothesis changes the common correction family to `3K+3`.
Every existing count test must use that correction too. The shared ordinary
action budget stays at five percent; population stationarity remains a
separate law with its original participation threshold and row-reset rules.

If the aggregate trigger is positive and ordinary budget remains, test
mean-directed parent/child pairs with identical cell and inside/out category
in both current FAST and EMA views and the same learned real group. Parents
and children must also be supported and inside in both views. This conservative
restriction was selected during source review before numerical execution.
Parents and children must be unique and no source row may be deleted. Preview the
existing paired bounded jitter; retain only actual EMA mean progress with
supported inside category retention. Mutate scratch copies only. No model-weight, fixture,
quality gate, noise law/floor or serving-rule changes are part of this test.

The fixed cases are the saved final RA9 grid and toy. Test whether the trigger
and legal paired transports have useful capacity on grid and behave
conservatively on toy. Record a negative result without retuning.

## Limits and next gate

The current D, EMA and FIFO share training data. An even/odd FIFO split is not
an independent prospective null sample. The bounded concentration expression
is a conditional model calculation and an empirical negative-evidence
trigger here; it establishes neither population stationarity nor distribution
equivalence. A global discrepancy also does not certify individual local
shifts. Explicit preview checks and the original quality evaluations remain
required.

Independent mathematical and integration reviews precede the prototype.
Only useful legal progress can motivate a separately frozen production law
with typed state and replay contracts. Any resulting candidate must rerun
the unchanged full CUDA toy and canonical grid, including all five terminal
clouds and the independent holdout. Required state, replay and portability
checks follow a package passing both. No current package qualifies.
