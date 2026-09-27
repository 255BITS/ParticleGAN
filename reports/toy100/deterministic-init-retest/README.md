# Leaderboard retest with develop initialization

Every new measurement in this directory must use develop's actual deterministic
initialization (`batch_feature_zero`) for fresh networks and recipe-created
learnable priors. Sampling remains stochastic on the existing declared streams.
The user requested this baseline change; merely merging code while restoring an
old random initial model does not satisfy it.

Develop pin: `c720645ecae6b648e9fc6034e9d6b48ccff06ed3` (PR194).
Research merge: `c714f59d`. The API candidate merge is being validated separately.
Earlier measurements remain under `../continuous-api-search/` with their original
sources and initialization. Neither PR is being merged into develop here.

The quick hard screen is the existing small-particle `mode_hold`: 12 particles,
latent dimension4, batch128, 1200 updates, 24 observations and the unchanged
final-five-check rule. It was a useful discriminator under the old initialization;
its new difficulty and results will be measured. No old pass transfers.

Before each run, verify the exact candidate source plus initializer source, the
public factory route, deterministic initial parameters and derived buffers, and
unchanged sample/data stream semantics. Preserve each existing candidate's
algorithm, declared rate/noise policy and eligibility classification. This is a
retest of existing candidates, not a coefficient or random-seed search.

The inventory must include historical research leaderboard entries, exact public
API variants/revisions, released-formulation references, and the previously
measured constant-rate public KA2 control. Entries without an old measurement on
this exact screen receive an explicit missing-comparison label. Missing source
must be reported, never replaced with a guessed implementation.

For every previously failing configuration that passes the new screen, investigate
its other recorded weaknesses under the same new initialization: image/vector
quality, acquisition and retention, repeated target changes, complete API binding,
and checkpoint/budget independence as applicable. Preserve new regressions too.
A screen pass does not establish continuous-learning eligibility or a release
winner. Horizon-dependent configurations keep that separate limitation.

Three external Astra/max sessions may run independent frozen-candidate batches,
with one GPU worker each. Freeze and review the common harness first; finish and
archive old-init work before launching the new batches. Keep all samples random
according to their recorded streams and all scored observations unchanged.
