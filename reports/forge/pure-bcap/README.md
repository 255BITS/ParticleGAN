# Pure BCAP baseline draft

Prepared on `research/pure-bcap` from develop commit
`5f2ac0116de93ab697a7d2faa840245430a3af90`, separately from PR #291's
K3P-derived BCAP repair study. No training or sampling has started.

The [candidate card](../../../configs/forge/ideas/bcap-pure-adam-v1.json)
defines a plain-Adam GAN with the public relativistic-pairing logistic loss and
only the fixed BCAP critic penalty. Its coefficient and cap are both 1, applied
every update. LR is .00425, critic multiplier 1, prior multiplier 2, with fixed
Adam betas (0,.999) and constant learning rates.

Critic clipping, A2 latent damping, direct-particle response, EMA anchoring,
prior regularization and additive training input/output noise are disabled.
Scoring uses live weights. The task owns architecture, initialization, particle
count, batch size, duration, learned MoG width and sampling law. MoG kernel noise
is part of the declared prior distribution and remains present.

`Recipe.make_loss()` currently fixes RpGAN logistic loss; `loss` is not an
accepted recipe field. An alternative adversarial loss needs a public API
implementation and a separately declared comparison. BCAP specifies the
penalty, independently of that loss choice.

The legacy kernel dispatch still records `critic_formulation="k3p"` when
`reg_arm="b_cap"`. `optimizer_family="adam"` selects native PyTorch Adam, and
the fixed BCAP kernel uses no K3P blend or EMA-gradient anchor. A zero-update CPU
construction check verified both optimizer types, absent intervention hooks,
fixed BCAP selection and constant rates at updates 0, 200, 600 and 999.

Forge metadata does not add training mechanisms. `requires_capabilities` lists
prerequisites, `claim_contract` declares schedule/scoring/sampling scope, and
`parent` records lineage rather than live parameter inheritance. Active fields
come from public defaults or a preset, candidate recipe overrides and task-owned
resources; execution receipts contain the resolved values.

The decision contract remains **draft**. Before executing, freeze a finite round,
current task/source/runtime bindings, an actual comparable control and a numerical
prediction/falsifier. This baseline removes several interventions together and
cannot isolate the causal contribution of any single removed feature. Preserve
existing gates, failed evidence and the single current leaderboard; no seed-only
trials. Both GPUs remain available under the user's authorization.
