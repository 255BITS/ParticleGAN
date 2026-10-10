# Initial-position proximal prior spring

`Recipe.from_preset("bcap").replace(prior_proximal_strength=1)` adds a final
restoring update to the BCAP leader. The default is zero and retains the original
optimizer types, checkpoint packets, initial tensors and consumed RNG streams.

For each unique prior row owned by the generator update, first complete the
ordinary BCAP optimizer step, including direction blend. Then apply

```text
z_final = (z_base + lr_prior * lambda * z_initial) / (1 + lr_prior * lambda)
lambda = Recipe.prior_proximal_strength
```

The implementation uses the equivalent `lerp` fraction
`lr_prior * lambda / (1 + lr_prior * lambda)`. A gradient-space spring can lose
its magnitude under row normalization. This final position update preserves
the relationship between displacement and restoring motion: twice the distance
from the initial row produces twice the restoring displacement. It consumes
no additional draws, target data, labels or evaluation information. Initial
capacity, width, masses and sampling remain task-owned.

The motivation is the [scalar saved-state diagnosis](../reports/forge/gaussian1d-diagnosis/README.md):
an unchanged architecture/prior admits a good scalar fit, but learned prior
locations move while acquired quality degrades. That older cohort supports a
stability hypothesis, not a current BCAP repair. The measured reference for new
comparisons is the [BCAP direction-blend leader](../reports/forge/bcap-default-baseline/README.md),
with 6/6 Tier 1 and 9/21 Tier 2 passes. The spring is an additional recipe delta.

Only a sampled latent-prior group receives restoration, and only if its current
gradient was consumed. Unsampled rows remain fixed even with dense auxiliary
gradients; duplicates count once. Zero scheduled rate, absent gradients and
empty ownership apply no force. Frozen priors and direct generated coordinates
receive none. Enumerated behavioral hosts use their existing explicit full-row
ownership. The optimizer captures independent initial references after the
public initializer; create optimizers after initializing and moving tables.
Checkpoints preserve references, strength, update ordering, pending ownership
and aggregate restoration counters, including restoration of an arm's own
Gaussian or word prefix. Loading a trained checkpoint never rebases its anchor
to the trained locations. Adding new prior groups later is rejected.

The supported update law is zero-momentum full DualNorm with geometry `none`
or `direction_blend`, without adaptive prior policies. Other compositions are
rejected at the recipe boundary. All Forge hosts using the shared public
generator optimizer inherit the rule; no private training loop is introduced.
Forge treats strength as a tunable recipe hyperparameter and treats enabling
the spring as a structural mechanism change. A positive search cannot include
zero as a same-technique configuration; use the separate leader control.

Restoration occurs **after** direction blend. Existing direction-blend
derivative diagnostics describe its base displacement; the final spring can
change a protected objective's derivative. The mechanism does not promise
finite-step descent or preservation of those diagnostics. An anchor can also
prevent useful latent relocation, block adaptation or shift necessary work
onto the generator. Latent stability alone cannot establish output density or
100-Gaussian coverage.

The initial bounded search bracket is one global recipe per strength:

| Strength | Restoration fraction at leader prior LR .03 | Purpose |
| --- | --- | --- |
| .1 | .002991 | Weak restoring bias |
| 1 | .029126 | Moderate restoring bias |
| 10 | .230769 | Strong restraint approaching a fixed input law |

The hypothesis is falsified as a useful global repair if no eligible setting
converts a full sustained Tier 2 failure to PASS, or if it loses required Tier 1
passes. Reduced prior drift or an isolated better endpoint is insufficient.
Compare each setting with the complete leader on the same executed source,
protocol seed0, initializer, task data/architecture/prior, budgets and cadence;
own-prefix continuation remains required. All six runnable Tier 1 tasks must
finish, and a failure blocks ordinary Tier 2. The declared search budget and
campaign are registered separately before execution. No Forge run is launched
by this feature PR.

The [software illustration](../reports/forge/prior-proximal-software/README.md)
records numerical software gates and an actual-training GIF under the
[public-API toy contract](../reports/toy_audit/api_contract/README.md). Its narrow
64-update CPU cohort supplies implementation evidence, not Tier 2 credit.
