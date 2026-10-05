# Finite search spaces and the Halloween transfer

Forge can now compile a finite dictionary of choices into ordinary bounded
searches. Categories refer to declared structural candidates; each numerical
grid stays within its candidate's technique signature. Compilation validates
the public API, field activity, matched source/runtime/protocol/view, complete
candidate reservations and the aggregate campaign ceiling. It launches no
training and selects no public default.

The [example declaration](../configs/forge/search-spaces/halloween-transfer-v1.json)
compares the existing Pure BCAP least-squares base with the new
[Halloween optimizer/loss transfer](../configs/forge/ideas/halloween-optimizer-loss-v1.json).
It samples four complete configurations from a finite log-scale LR domain,
under one campaign with a 10,080-second ceiling and 2,520 seconds per candidate.
Those reservations cover this view's six required Tier 1 tasks and separate
clock diagnostic. The example is unexecuted: existing results remain bound to
their original sources and recipes.

```sh
python -m experiments.forge search compile \
  configs/forge/search-spaces/halloween-transfer-v1.json \
  --output runs/forge/halloween-transfer-v1.json
python -m experiments.forge search plan runs/forge/halloween-transfer-v1.json
```

An explicit later `search enqueue`, `search run` or `search report` accepts the
same compiled manifest. All categories freeze before any request is admitted.
Admission and retries use ordinary Forge registration, recovery, evidence reuse
and shared budgets. Run finishes independent peers in the current tier before
required failures block higher tiers. Report selects one whole configuration
across the matched sampled candidates; unsampled categories remain visible.
There is no separate leaderboard or per-task recipe selection.

Execution logs use the existing queue's `events.jsonl`, campaign
`progress.jsonl` and attempt `run.log`. For this example:

```sh
python -m experiments.forge logs --follow --campaign halloween-transfer-space-v1
```

## Tagged values and conditional categories

Each `parameters` value has an explicit tag:

| Tag | Example | Meaning |
| --- | --- | --- |
| `literal` | `{"kind":"literal","value":[0,0.9]}` | A fixed pair, not two alternatives |
| `choice` | `{"kind":"choice","values":[0.001,0.002]}` | Pick one finite alternative |
| `logspace` | `{"kind":"logspace","low":0.0001,"high":0.01,"count":8}` | Eight fixed logarithmic choices including endpoints |

A grouped axis can choose dictionaries assigning the same numerical fields,
for example `rates: {"kind":"choice","values":[{"lr":0.001,"d_lr_mult":1},
{"lr":0.002,"d_lr_mult":0.5}]}`. Fields must not overlap another axis.
Each roster member supplies `base_candidate`, `trainer_family` and optional
`parameters`. Its conditional parameters replace the common parameters.
Unsupported optimizer/loss strings are not dynamically imported; they need
registered structural candidates and supported public implementations.

The sampler indexes the finite population without expanding its Cartesian
product. It draws without replacement using an isolated, versioned Python RNG
derived from protocol seed 0, the space ID and the domains. Hypothesis prose
and budget edits do not perturb that RNG. The manifest saves initial/final RNG
states, draw indices/settings, category identities, source bindings and its
content hash. Sampling admits at most 256 complete configurations. A changed
manifest, source or declaration requires a new study identity; there is no
adaptive resampling during training or recovery.

## Public Recipe settings

G and E retain shared `betas` / `eps`. D uses `d_betas` / `d_eps` when supplied,
otherwise inherits the shared values. Learned prior groups use `prior_betas`
and `prior_eps` when supplied, otherwise inherit G's shared settings. Public
factory kwargs and explicit parameter-group options remain caller overrides.
Recipe role fields pass through Forge's effective field receipts and adapters;
hidden role kwargs are unnecessary for a search.

`beta2_end`, where enabled for PyTorch Adam, remains one terminal value for all
roles, interpolating from each group's own initial beta2. New role moments
cannot silently cross a zero/positive activation boundary in a numerical grid.
Prior-only fields are inactive on tasks without a learned latent table.

`loss="least_squares"` accepts `loss_labels=(fake, real, generator)`. The default
is `(0,1,1)`; `(-1,1,1)` represents the inspected Halloween scalar objective.
Joint generator/encoder hosts include the reversed real-stream target. Labels
are technique fields and require a structural candidate, not a numeric axis.

`adam_variant="tensorflow_v1"` selects the inspected dense legacy update law:
epsilon is added before second-moment bias correction. Each optimizer saves its
application clock, per-group beta powers and parameter moments, including
skipped-gradient clock behavior. Sparse/complex gradients, AMSGrad, weight
decay, changing betas and accelerated/differentiable modes are explicitly
unsupported. Native `adam_variant="pytorch"` and existing formulation optimizers
retain their defaults. A backend change is a structural candidate.

`lr_schedule` selects `cosine` (existing behavior), `constant` or `exponential`.
Exponential uses `lr_decay_rate`, `lr_decay_steps` and `lr_decay_staircase`,
counting completed whole training updates. Sampling does not advance it, and
shortened execution does not rescale it. It applies the same multiplier to
network and prior roles. It does not infer legacy TensorFlow's shared per-role
application clock. Noncosine schedules reject controller/horizon transitions;
inactive cosine floor/timing settings are rejected as search axes.

```python
from particlegan import get_recipe

recipe = get_recipe("halloween")
opt_g, opt_d = recipe.make_optimizers(generator, discriminator, prior)
loss = recipe.make_loss()
```

The Halloween preset carries the extracted G/D rates, distinct moments and
epsilon, dense legacy Adam and `(-1,1,1)` labels. It declares constant rates and
zero critic penalty. Its learned prior explicitly uses betas `(0,.999)`,
epsilon `1e-8` and LR multiplier 2. Architecture, prior law, data, initialization,
budget and sampling remain task-owned. It excludes AlphaGAN's auxiliary `eloss`
and is an optimizer/loss transfer, not a reproduction of an unbound original
training run. Original decay semantics and historical success remain unknown.

Default-valued additions are projected out of archived configuration identities
and ordinary Recipe packets. Nondefault variants are recorded explicitly in
new recipe hashes and checkpoints. Archived qualification receipts, results,
search hashes and scientific cohorts remain their original evidence.

Scientific training comparisons still require protocol seed 0, isolated and
checkpointed streams, matched initial networks/batches, fixed task conditions,
one complete global configuration and an actual-training goal GIF. Screening
remains provisional. Confirmation, accepted calibration and registered
robustness are required for public-default adoption.
