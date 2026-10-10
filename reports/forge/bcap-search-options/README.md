# BCAP optimizer loss and hyperparameter search options

**Forge can search the existing BCAP options now.** It supports finite numerical
grids and a bounded sampler over registered optimizer/loss candidates. This
audit compiles **15 existing structural categories**, all READY for the current
Tier 1 on CPU, and validates **40 public optimizer/loss pairs**. Changing an
optimizer, loss, momentum activation or smoothing activation requires a
structural base; numerical strengths then vary within that base.

The inventory covers **all 92 public Recipe fields, flattened into 102 scalar
paths**, with defaults, ownership and search permission. Start with the tables
below; [the complete flat table](flat-inventory.md) and
[machine-readable inventory](inventory.json) include task conditions and inactive
policy settings too. Flattening exposes pair components for inspection: Forge
still accepts `betas`, `d_betas`, `prior_betas` and `direct_particle_betas` as
whole pairs, not dotted or indexed scalar keys.

Inspected `develop` at `59941e9e8` on 2026-10-08; the
[audit receipt](audit-receipt.json) records the full commit and source hashes.
This is a software/readiness report. No research campaign was submitted, no
scientific result was regraded, and the
[current technique inventory](../technique-inventory.md) remains the only
leaderboard for this goal.

## Baselines and prior evidence

The current public `get_recipe("bcap")` selects unsmoothed DualNorm: G/E step
`.012`, D multiplier `1.5` (step `.018`), prior multiplier `2.5` (step `.03`),
zero momentum, relativistic loss, BCAP coefficient/cap `1`, and constant rates.
`get_recipe("bcap_adam")` retains native Adam at `.00425`, D multiplier `1`,
prior multiplier `2`, moments `(0,.999)`, and epsilon `1e-8`.
Both disable the guard, anchor, latent damping, direct-particle gain, extra
prior regularization, EMA tracking and additive training noise.

**Forge API v1 intentionally resolves `recipe_preset="bcap"` to the historical
Adam preset.** It preserves archived cards. A new DualNorm candidate must
explicitly bind `optimizer_family`, `lr`, `d_lr_mult` and `prior_lr_mult`;
the preset name alone will not reproduce the public DualNorm starting recipe.
See [the versioned resolver](../../../experiments/forge/api.py) and
[public presets](../../../particlegan/recipes.py).

Relevant evidence from [compiled memory](../EXPERIMENT_MEMORY.md), in its
original cohorts:

- The [41-configuration optimizer screen](../dualnorm-tier1/README.md) found no
  optimizer beating Adam's historical 3/6 Tier 1 count. Zero-momentum DualNorm
  improved ring acquisition and tied that count. Its older task revision is
  distinct from today's acquisition/retention split.
- The [25-configuration pacing study](../dualnorm-pacing-v2/README.md) raised
  that historical count to 4/6 using `.012/.018/.03`. Those rates are already
  explored; do not repeat the same grid as fresh evidence.
- The [three-strength smoothing study](../bcap-six/README.md) passed all six
  revision-8 Tier 1 tasks at `1e-5`, `1e-4` and `1e-3`. The PASS-count/hash
  tie-break selected `1e-5`; it did not establish a fastest or best-retaining
  value. The public preset still has smoothing `0`.
- The [frozen selected-recipe Tier 2 study](../bcap-tier2/README.md) had
  6 PASS, 11 FAIL and four image setup errors. Gaussian retention and the
  joint-word inverse map failed; acquisition alone is insufficient.
- The [convolution capability study](../bcap-convolution/README.md) completed
  all four formerly unsupported images with `optimizer_convolution="per_offset"`.
  All four sustained gates failed. Its source and structural candidate differ
  from the earlier setup-error receipts; these results cannot fill those cells.

These reports contain the original numerical gates, actual-training GIFs,
failures, costs and provenance. Revision-8 screening remains provisional and
cannot by itself authorize public-default adoption.

## Optimizer categories

All eight BCAP-compatible families are implemented by the public Recipe
factories; G/E share one player and D remains separate. Native-Adam steps and
normalized steps have different units, so copy neither one's LR range blindly
into the other. [Implementation](../../../particlegan/optim/dualnorm.py).

| `optimizer_family` | Network update | Learned prior update | Applicable additional knobs / registered base |
| --- | --- | --- | --- |
| `adam` | Native PyTorch Adam | Adam on locations | G/E, D and prior moments/epsilon; AMSGrad; optional beta2 schedule. `bcap-pure-adam-v2` |
| `sgda` | Raw simultaneous SGD | Raw SGD | Rates only; moments, AMSGrad and epsilon do not change this update. `bcap-sgda-v1` |
| `nsgda_global` | Normalize the concatenated gradient of each player | Separately normalized prior player | Rates and role epsilon; no Adam moments. `bcap-nsgda-global-v1` |
| `nsgda_layer` | Normalize each parameter tensor separately | Normalize the whole prior tensor | Rates and role epsilon; this is tensor normalization, not one norm per multi-tensor layer. `bcap-nsgda-layer-v1` |
| `ada_nsgda` | Tensor-normalized raw direction with Adam step magnitude | Same magnitude graft | beta1 must stay `0`; beta2, epsilon, AMSGrad and rates. `bcap-ada-nsgda-v1` |
| `dualnorm` | Matrix polar direction; normalized scalar/vector direction | Normalize actual sampled rows; other rows stay fixed | Network momentum `0/.5/.9`, shared smoothing, role epsilon and optional convolution mode. `bcap-dualnorm-zero-v1`, `bcap-dualnorm-momentum-v1`, `bcap-dualnorm-smoothed-v1`, `bcap-dualnorm-convolution-v1` |
| `dualnorm_D_only` | DualNorm D; Adam G/E | Adam | D momentum; Adam moments for G/E/prior; optional separate Adam rate. Smoothing/convolution opt-ins are unsupported for this family. `bcap-dualnorm-d-only-v1` |
| `particle_rownorm_only` | Adam G/E/D | Normalize actual sampled rows | Network moments/epsilon and optional separate Adam rate; prior moments inactive. `bcap-particle-rownorm-only-v1` |

DualNorm's sampled-prior row update has no momentum. `optimizer_momentum`
controls normalized network tensors, not latent rows. Default matrix rank
truncation and its dtype/shape threshold are fixed implementation rules, with
no Recipe search field. Smoothing `lambda>0` changes singular weights to
`s / hypot(s,lambda)` and vectors/rows to `g / hypot(norm(g),lambda)`; one
global scale applies to all these roles.
[Smoothing contract](../../../docs/dualnorm-smoothing.md).

`optimizer_convolution` is `"none"` or `"per_offset"`. The latter binds actual
Conv2d/ConvTranspose2d module layouts and uses a polar update per channel group
and spatial offset, scaled by `sqrt(out/in)/(height*width)`. It is a structural
category, not a numerical axis. Conv1d, Conv3d and unlabelled high-rank parameter
iterables remain unsupported. [Convolution contract](../../../docs/dualnorm-convolution.md).

`adam_variant` is `"pytorch"` or `"tensorflow_v1"`. The dense legacy variant
changes epsilon/bias-correction placement and is a structural category. It
requires `optimizer_family="adam"`, constant betas, no AMSGrad, and no weight
decay or accelerated/differentiable modes. The registered Halloween base uses
**zero BCAP strength**, so it is not a BCAP-on legacy-Adam control; retaining
positive BCAP needs a separately declared base.
[Legacy settings](../../../docs/forge-search-spaces.md).

`optimizer_family="formulation"` selects the KA2/K3P intervention machinery,
not pure BCAP. Historical `k3p-bcap-matched-v1` belongs to the separate
**BCAP with K3P** formulation. It cannot be obtained by changing a single
optimizer axis on the pure-BCAP base. No public BCAP categories exist for
AdamW, RMSProp, Adagrad, L-BFGS, extragradient, optimistic updates, Nesterov or
general momentum SGD; research scratch implementations are not Forge capability
bindings.

## Adversarial loss categories

The five losses are independent of the fixed BCAP penalty. With real score
`r`, fake score `f`, mean `E` and `sp=softplus`, both losses below are minimized.
[Exact public implementation](../../../particlegan/gan_loss.py).

| `loss` | Critic loss | Scalar generator loss | Existing native-Adam base |
| --- | --- | --- | --- |
| `relativistic` | `E sp(f-r)` | `E sp(r-f)` | `bcap-pure-adam-v2` |
| `non_saturating` | `E sp(-r) + E sp(f)` | `E sp(-f)` | `bcap-pure-non-saturating-joint-v2` |
| `hinge` | `E max(1-r,0) + E max(1+f,0)` | `-E f` | `bcap-pure-hinge-joint-v2` |
| `wasserstein` | `E f - E r` | `-E f` | `bcap-pure-wasserstein-joint-v2` |
| `least_squares` | `.5 E(r-b)^2 + .5 E(f-a)^2` | `.5 E(f-c)^2` | `bcap-pure-least-squares-joint-v2` |

Least-squares `loss_labels=(a,b,c)` flatten to fake, real and generator targets.
Defaults are `(0,1,1)`; `(-1,1,1)` is also representable. All three labels must
be finite. Nondefault labels require least squares, and labels are technique
fields rather than numerical grid axes.

Joint BiGAN hosts must use `joint_g_loss`: the non-saturating objective adds
`E sp(r)`, hinge/Wasserstein add `E r`, and least squares adds `.5 E(r-a)^2`
for the encoder's real stream. The relativistic joint expression stays paired.
The `*-joint-v2` bases represent this public objective; retain older v1
evidence under its original implementation.

All **8 optimizers x 5 losses** construct as public recipes in the audit.
Forge does not automatically register their full Cartesian product. The
current registry supplies five Adam loss bases and the seven other optimizer
bases primarily with relativistic loss. For example, DualNorm+hinge requires
a new structural candidate binding both settings before it can enter a
categorical space. Hinge margins, non-saturating labels, Wasserstein drift,
label smoothing and per-loss scale factors have no public Recipe knobs.

## BCAP penalty and all numerical grid fields

The critic adds
`reg_coeff/2 * (E max(norm(grad_real)-reg_kappa,0)^2 + E max(norm(grad_fake)-reg_kappa,0)^2)`.
These are **L2 input-gradient units**, with no dimension/RMS normalization.
`reg_every=k` applies every k-th call and multiplies strength by k.
The critic's observation clock is checkpointed.
[Penalty implementation](../../../particlegan/grad_regularizers.py).

`reg_arm="b_cap"` is required for pure BCAP. `"a_r1r2"` is the separately
declared fixed squared-gradient alternative; swapping it changes formulation.
KA2/K3P penalties have their own optimizers and controls. Penalty-off, a
zero cap, or a scheduled endpoint reaching zero also crosses a mechanism
boundary. BCAP has no real/fake coefficient split, independent cap split,
norm selector or finite-difference alternative in Recipe.

The following covers **every one of Forge's 27 numerical grid fields**.
Defaults are the current public BCAP preset; inherited Adam moments are listed
for completeness, although full DualNorm does not consume them.
`null` role values inherit from G/E. Numerical values must be finite.
[Whitelist and ownership](../../../experiments/forge/boundaries.py),
[activity and mechanism checks](../../../experiments/forge/techniques.py).

| Recipe field / flattened components | Default | Domain | Activity and same-base constraints |
| --- | --- | --- | --- |
| `lr` | `.012` | `>0` | G/E base step; normalized roles use normalized units |
| `d_lr_mult` | `1.5` | `>0` | D step is base step times multiplier |
| `prior_lr_mult` | `2.5` | `>0` | Sampled learned-prior step; inactive for direct coordinates or absent/frozen priors |
| `betas[0]`, `betas[1]` | `0`, `.999` | Each in `[0,1)` | Adam G/E; magnitude graft consumes only beta2 and requires beta1=0. Positive/zero moment activation is structural |
| `d_betas[0]`, `d_betas[1]` | `null`, `null` | Pair in `[0,1)` or `null` | Inherit shared pair; active for Adam D, graft and row-only hybrid |
| `prior_betas[0]`, `prior_betas[1]` | `null`, `null` | Pair in `[0,1)` or `null` | Inherit shared pair; active for learned Adam/grafted prior, not row-normalized prior |
| `direct_particle_betas[0]`, `direct_particle_betas[1]` | `0`, `.9` | Each in `[0,1)` | Only formulation optimizer with direct generated coordinates; inactive for all pure BCAP optimizers |
| `eps` | `1e-8` | `>0` | G/E plus inherited D/prior denominator or epsilon skip; SGDA ignores it |
| `d_eps` | `null` | `>0` or `null` | D override, else shared epsilon; SGDA ignores it |
| `prior_eps` | `null` | `>0` or `null` | Sampled prior override, else shared epsilon; SGDA ignores it |
| `amsgrad` | `false` | Boolean | Adam/graft only; switching on/off is structural |
| `optimizer_momentum` | `0` | Exactly `0`, `.5`, `.9` | DualNorm or D-only; `0` versus positive needs separate base. `.5` and `.9` can share positive-momentum grid |
| `optimizer_smoothing` | `0` | `>=0` | Full DualNorm only; off/on structural, positive strengths searchable within enabled base |
| `optimizer_adam_lr` | `null` | `>0` or `null` | D-only or row-only hybrid; activation of independent Adam-rate path is structural |
| `reg_coeff` | `1` | `>=0` | Positive strengths tune BCAP; zero is a separate penalty-off base |
| `reg_kappa` | `1` | `>=0` | Positive cap strengths tunable; zero cap changes mechanism |
| `reg_every` | `1` | Positive integer | Lazy penalty interval; updates, data batches and scoring cadence stay task-fixed |
| `reg_coeff_end` | `null` | `>=0` or `null` | Cosine coefficient endpoint; declaring/removing/changing constant-to-varying schedule is structural; preserve endpoint activity |
| `reg_coeff_anneal_end` | `.2` | `(0,1]` | Fraction of fixed task horizon; active only with a changing coefficient schedule |
| `prior_reg` | `0` | `>=0` | Public prior spread/decorrelation objective; off/on structural; behavioral objectives can be task-owned |
| `lr_anneal_start` | `.6` | `[0,1)` | Cosine schedule only; inactive while both resolved floors are one |
| `lr_floor` | `1` | `[0,1]` | Prior floor, or inherited network floor; zero terminal updates are structural |
| `network_lr_floor` | `1` | `[0,1]` or `null` | G/E/D floor; `null` inherits prior floor. Positive floors stay in the existing cosine mechanism |
| `beta2_end` | `null` | `[0,1)` or `null` | Plain Adam with `adam_variant=pytorch` only; one terminal beta2 for all roles; schedule activation structural |
| `beta2_anneal_end` | `.2` | `(0,1]` | Active only for a changing beta2 schedule; fixed task horizon |
| `lr_decay_rate` | `.96` | `(0,1]` | Exponential LR base only; `1` disables decay and crosses the signature boundary |
| `lr_decay_steps` | `50000` | Positive integer | Exponential LR base with rate below one; counts whole training updates |

`lr_schedule` has `"cosine"`, `"constant"` and `"exponential"` categories.
The BCAP preset selects the existing **cosine path with both floors equal to
one**, so actual rates stay constant. An explicit constant/exponential selector
is a structural candidate. `lr_decay_staircase` is a Boolean technique field.
Exponential decay applies `rate^(completed_updates/steps)`, flooring that
exponent for staircase mode. It advances once per whole update, not separately
for each optimizer application. Noncosine modes require no continuous policy
and no network horizon cap.

Public component factories also accept options outside Recipe. They do not
automatically become search parameters or public-trainer overrides:

| Factory-only setting | Default / meaning | Forge status |
| --- | --- | --- |
| Native Adam `weight_decay`, `maximize`, `foreach`, `fused`, `capturable`, `differentiable` | Native optimizer kwargs; weight decay `0`, maximize/capturable/differentiable `false`, execution modes unspecified | Not Recipe axes. Normalized and legacy optimizers reject enabled unsupported modes. Expose and bind any new scientific setting before searching it |
| `make_prior_regularizer(target_std, eps, weight)` | Spread floor `1`, stabilizer `1e-4`, weight inherited from `prior_reg` | Only weight has a Recipe axis. Target/std stabilizer need an explicit public binding; they are not the optimizer epsilon fields |
| `make_critic_penalty(coeff, kappa, lazy_k, anchor_weight, r1_real)` overrides | Replace local penalty settings; `output` selects logits and `collect_stats` enables diagnostics | Search the corresponding Recipe fields, not unrecorded factory overrides; KA2-only switches do not add BCAP capabilities |
| `make_prior(sigma, ...)`, `encode(draws=2, ...)` | Prior construction and posterior sampling arguments | Sampling/prior conditions belong to the task and remain fixed |

The [prior regularizer](../../../particlegan/vicreg_loss.py) adds a standard
deviation hinge and off-diagonal covariance penalty; it does not impose a
Gaussian target or provide separate public variance/covariance weights.

## Remaining fields and fixed comparison conditions

These fields are all present individually in the [102-path appendix](flat-inventory.md).
They are outside the finite numerical grid whitelist, even when the ownership
registry calls them hyperparameters:

| Fields | Meaning / options | BCAP search treatment |
| --- | --- | --- |
| `network_lr_horizon_cap` | Positive integer or `null`; separate network horizon | An ordinary structural declaration if required; fixed task horizon is never shortened by a search |
| `ema_decay`, `serve_average` | EMA tracking `[0,1)`; policy serving-average scale `>=0` | Default both zero; live/EMA serving cohorts stay separate; policy averaging requires supported lifecycle bindings |
| `input_noise_std`, `input_noise_anneal_end`, `output_noise_std`, `output_noise_warmup` | Nonnegative noise scales; input end `(0,1]`, output warmup `[0,1]` | Off in BCAP; changing training law needs declared candidates; clean/noisy serving gates stay separate |
| `routing_temperature`, `observation_sigma`, `reconstruction_weight` | Positive routing/likelihood scales; nonnegative reconstruction strength | Encoder/auxiliary losses; behavioral hosts own their frozen objectives. Do not turn these into host-specific tuning |
| `ucd_weight`, `alpha_bar[0..4]` | Nonnegative UCD weight; strictly decreasing positive DDGAN diffusion sequence starting at `1` | Conditional/DDGAN mechanisms, inactive on scalar BCAP; task applicability must be explicit |
| `reg_anchor_min_decay`, `reg_anchor_weight`, `d_guard_ratio`, `d_guard_min_steps`, `latent_damping_max_rate`, `direct_particle_gain` | Anchor, spike guard, A2 damping and direct-coordinate gain | Pure optimizers require guard/anchor/damping zero and direct gain false; KA2/K3P controls cannot simply be enabled on DualNorm |
| `birth_death_cells`, `birth_death_metric_rank`, `birth_death_chunk` | Positive feature-law sizes | Inactive BCAP policy fields; feature-cell variants need DV12 stationarity and critic birth/death |
| `model`, `encoder_mode`, `conditioning`, `num_classes`, `ucd_target`, `distance_reduction` | `gan/ddgan`; `none/ae/categorical/hard`; `scalar/conditional/ucd`; positive class count; `class/time_class`; `sum/mean` | Technique/host compatibility, not optimizer search axes |
| `continuous_policy`, `lr_control`, `output_noise_mode`, `critic_r1_real`, `critic_payoff_damping` | Continuous controller, `mobility/stationarity`, `fixed/learnable/mobility`, KA2 switches | Pure BCAP has no continuous controller; compatibility validation rejects unsupported combinations |
| `particle_birth_death`, `row_evidence_gate`, `table_release_rule`, `row_evidence_hot`, `row_evidence_exclude`, `row_evidence_hold`, `row_evidence_null` | Independent/routed evidence and release controls; release `any/both/never/anchor`; null `theory/scaled` | E22/Atlas policy mechanisms; not ordinary BCAP numerical settings |
| `birth_death_space`, `birth_death_isolation`, `birth_death_feature_scale`, `birth_death_backend`, `birth_death_parent_policy`, `row_policy` | `data/critic`; isolation Boolean; `none/std`; `knn/feature_cells/auto`; `real_anchor`; `independent/routed_paired` | Separate policy candidates and supported task lifecycle required; no silent transplant into BCAP |
| `reopen_signal`, `reopen_anchor`, `reopen_guard` | `data/none/optimizer`; `hold/release`; `null/settled` | Controller-only settings, inactive on ordinary BCAP |
| `name` | Report label | Metadata only; not an algorithm or search dimension |
| `z_dim`, `num_particles`, `prior_kind`, `sigma_rel`, `standardize`, `batch_size`, `total_steps` | Architecture/resources, prior and schedule horizon | Task-owned; forbidden numerical search axes |

A loss search must keep task architectures, target laws, initial states, batch
sequences, priors, clean/live sampling, update budgets and evaluation cadence
unchanged. All tasks receive one global trainer recipe. Protocol seed is `0`;
constructor, data, prior/noise and evaluation streams stay isolated and
checkpointed. Explicit identity/zero fixtures remain separate initialization
cohorts.

The current `discriminator_stability` view is **revision 8**, with required
denominators **6 / 21 / 2**. Its Tier 1 conditions are:

| Task | Updates | Declared prior | Gate scope |
| --- | ---: | --- | --- |
| `gaussian1d_smoke` | 1000 | Learnable MoG, sigma `.1` | Acquisition with independent same-state confirmation |
| `two_pole` | 80 | Direct sample coordinates; explicit cloud exception | Sustained movement/slope checks |
| `unused_token_hold` | 200 | Parameter-only cloud exception, no sampled latent prior | Sustained movement and unused-parameter hold |
| `ae_gan_hold` | 250 | Learnable MoG, sigma `.025` | Sustained reconstruction and hold |
| `ring16_acquisition` | 1600 | Learnable MoG, sigma `.1` | Full quality, coverage, mass and component-shape checks |
| `five_word_joint_smoke` | 20001 | Five learned cloud rows; explicit finite-vocabulary exception | Joint acquisition with independent same-state confirmation |

Do not apply the old universal five-terminal-check description to the new
Gaussian/word smoke tasks. Their Tier 2 continuations test retention under their
own declarations. Full threshold/cadence bindings are in the audit receipt and
[task/view declarations](../../../configs/forge/views/discriminator_stability.json).

## Forge readiness checks and gaps

[audit.py](audit.py) exercises the real configuration and categorical compiler
against this checkout. [plan-space.json](plan-space.json) is an **admission
fixture**, with one literal LR per category, not a proposed scientific search.
Its 15 categories deliberately inspect existing declarations, including older
rates; a training study must declare its intended numerical domains separately.

- **15/15 categories READY**, using an empty isolated CPU queue and all six
  current required Tier 1 tasks. All 29 required task cells plus the optional
  clock diagnostic remain in each row, including untuned later tiers.
- **81 axis probes** cover all 27 whitelist fields on Adam, unsmoothed
  DualNorm and smoothed DualNorm. Receipts retain both acceptance and exact
  rejection reasons. A rejection for an absent schedule or incompatible
  optimizer is expected, not missing numerical-search support.
- **40/40 public optimizer/loss pairs** pass Recipe, loss-factory and penalty
  validation. This checks representation, not trained behavior or all host support.
- **11 expected guards pass**: optimizer/loss/convolution/resource axes,
  off/on mechanisms, the 256-configuration limit and aggregate reservation
  rejection. Recompilation is identical and leaves global Python RNG unchanged.
- **205 existing tests pass** for configuration search, finite spaces,
  boundaries, smoothing admission, losses and public BCAP/legacy recipes.
  `forge compile --check` also reports research memory CURRENT; these report
  files do not change scientific recall inputs.

The Tier 1 reservation is **2,520 seconds per configuration**, including the
separate 300-second clock diagnostic. The fixture's 15 categories reserve
**37,800 seconds on paper** and spend zero training seconds. A real N-trial
Tier 1 study must cover at least `2520*N` seconds with the current declarations;
later-tier studies must use their actual grouped reservations. Each admitted
configuration completes its runnable current-tier peers before required
failures block higher tiers. No timing/speed objective is currently available.

Remaining gaps and precautions:

1. **SGDA epsilon activity is overdeclared.** All three `eps/d_eps/prior_eps`
   mutations are admitted for `bcap-sgda-v1`, but its update is exactly
   `parameter += -lr*gradient` and reads no epsilon. Exclude those axes;
   admission alone does not prove numerical influence. This report preserves
   the gap without modifying the trainer/search validator.
2. Numerical search cannot directly tune every public numeric field. The
   nonwhitelisted settings above require a supported ordinary candidate, or a
   separately reviewed ownership/activity extension if numeric grids are needed.
3. The optimizer/loss Cartesian product needs registered structural bases.
   Role optimizer families cannot be chosen independently beyond the existing
   two hybrids. There are no per-role smoothing, separate G/E moments, weight
   decay, clipping or D:G update-count Recipe axes.
4. The compiler's roster check covers sampled trials and an initial point for
   unsampled categories. With a large conditional domain, compilation alone
   does not prove every unsampled combination is legal. Keep ranges inside a
   reviewed signature and use full grids for small categorical comparisons.
5. All categories are sampled from one finite population; categories with
   more numerical combinations receive more probability. A random sample need
   not include every optimizer/loss. Use an explicit grid union or sample the
   complete small population when coverage is the purpose. Adaptive Bayesian,
   evolutionary and asynchronous pruning strategies are not implemented.

## Suggested next search

Use the selected positive-smoothing DualNorm recipe as a declared reference,
and keep the already measured smoothing points and old rate grids visible as
prior evidence. A bounded new numerical question can vary positive BCAP
strength/cap and player pace while retaining positive smoothing and zero
momentum. Those fields are already searchable. Choose the finite domain and
budget before execution; the report does not authorize a sweep.

For an optimizer/loss comparison, first declare the missing combinations as
reusable structural bases, retaining one global recipe per trial. Start with a
small explicit roster rather than the full product of every field. Prioritize
Gaussian and joint-word **retention** once smoke prerequisites pass, because
the current 6/6 acquisition result does not resolve their documented failures.
If later-tier results participate in selection, declare that tier cap and full
budget up front; they then count as tuning evidence, not independent confirmation.

Rank complete configurations by Forge's declared required PASS-count objective
and deterministic content-hash tie-break, with raw metric failures and cost
beside the result. Update the existing single leaderboard after completion.
No per-task winners, seed-only repeats, default adoption or speed ranking follow
from this readiness report.

## Reproduce and tail logs

From the repository root in the project environment:

```sh
mkdir -p runs/software/bcap-search-options
python -u reports/forge/bcap-search-options/audit.py \
  --output runs/software/bcap-search-options > runs/software/bcap-search-options/audit.log 2>&1
tail -f runs/software/bcap-search-options/audit.log

python -m pytest -q tests/test_forge_configuration_search.py \
  tests/test_forge_search_space.py tests/test_forge_boundaries.py \
  tests/test_forge_dualnorm_smoothing_search.py tests/test_adversarial_losses.py \
  tests/test_halloween_recipe.py tests/test_pure_bcap_recipe.py
```

The direct public CLI also accepts the admission fixture, without training:

```sh
python -m experiments.forge search compile \
  reports/forge/bcap-search-options/plan-space.json \
  --output runs/software/bcap-search-options/manifest.json
python -m experiments.forge search plan runs/software/bcap-search-options/manifest.json
```

Compiled manifests bind source/runtime and must be regenerated with a new
identity after a binding change. Future admitted campaigns use
`python -m experiments.forge logs --follow --campaign CAMPAIGN_ID`.
Keep stdout, JSONL, JUnit, checkpoints and manifests local; commit only compact
reports/receipts and reproduction sources.
