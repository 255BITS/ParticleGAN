# Recipe-owned priors

The task defines the initial prior distribution, capacity and sampling. The
recipe defines whether locations learn and how they are regularized:

```python
prior_update = "learned"       # or "frozen"
prior_regularizer = "vicreg"   # or "none", "coulomb", "sigreg", "moment_tether", "initial_mmd"
prior_reg = 0                 # existing coefficient; presets unchanged
prior_reg_target_std = 1
prior_reg_eps = 1e-4
prior_l2 = 0

prior_loss = prior_reg * variance_and_covariance(rows) + prior_l2 * mean(rows**2)
generator_side_loss = original_host_objective + prior_loss
```

`vicreg` is a variance floor plus off-diagonal covariance penalty; it has no
paired invariance term. Weights apply once. Frozen priors receive no penalty or
optimizer group. Initializing identical locations before freezing preserves
initial tensors and RNG consumption. Direct generator coordinates remain
generator parameters; they are not a sampled latent prior.

`initial_mmd` compares bounded current-location and fixed initial-location
panels with a multiscale IMQ kernel. It preserves distribution geometry without
anchoring particle identity. `GANTrainer` binds it automatically; caller-owned
loops call `make_prior_regularizer(prior=prior)` after public initialization and
checkpoint that prior. Weight zero captures no reference. Its exact estimator,
finite-panel approximation, reference buffers, disabled parity and software
training demonstration are documented in the
[feature readout](../reports/forge/prior-mmd/README.md).

`GANTrainer` keeps its historical unweighted `prior_regularization` diagnostic
for learned VICReg-like priors, including at weight zero. It contributes to the
loss only through `prior_reg`. Disabled weighted factories do no variance or
covariance work; frozen priors receive neither the diagnostic nor a penalty.
Frozen locations must have storage independent of network and optimizer state;
shared parameters and tensor views are rejected before freezing.

Opt-in `sigreg` uses the Gaussian characteristic-function quadrature from
[LeJEPA's minimal implementation](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md).
It uses 32 normalized random projections and 17 trapezoidal knots on [0,3],
preserving the upstream multiplication by the selected row count. The existing
`prior_reg` coefficient multiplies that statistic once. Mean shifts, expansion
and higher moments are constrained; rows are never centered or standardized.
It does not constrain generator output geometry or alter the initial prior.
`prior_reg_target_std` and `prior_reg_eps` belong only to VICReg.

The factory's SIGReg module owns a private CPU projection RNG with seed 0.
Retain it across updates and checkpoint it with the new
`Recipe.prior_regularizer_checkpoint(module)` /
`Recipe.restore_prior_regularizer(module, state)` helpers in caller-owned loops.
GANTrainer and the current Forge hosts save it automatically. Disabled or
frozen penalties draw nothing and add no checkpoint field. Projection draws
do not consume existing constructor, data, sampling, noise or evaluation RNGs.
The same projection law runs on CPU and CUDA; arithmetic stays on the rows'
device. See the [feature checks and bounded demo](../reports/forge/prior-sigreg/README.md).

```python
regularizer = recipe.make_prior_regularizer()
loss = original_loss + recipe.prior_regularization(selected_rows, regularizer=regularizer)
checkpoint = {**original_state, **recipe.prior_regularizer_checkpoint(regularizer)}
# After reconstructing the same recipe/module:
recipe.restore_prior_regularizer(regularizer, checkpoint.get("prior_regularizer"))
```

`moment_tether` preserves the initialized mean and full covariance in both
directions, without anchoring particle identities or imposing a Gaussian law.
After the public initializer, `Recipe.apply_prior_policy(prior)` records fixed
`moment_tether_mean`, `moment_tether_covariance` and `moment_tether_count` buffers
only for a learned prior with positive `prior_reg`. They remain part of the
prior's state, including own word/Gaussian continuation checkpoints. Rebinding
never replaces them with trained moments. Component callers bind the reference
explicitly:

```python
recipe.apply_prior_policy(prior)  # after initialization, before training
regularizer = recipe.make_prior_regularizer(reference=prior)
prior_loss = recipe.prior_regularization(rows, regularizer=regularizer)
```

Writing `v_i = C0_ii + prior_reg_eps`, its dimensionless loss is the mean over
dimensions of `(mu_i - mu0_i)^2 / v_i`, plus the mean over all covariance entries
of `(C_ij - C0_ij)^2 / (v_i * v_j)`. Covariance uses the unbiased denominator
`max(n - 1, 1)`. The variance epsilon makes tiny/rank-deficient references
finite without inverting covariance. `prior_reg` multiplies this entire loss
once; `prior_reg_target_std` is inactive. Positive epsilon is in latent variance
units. Multiple behavioral priors use count-weighted pooled initial moments,
matching their existing concatenated regularization rows.

The regularizer consumes no RNG and retains each host's row selection. Large
`GANTrainer` tables compare unique sampled rows to the fixed full initial
moments, so their penalty contains minibatch sampling noise even at initial
equilibrium. It does not change MoG width, sampling, or optimizer row ownership.
Weight zero and frozen policies create no reference buffers and do no moment
work. With the BCAP leader's DualNorm, the penalty changes the combined prior
gradient `g` before each unique sampled row is scaled by
`1/sqrt(||g||² + 0.001²)`. This nonlinear smoothed saturation means its
coefficient is not a displacement bound. Unsampled rows stay fixed even when
the full-table penalty produces dense gradients.

New tasks declare `execution.prior_contract: "recipe_owned_v1"` and omit
`execution.prior.learnable`. This replaces hidden behavioral prior penalties:

| Host | Previous host penalties |
| --- | --- |
| trajectory / residual_student | VICReg-like weight .05, target std 1; L2 .02 |
| cover_leftover | VICReg-like weight .05, target std .05; L2 .02 |
| ae_gan_hold | L2 .02 |

One global recipe now supplies these terms across tasks. Public preset weights
are unchanged; release07's explicitly declared .05 coefficient is retained on
behavioral hosts too. Host reconstruction, cover, residual and GAN objectives
keep their existing definitions and gates.

The ordinary view advances to revision 9. Its 30 assigned task cards and three
policy parent cards adopt the new contract; 26 other legacy cards retain their
original declarations. Archived source, receipts and outcomes remain intact.
Fresh Tier 1/Tier 2 measurements are required for every representative recipe.
Calibration profiles keep their separately declared learned-MoG admission rule;
their reference prior policy is resolved from the recipe, not the initial task
prior. Existing failed/provisional profiles gain no qualification from this change.

See [field ownership](forge-field-boundaries.md) and the
[research glossary](../RESEARCH_GLOSSARY.md).


Opt-in `prior_regularizer="coulomb"` uses softened inverse-distance repulsion
plus quadratic confinement to the initialized centroid and RMS radius. Its
fixed reference is captured by `apply_prior_policy` after public initialization
and saved as persistent prior buffers; component factories accept `prior=prior`.
The reusable knobs are `prior_coulomb_softening=.1`,
`prior_coulomb_confinement=.35`, and `prior_coulomb_neighbors=8` (bounded 1..64).
Caller row selection and sampled-row optimizer ownership stay unchanged.
See the [force law, public API demo and starting values](../reports/forge/prior-coulomb/README.md).
