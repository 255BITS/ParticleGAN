# Recipe-owned priors

The task defines the initial prior distribution, capacity and sampling. The
recipe defines whether locations learn and how they are regularized:

```python
prior_update = "learned"       # or "frozen"
prior_regularizer = "vicreg"   # or "none" (requires prior_reg=0)
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

`GANTrainer` keeps its historical unweighted `prior_regularization` diagnostic
for learned VICReg-like priors, including at weight zero. It contributes to the
loss only through `prior_reg`. Disabled weighted factories do no variance or
covariance work; frozen priors receive neither the diagnostic nor a penalty.
Frozen locations must have storage independent of network and optimizer state;
shared parameters and tensor views are rejected before freezing.

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
