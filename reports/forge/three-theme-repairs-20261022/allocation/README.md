# BISPF output-population repair: energy witness

**Candidate:** `bispf-output-energy-v1`, parent `bispf-optimized-quintic-v1`.
The proposed single global delta is `output_energy_weight=1.0`. The common
three-theme diagnostic study owns its frozen controls, task panel and finite
budget. This document records the mechanism before training; it does not claim
a measured repair, qualification or default change. The parent comparison owns
the one current leaderboard for this goal.

The selected BISPF rare-mass run loses its rare population between updates 350
and 400; anisotropic density has wrong proportions despite acceptable endpoint
shapes; bars covers two of four modes. Those failures retain the identities in
investigation PRs [470](https://github.com/255BITS/ParticleGAN/pull/470),
[472](https://github.com/255BITS/ParticleGAN/pull/472) and
[464](https://github.com/255BITS/ParticleGAN/pull/464). Their original tensors are
unavailable, so this is a prospective mechanistic comparison, not a claimed
causal reconstruction or unchanged checkpoint-recovery rerun.

For existing G-phase generated samples X and real samples Y, add

`[mean |X-Y| - .5 mean_distinct |X-X'| - .5 mean_distinct |Y-Y'|] / real_pair_RMS`.

Distances are full-output Euclidean norms after flattening each sample. Self
terms use ordered off-diagonal pairs; finite estimates may be negative, and
clipping would alter their gradient. The numerator is an unbiased estimate of
half population energy distance; division by a batch-derived target scale means
the complete ratio is not unbiased. The real batch and its global RMS
distinct-pair distance are detached. Exactly zero target RMS uses one. This
avoids the prior entropic study's constant-panel epsilon amplification without
fitting task-specific widths or components. Near-constant nonzero target panels
still carry their own scale; no universal finite-sample variance guarantee is
claimed. Coincident samples use the norm's zero subgradient, so exact duplicate
particles need not immediately split.

The intended repair is a long-range distribution restoring field. Real
attraction includes missing populations even when they are far from generated
support; generated repulsion discourages moving every sample into a common
mode. It consumes only existing batches and adds no forward, RNG draw, retained
state, component label, oracle centre, projection or sampling rule. Both G and
learned prior receive its ordinary pullback. The adversarial signal remains the
protected loss; this additional marginal signal is not protected.

The recipe retains non-saturating adversarial loss, learned prior locations,
fixed task-owned initial prior distributions/capacities and widths, SBNP .01,
all optimizer rates/momentum/smoothing, Born-Infeld critic penalty, thermal
controllers, initialization, quintic convolution policy, update allowances,
named-stream batches, sampling and evaluation cadence. This is a trainer
objective change, not an evaluator or target-law change.

Prior negative controls matter. [Round five](../../bcap-physics/round5/README.md)
found that sliced/local transport could repair rare density while losing narrow
shape; balanced assignment repaired width but lost rare/native gates; routing
transport to prior alone did not preserve rare quality. The compiled
[moonshot memory](../../EXPERIMENT_MEMORY.md) records entropic/confidence
anisotropic repairs with pole/unused-token regressions, and constant-panel
normalization amplifying squared error 4,194,304 times. Energy uses neither
quantile matching, a local bandwidth, balanced assignment nor epsilon-scaled
squared residuals. These differences justify one new comparison, not an
expectation of guaranteed transfer.

Predictions and falsifiers:

- Unequal-mass scheduled rare occupancy and `min_mass_ratio` improve; complete
  covariance gates must also pass before calling it a repair.
- Anisotropic `mass_tv` and bars mode count improve without trading away shape
  or clean image quality.
- Ring16 acquisition, mode hold, stripes and Gaussian smoke retain every
  matched-control gate. Any lost gate rejects the global replacement.
- A smaller energy value alone, passing broad moments, isolated endpoint wins
  or a selected checkpoint cannot satisfy a sustained task gate.

The module's analytical software checks verify the distinct-pair denominators,
finite-difference gradient, long-range mass-correction direction, rotation/
translation/positive-scale invariance, zero-target unit scale, target gradient
isolation, finite duplicate-sample gradients and zero RNG consumption. Public
trainer checks additionally require opt-in behavior, identical consumed stream
states and exact checkpoint continuation. These checks are separate from
learned-quality toy evidence. The controlled study's actual API training GIFs
and numerical sustained metrics will supply that evidence after execution.

Shared integration: `Recipe.output_energy_weight` defaults to zero and accepts
finite nonnegative values. In `GANTrainer._step`, calculate
`weight * energy_distance(fake_g, real_g)` only when enabled, add it to total
`loss_g`, and expose detached `output_energy`. Every one of the fifteen planned
ordinary Forge hosts uses this trainer, including native100. No second host
training loop or recipe-specific evaluator is introduced. Other public
component/model hosts require an explicit consumer before admitting an active
candidate.
