# Two bounded GAN-only mechanism checks

The user clarified that the GAN formulation itself must work. Likelihood and
anchor fitting remain reference evidence. These two tests retain an
adversarial generator objective and do not use coverage or density fitting to
choose an accepted generator update.

**Nonlocal adversarial proposals:** the discriminator can value an absent
mode while its local input gradient points away. Enumerate actual observed
real points as possible destinations, score each one-particle move using the
unchanged native Rp generator loss, and then evaluate the selected move on
reserved G batches. The [first output-space test](pr84-adversarial-reallocation-assay.md)
passes: warm8→8 in both critic controls and cold3→4; all24 reserved losses
improve. Runtime1.36s. This is an output-space diagnostic, not a neural
acquisition pass. The [actual G/prior realization](pr84-adversarial-landing.md) also passes
all three copied-state fits and24 heldout comparisons in1.48s. An initial
baseline-shape bug is preserved as invalidV1; correctedV2 and direct paired
loss checks support these results. A short alternating-player continuation
is next. Fixed-critic loss descent
alone cannot prevent collapse, because the generator loss is separable over
fake samples.

**Common fixed instance noise:** test a single data-derived Gaussian input
noise channel on both real and fake samples inside each player's paired Rp
loss. This is an expectation of the loss on noisy pairs, not the old rule
that averages critic logits before applying the loss. Adapt a copied critic
to that game before testing warm and missing-mode G responses. The scale is
fixed before the assay; no radius or gain sweep follows a failure. It remains
a changed GAN observation model, with no standalone data-fitting objective.
There is no result yet.

Relevant primary research, checked September24,2026:

- [Mescheder et al., ICML2018](https://proceedings.mlr.press/v80/mescheder18a.html)
  analyze local convergence with instance noise. Their assumptions do not
  establish stability of this finite, misspecified, capped Rp/Adam host.
- [Sohn and Song, JASA2026](https://doi.org/10.1080/01621459.2026.2688612)
  train a joint temperature-conditioned adversarial model using convex
  interpolations of real observations. Their parallel tempering avoids an
  annealing schedule. It is a different method from fixed Gaussian instance
  noise: it adds temperature-conditioned G/D networks and uses a scaled
  Wasserstein objective with a coherency gradient penalty. Its estimation
  analysis assumes a globally minimizing generator,
  rather than proving stability of our stochastic updates. It is a possible
  observation-model reference, not a third active experiment.
- [Zhang et al., PRE, accepted August27,2026](https://doi.org/10.1103/ch8n-wrv6)
  alter the discriminator's final CDF activation with a learnable scale.
  This concerns activation/gradient behavior and is not evidence for the
  proposed input-noise channel.

These references motivate falsifiable mechanisms. None establishes a
constant-LR production solution, and none warrants skipping cold acquisition
and own-state continued training on the actual trainer.
