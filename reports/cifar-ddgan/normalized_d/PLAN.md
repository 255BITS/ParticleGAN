# Discriminator normalization round

Status: both full runs completed and certified. See ../READOUT.md for results.

Question: does per-image GroupNorm in the discriminator improve CIFAR learning
without changing the train_denoising.py DDGAN formulation?

Two width32/UCD configurations: inherited toy rates and the existing image-rate
anchor. Each differs from its saved unnormalized control only by d_norm: group
and output directory. Seed24002, 10k updates, batch64; no seed sweep.
Four-step posterior, continuous xt/time conditioning, Rp logistic, UCD CE .02,
candidate-only bcap1/kappa1, learned20k x128 prior with VICReg1, Gaussian step
noise, EMA .995 and LR schedule all retained. GroupNorm retains D additive
conditioning and independent per-sample scores.

Validation before launch: 6 image/resume tests +19 toy/regularizer tests passed;
archived-source G and unnormalized D have exactly matching initialization,
state keys and outputs. Both real-CIFAR GPU smoke runs completed and certified.
Known test warnings are the pinned FID SciPy deprecation and an existing
regularizer-test scalar conversion; no new training warning.

Final comparison: 50k FID using the existing cached 50k real protocol, actual
class-row sample grids, training time, and checkpoint diagnostics. Intermediate
5k FID is diagnostic only. No automatic default promotion before final results.

Tail: tail -F results/cifar_ddgan/live.log
Manifest: configs/cifar_ddgan/normalized_d/manifest.json
