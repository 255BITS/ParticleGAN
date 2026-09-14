# Established-default audit (2026-09-14)

An independent read-only subagent audit compared `examples/100gaussians.py`
with the denoising trainer, model code, and every initial screening config.

The bcap formulation is the same shared `GradRegularizer`: half the sum of
real/fake expectations of `relu(||grad_x D||_2 - 1)^2`, coefficient 1, evaluated
every update. For both concat and UCD, the auditor numerically verified exact
equality of penalty values and all discriminator parameter gradients against
this explicit formula. The transition wrapper correctly holds noisy input,
timestep, and class fixed while differentiating the candidate sample.

Inherited defaults match: RpGAN logistic; Adam LR .0006, D multiplier 1.5,
prior multiplier 10, betas (0,.999); unique-row VICReg weight 1; three hidden
layers of width 128; LeakyReLU .2; Xavier initialization; Fourier frequencies
pi and 2pi; batch 256; 7000 updates; EMA .995; cosine decay starting at 60%
of training with a 5% floor.

One material mismatch: the initial screen used 4096 latent particles, while
the established example uses 20,000. This changes both support size and row
update frequency. The original run artifacts/configs remain intact.
`configs/denoising/screen_20k/reruns.json` specifies 24 corrected learned-prior
runs; `manifest.json` combines these with the original 24 Gaussian controls.
Gaussian controls never sample the initialization table, whose RNG is also
independent, so changing its unused count would not change training.
The analyzer explicitly ignores this inactive count when matching Gaussian
versus learned priors, records both counts, and rejects ambiguous matches.

Class/transition inputs and UCD heads are experimental adaptations.
Diffusion-step noise particles have no counterpart in the example: their
count 1024, LR multiplier 1, and moment penalty 0 are new experimental choices,
not recovered historical defaults.
