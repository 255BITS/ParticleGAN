# Lower-learning-rate stability round and constant-KL posterior

User agreed to the proposed lower-LR control round and asked whether we can
have a variational formulation without a KL regularizer. Five configurations,
one shared seed24002, no seed sweep. Four controls: GAN, deterministic particle
AE-GAN, soft categorical stochastic AE-GAN without KL, categorical VAE-GAN.
Fifth: hard particle VAE-GAN with constant KL omitted from optimization.

Same toy, K400, z2, width128, batch256, 6000 updates, fixed sigma .0023566126.
G/E .0003, D .00045, prior .003 (all half the previous round). Eval generation
100k at every 2000 updates, reconstruction8192 with eight draws; online final
weights, no best-checkpoint selection. All likelihood scales tau=.03. This
change affects evaluated likelihood, not no-KL control training. AE surrogate
temperature .25, soft categorical temperature .0025. Hard surrogate .25.
Two posterior samples per stochastic training input. Original spread/GAN losses
unchanged, same initialization and independent data/prior/posterior RNG streams.
Both GPUs one worker each; five200-step implementation pilots precede full runs.
Five-minute training cap/run; pipeline preserves failed attempts. No exact resume.

## Why an explicit KL penalty is unnecessary for the new family

```
Prior:      k ~ Uniform(K); eps ~ N(0,I)
Posterior:  E(X) -> one particle k(X); eps ~ N(0,I)
Decode:     z = particle[k] + fixed_sigma * eps -> G(z)
Likelihood: X | z ~ N(G(z), tau^2 I)
```

For the joint latent `(k,z)`, q(k|X) is one-hot and q(z|k,X)=p(z|k).
Thus `KL(q(k,z|X)||p(k,z)) = log(K) + 0`. Learned particle means do not
change this cancellation: the same center and sigma appear in both conditional
Gaussians. Negative ELBO in 2D is `MSE/tau^2 + log(2*pi*tau^2) + log(K)`.
Training can drop both constants; reconstruction plus GAN/spread remains an
augmented variational objective. Reported ELBO retains log(K) (~5.9915 nats).
This is joint KL, not an assertion that KL to the marginal overlapping MoG
p(z) is exactly log(K). The latter can be smaller.

Hard choice discards uncertainty over particle identity. Randomness is within
the chosen component, sampled during training and evaluation. There is no
learned local offset/variance. Adding them generally restores a nonconstant
Gaussian KL. Soft q(k|X) instead has KL `log(K)-entropy(q)`, so deleting its
KL is not equivalent; that arm remains named stochastic AE-GAN.

The restricted family has a valid ELBO, but hard argmin supplies no useful
ordinary encoder gradient. We use the existing soft-routing backward surrogate
at .25: exact hard forward, selected-particle gradients, soft query gradients.
This is a BIASED straight-through optimizer, not unbiased ELBO gradient descent.
Soft categorical controls retain their unbiased two-draw score estimator.
There is no exhaustive best-reconstruction routing. As before, routing distances
cost O(batch*K*z_dim), reconstruction only two sampled codes/input.

The constant-KL argument is established in
[VQ-VAE, section3.1](https://arxiv.org/html/1711.00937v2#S3.SS1).
Our formulation extends it by retaining equal posterior/prior Gaussian noise
inside the selected particle. It is related to VQ-VAE, not a novelty claim or
an implementation of its exact commitment/codebook losses. Our shared particles
learn from reconstruction, GAN and spread. No aggregate posterior/prior usage
constraint is added: constant KL does not guarantee balanced particle use or
well-calibrated output diversity.

## Evaluation

Same numerical metrics as [first scout](../PROTOCOL.md), including prior decoder
coverage/HQ/width/SW1, expected and MAP reconstruction, shuffled codes, same-mode
retention, pair RMS, categorical/local KL, categorical MI and aggregate usage,
full likelihood predictive sampling and runtime. Hard diagnostics use the actual
one-hot posterior, never the soft gradient-surrogate probabilities. All eight
hard posterior draws share one identity but have independent local noise.

Five-arm leaderboard ranks final coverage, HQ, SW1. Same-count learning curves
and comparison to prior full-LR final checkpoints assess stability; no historical
checkpoint is substituted into ranking. This single paired trajectory cannot
establish general superiority. Source/config/checkpoint provenance, fixedsigma,
matched init/data/prior RNGs and per-input arrays will be verified. Original
trainer remains untouched so the first scout's completion certificates stay valid.
New entry point: experiments/train_mog_vae_stability.py.
