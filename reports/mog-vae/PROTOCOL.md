# Particle VAE scout: prespecified protocol

Same feature worktree, same 100-Gaussian 2D dataset as the particle AE-GAN scout.
Twelve mechanism/settings arms, one shared seed (24002), no seed sweep.
Final online weights at 6,000 updates; no best-checkpoint selection. Both GPUs,
one worker per GPU. Five 200-update implementation pilots must pass first.
Each full run has a five-minute training cap; failures remain failures.

## Model and objective

Prior: `k ~ Uniform(400); eps ~ N(0,I); z = p[k] + sigma*eps`.
`p[k]` are learned standardized particles. Sigma is calibrated once at .025
of initial median nearest-neighbor distance and then fixed (about .0023566).

Main posterior:

```
E(X) -> query
q(k|X) = softmax(-squared_distance(query, p[k]) / temperature)
k ~ q(k|X)
z = p[k] + fixed_sigma * eps, eps ~ N(0,I)
G(z) -> reconstructed X
```

This is an explicit variational posterior over `(k,z)`. Its conditional local
Gaussian equals the prior conditional, so its continuous KL is exactly zero.
The only KL is `sum_k q(k|X)*log(400*q(k|X))`. There is no per-input learned
Gaussian ball in this main family. Reparameterization and KL are different
parts of variational inference; removing learned local variance does not
require removing the variational objective.

Local ablation: `E(X) -> (query,u,log_s)` and
`z = p[k] + sigma*(u + exp(log_s)*eps)` with shared local offset/scale across
components per input. Adds exact `0.5*sum(u^2 + exp(2*log_s)-1-2*log_s)` KL.
Offset is bounded to (-3,3), log scale to (-3,3), initialized at zero. Thus
local posterior std starts equal to fixed prior sigma and may learn independently.

Decoder likelihood: `p(X|z)=N(G(z), tau^2 I)`, with fixed tau (.03/.1/.3).
In two dimensions, negative ELBO is `MSE/tau^2 + KL + log(2*pi*tau^2)`.
Training minimizes `MSE + tau^2*KL + GAN_loss + particle_spread` where applicable.
Multiplying ELBO by tau squared gives reconstruction coefficient one, matching
the AE baseline. Tau sweep changes the likelihood and KL strength relative to
GAN/spread. It is not a fixed-likelihood beta sweep. VAE-GAN denotes ELBO plus
GAN/parameter regularization, not that the combined objective itself is an ELBO.
Even no-GAN VAE retains the existing particle spread regularizer (weight one).
No-KL stochastic control is not called a VAE. AE control is deterministic.

Two iid categorical/continuous draws per training input. Pathwise derivative
for continuous samples; unbiased categorical score-function estimator with the
other draw's reconstruction cost as a leave-one-out baseline. KL categorical
is summed analytically, local KL analytic. No relaxed categorical forward or
straight-through VAE estimator. Router distances differentiate through learned
particle positions. No best-particle reconstruction search or 400-way decoding.
Routing still costs O(batch*particles*latent_dimension); sampled reconstruction
decodes two codes per input. Inference samples actual particles.

AE control preserves old nearest-forward/global-soft-backward routing at
summed-distance temperature .25, and bounded offset. VAE uses categorical
sampling at .0025 or .025. Thus AE versus VAE changes routing estimator and
posterior together; stochastic no-KL control isolates KL at sharp/tau=.1.
Both VAE temperature settings are fixed, not annealed.

## Matched training

K400, latent2, width128, batch256, 6000 updates, fixed sigma_rel .025.
Same G/D/prior and encoder (six-output head) construction for all arms, checked
by initialization hash. The extra zero head does not change initial AE function.
Encoder query has the existing toy spatial skip `X/sqrt(8.25+.03^2)+h(X)`.
This is useful toy inductive bias, not an image scalability result.
G/E Adam .0006, D .0009, prior .006; betas (0,.999), prior (.5,.999).
Rp GAN, existing lazy bcap penalty every four updates, existing particle spread.
Independent seeded training data, prior, posterior RNGs; data and prior draws
consumed identically even in no-GAN control. No training RNG used by evaluation.
No exact resume; pipeline archives failed attempts and restarts.

## Queue

Controls: GAN, deterministic AE-GAN, categorical stochastic AE-GAN without KL.
Main: categorical VAE-GAN at two temperatures times three likelihood scales.
Ablations: categorical VAE without GAN at sharp/.1; local VAE-GAN at sharp/.03
and sharp/.1. Total 12. All arms were chosen before pilot/full results.

## Evaluation and ranking

Generation: 20k at 2k/4k, 100k at 6k. Standard decoder means `G(z)` for all arms,
coverage (>=10 HQ samples/mode), HQ (distance <=.09), width/real, balance,
and sliced W1 (8192 samples). Rank by coverage, then HQ, then SW1, as before;
this ordering does not establish a universal best model. Also evaluate the
actual Gaussian likelihood samples `G(z)+tau*noise` separately, because the
ELBO describes that distribution. Large tau can mask poor likelihood fit if
one only evaluates decoder means. No held-out evaluation drives training.

Reconstruction: 8192 independently generated held-out inputs, eight posterior
draws/input. Report expected sampled MSE, MAP-particle/mean-offset reconstruction,
p99 MSE, shuffled whole code and particle-center ablations. Pair RMS is the
square root of average squared Euclidean output difference over all 28 draw
pairs, and same-mode fraction checks whether variation retains the input mode.
AE draws are repeated deterministic reconstructions, so pair RMS is zero.
Report conditional effective particle count, aggregate usage, categorical and
local KL, categorical mutual information estimate, local posterior std/prior
std, and negative ELBO in nats. The latter is comparable as a likelihood bound
but changes likelihood model when tau changes, and uses eight MC draws.
Aggregate usage/MI use analytic categorical probabilities over held-out inputs.
No claim of semantic variation follows from distinct floating point outputs.

Full configs, source archives, per-input numerical arrays, interval checkpoints,
flushed logs, and pipeline completion certificates retained. Analyzer verifies
certificates/configs, initialization, matching data/prior RNGs, fixed sigma,
checkpoint hashes and saved reconstruction array means before leaderboard.

Variational foundation: [Kingma & Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
The categorical-family restriction and two-draw estimator here follow directly
from ELBO decomposition and the score-function identity; no novelty claim.
