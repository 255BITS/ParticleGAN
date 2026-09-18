# Particle AE-GAN, VAE-GAN and AE-DDGAN

These are composable encodings and recipes, not trainers. You supply E, G, D,
data, optimizers, EMA and the loop. The package never adds a training loss,
backpropagates or steps an optimizer for you. The core dependency remains Torch.

## Three recipes

```text
AE-GAN:   E(X) -> (query, offset) -> particle[k] + sigma * bounded_offset -> G -> X_hat
VAE-GAN:  E(X) -> query -> particle[k] + sigma * fresh_noise              -> G -> X_hat
AE-DDGAN: E(X) -> (query, offset) -> z_X; G(z_X, X_t, t) -> predicted clean X
```

`k` is the nearest particle. Particle centers and networks are learned; the
MoG prior's sigma is calibrated once and remains fixed. AE bounds the raw
encoder offset as `3*tanh(offset/3)`. VAE uses independent standard Gaussian
noise. Both hard choices use a **biased straight-through query gradient**:
exact selected center in the forward pass, a distance-softmax surrogate in
backward. Center gradients pass through the selected read; standardization
can couple the raw table rows. This is not unbiased differentiation of a
categorical choice.

| `get_recipe(name)` | Encoding | Study defaults |
| --- | --- | --- |
| `ae_gan` | Deterministic bounded offset | Toy: K=400, latent=2, 6k updates, batch=256 |
| `vae_gan` | One selected particle plus Gaussian noise | Same toy budget; constant joint KL |
| `ae_ddgan` | Deterministic AE with a DDGAN generator | CIFAR: K=1024, latent=64, 10k updates, batch=64 |

All three use sigma_rel=.025, G/E LR=.0003, D LR=.00045, prior LR=.003,
prior Adam betas=(.5,.999), G/E/D betas=(0,.999), and constant LR. Penalty is
b-cap with lazy interval 4. Toy distance reduction is sum, temperature .25;
image distance reduction is mean, temperature .125. `reconstruction_weight=1`
and `observation_sigma=.03` are explicit settings for caller use, not automatic
losses. The observation sigma is distinct from latent prior sigma.

These presets expose component settings, not complete reproductions. The toy
encoder used a spatial skip; the image encoder used layer normalization of its
query. Architectures and that preprocessing remain caller-owned. Toy results
used online weights, while CIFAR evaluations used EMA=.995. Existing GAN,
MoG and DDGAN recipes retain their defaults. Configuration roundtrips through
`recipe.to_dict()` and `Recipe(**config)`.

## A runnable reconstruction step

This small example exercises the installed API. It is not a new benchmark or
a full adversarial training loop. Change the name to `ae_gan` for deterministic
reconstruction.

```python
import torch
from torch import nn
from particlegan import get_recipe

recipe = get_recipe("vae_gan", num_particles=32)
prior = recipe.make_prior()
E = nn.Linear(2, 2 * recipe.z_dim)
G = nn.Linear(recipe.z_dim, 2)
D = nn.Linear(2, 1)
opt_g, opt_d = recipe.make_optimizers(G, D, prior, encoder=E)
x = torch.randn(8, 2)
query, offset = E(x).chunk(2, dim=1)
encoded = recipe.encode(query, prior,
                        offset=offset if recipe.encoder_mode == "ae" else None)
# codes: [batch, draws, latent_dim]; AE has one draw, VAE defaults to two.
x_hat = G(encoded.codes)
reconstruction = encoded.reconstruction_loss(x_hat, x)
opt_g.zero_grad()
(recipe.reconstruction_weight * reconstruction).backward()
opt_g.step()
```

A real encoder can have a query head alone for VAE. `make_optimizers(...,
encoder=E)` adds E to the G optimizer at G's learning rate and deduplicates
parameters shared with G or the prior. Move modules to the device before
constructing optimizers.

Compose the generator objective explicitly, after your discriminator update
and with its weights frozen while preserving gradients through its inputs:

```python
encoded = recipe.encode(query, prior)  # VAE; recompute query with E after updates
x_hat = G(encoded.codes)
z, _ = prior.sample(len(x))           # independent unconditional prior codes
adversarial = recipe.make_loss().g_loss(D(G(z)), D(x).detach())
reconstruction = encoded.reconstruction_loss(x_hat, x)
spread = recipe.make_prior_regularizer()(prior.z)
loss = adversarial + recipe.reconstruction_weight * reconstruction + spread
# Your zero_grad(), backward(), step(), logging and EMA go here.
```

No KL regularizer is needed in the default VAE objective. `reconstruction_loss`
**never adds KL**, including in the optional categorical mode.

## What makes this variational without a KL penalty?

The prior chooses a uniform particle and adds local Gaussian noise:

```text
p(k) = 1/K
p(z | k) = Normal(particle[k], sigma^2 I)
q(k | X) = one selected particle
q(z | k, X) = p(z | k)
```

The joint KL over `(k,z)` is exactly `log(K)`: the local Gaussian terms cancel,
and the one-hot categorical contribution is constant. It has no training
gradient and can be omitted from optimization. `encoded.kl` retains this
constant for reporting. This uses an established constant-KL argument, related
to [VQ-VAE](https://arxiv.org/abs/1711.00937), rather than a new KL identity.
Matching the local posterior and prior causes cancellation; fixed sigma alone
is not sufficient. A shared jointly learned sigma could also cancel. Learning
a separate posterior mean or variance generally restores a variable KL.

For an explicit Gaussian decoder likelihood `Normal(G(z), tau^2 I)`, call
`encoded.negative_elbo(x_hat, x, observation_sigma=tau)` to get mean negative
ELBO in nats, **including** joint KL and the Gaussian normalizing constant.
For D observed coordinates this is
`D*MSE/(2*tau^2) + KL + D/2*log(2*pi*tau^2)`. GAN and spread are additional
objectives. AE has no such ELBO; the helper rejects AE encodings.

A valid bound does not make the hard routing gradient unbiased. Constant KL
does not guarantee balanced aggregate particle use, calibrated uncertainty or
broad semantic variation. Randomness is within one selected particle. The
joint KL is not the KL to the marginal overlapping MoG over z alone. Samples
`G(z)` are decoder means; likelihood samples also add observation noise.

## Optional categorical posterior and explicit KL

For uncertainty over particle identity, use
`get_recipe("vae_gan", encoder_mode="categorical", routing_temperature=.0025)`
or the low-level `particle_vae(query, prior, hard=False, temperature=.0025)`.
Then `E(X) -> q(k|X) -> sample k -> particle[k] + sigma*noise`.

`encoded.kl = log(K)-H(q)` now varies with the encoder. The reconstruction
helper uses independent draws and a leave-one-out score estimator; at least
two draws are required for its training gradient. To optimize the Gaussian
ELBO, explicitly use `negative_elbo`, or add the correspondingly scaled KL:

```python
reconstruction = encoded.reconstruction_loss(x_hat, x)
kl_weight = 2 * tau**2 / x[0].numel()
variational_loss = reconstruction + kl_weight * encoded.kl.mean()
```

Omitting this variable KL is allowed, but the resulting reconstruction+GAN
objective is a stochastic AE-GAN objective, not ELBO optimization. Custom
posterior families and penalties belong to your loop.

## AE-DDGAN integration

```python
recipe = get_recipe("ae_ddgan")
# E sees clean X, not the corrupted image:
query, offset = E(x)
encoded = recipe.encode(query, prior, offset=offset)
x_hat = G(encoded.codes[:, 0], xt=xt, t=t)
reconstruction = encoded.reconstruction_loss(x_hat[:, None], x)
```

Add reconstruction to your DDGAN generator objective. Use independently
sampled prior codes for adversarial transitions and fresh prior codes at every
reverse generation step, as in the [DDGAN API example](api.md#ddgan). There is
no encoder in unconditional generation. This is deterministic AE-DDGAN, not
VAE-DDGAN. Its reconstruction is a one-step clean prediction with noisy-image
side information, not latent-only reconstruction or a full encoded reverse
chain.

## API contracts and checkpoints

`particle_ae(query, offset, prior, temperature=.25, distance_reduction="sum",
offset_bound=3)` and `particle_vae(query, prior, temperature=.25,
distance_reduction="sum", draws=2, hard=True, generator=None)` return a
`ParticleEncoding`. The opt-in soft-posterior toy configuration uses temperature .0025.
Queries and offsets are `[B,latent_dim]`, matching prior device and dtype.
The prior must be `MoGParticlePrior`. Distances use O(B*K) storage; neither
routing nor the API decodes every particle.

Fields: `codes [B,S,D]`, `indices [B,S]`, `kl [B]`, `log_probs [B,K]` (None
for AE), and `mode`. Hard VAE exposes the true one-hot log probabilities, not
the soft backward weights. An explicit generator controls both categorical
and Gaussian draws without changing global RNG. Reconstruction predictions
are `[B,S,...]`, targets `[B,...]`; flatten draws for image networks and reshape
their outputs before computing losses. No reductions run over the batch's
identity dimension accidentally.

Save your E/G/D, prior, optimizers, recipe and RNG states. The encoding object
is an ephemeral batch result, not a module with trainable state. Inference
reconstruction needs E/G/prior; unconditional generation needs only G/prior
(and the diffusion schedule for DDGAN). Use matched EMA E/G/prior if evaluating
EMA reconstruction. Calling `eval()` does not remove VAE Gaussian draws.

## Measured evidence and limits

CIFAR32, matched 10k updates, FID measured from 50k unconditional samples:

| Model | FID50k ↓ | Training seconds |
| --- | ---: | ---: |
| Direct GAN | 19.483 | 474 |
| Particle AE-GAN | 20.054 | 526 |
| AE-DDGAN | 43.233 | 639 |
| DDGAN | 49.475 | 546 |

Direct AE test reconstruction MSE was .078996. AE improved DDGAN FID by 12.6%
at 16.9% extra training cost. These are one matched trajectory, not a universal
ranking. Frozen-model noise audits confirmed distinct outputs numerically;
AE post-training jitter is not a learned variational posterior. See the
[CIFAR protocol and leaderboard](../reports/cifar-particle-ddgan/README.md).

Toy, lower-LR matched 6k updates, 100k generation samples:

| Model | Modes ↑ | HQ% ↑ | Reconstruction MSE ↓ |
| --- | ---: | ---: | ---: |
| AE-GAN | 97 | 89.71 | .002341 |
| GAN | 96 | 75.49 | — |
| VAE-GAN (default constant KL) | 88 | 85.21 | .003264 |
| Categorical VAE-GAN with KL | 86 | 82.50 | .003884 |

Default VAE pairwise output RMS was .01066; all 65,536 reconstruction draws
retained the input mode. Both VAE variants remain **toy-tested only**. GAN had
the best global sliced Wasserstein score; all models produced modes narrower
than real data and some regressed late in training. The
[full leaderboard](../reports/mog-vae/stability/LEADERBOARD.md) also includes
the stochastic no-KL control. Prefer AE for the strongest tested reconstruction;
use VAE when the explicit noisy posterior matters. A matched, budget-limited
image VAE comparison is a future experiment, not a release claim.
