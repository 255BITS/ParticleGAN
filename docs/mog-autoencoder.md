# MoG particles with an encoder, without per-example KL

Design discussion, 2026-09-17. The original proposal below is retained for context.
The follow-up routing scout is implemented in `experiments/train_mog_autoencoder.py`;
its protocol and results live in `reports/mog-autoencoder/README.md`.
Branch: `feature/mog-autoencoder`, based on `47e504c`.

The proposal is a deterministic autoencoder with a learned MoG prior and a
data-space GAN loss. The research question is whether reconstruction and local
invertibility help particles cover data modes and represent within-mode variation.
Removing KL alone is already well explored; novelty is not established here.

## Separate three choices

A common Gaussian VAE uses:

```text
E(X) -> (mu_X, sigma_X)
z = mu_X + sigma_X * noise
G(z) -> reconstruction
loss = reconstruction + KL(q(z|X), prior)
```

The Gaussian family makes a cloud around each example. Reparameterization makes
sampling differentiable. KL is a separate training objective, not a geometric
ball or a prerequisite for differentiable sampling. A diagonal Gaussian is
generally an ellipsoid, and a VAE need not use Gaussian posteriors.

In the current repository, `MoGParticlePrior` already uses:

```text
k ~ Uniform(1..K)
z = p[k] + s * noise
G(z) -> generated sample
```

Here `p = prior.means()`, `noise ~ N(0,I)`, and `s = prior.sigma` is a shared
fixed scalar calibrated at initialization. The defaults are K=400 and dimension
4. Standardized particle reads remain differentiable; the existing spread loss
acts on raw `prior.z`. There is no per-example encoder or KL in this sampler.
We can keep this sampling reparameterization while discarding Gaussian
per-example posteriors entirely.

## First formulation: match collections of points

```text
Reconstruct: X -> E -> z_X -> G -> X_hat
Generate:    (k, noise) -> p[k] + s*noise -> G -> X_new
Match:       collection of z_X <-> collection of sampled MoG points
Judge:       D(X) versus D(X_new)
```

`E(X)` produces a single point. It predicts neither variance nor particle index.
The aggregate encoder distribution means the distribution of `E(X)` when X
ranges over real data. Individual examples need not each resemble the prior.

```text
L_EGP = a * mean(||G(E(X)) - X||^2)
      + b * SW2_squared({E(X)}, {p[k] + s*noise})
      + c * generator_GAN_loss(D(G(p[k] + s*noise)))
      + d * existing_particle_spread_loss(raw_particles)
```

Train D separately with real data, detached generated samples, and the existing
critic penalty. E receives reconstruction and distribution-matching gradients;
G receives reconstruction and GAN gradients; particles receive matching, GAN,
and spread gradients. This is an autoencoder objective, not a claimed ELBO or
calibrated posterior-inference procedure.

Sliced Wasserstein matching can be implemented with equal-sized batches: project
each cloud onto random unit directions, sort each pair of projections, average
squared differences of corresponding sorted values. It requires no density or
latent discriminator. A finite set of projections is only an approximation;
evaluate held-out matching with fresh directions and larger samples.

Matching only nearest particle centers would miss the Gaussian neighborhoods.
Matching only means and covariance would miss multimodal distribution shape.
In the ideal population case, exact reconstruction plus exact aggregate matching
implies correct generation: G maps the common latent distribution to the data
distribution. Finite losses and finite networks do not provide this guarantee.

## Particle-specific hypothesis: preserve local offsets

View a particle as a local coordinate origin:

```text
E(X) -> z
z = p[k] + s*u              # a possible component-relative description
G(p[k] + s*u) -> X_hat

p[k] + s*noise -> G -> X_new -> E -> recovered latent
```

The decomposition does not itself constrain E: an arbitrary residual can hide
everything. The baseline therefore matches the full noisy mixture directly,
without introducing a hard assignment or a residual head.

A useful second arm adds a local round-trip objective:

```text
z0 = stop_gradient(p[k])
z1 = stop_gradient(p[k] + s*noise)
center_error = ||E(G(z0)) - z0||^2
offset_error = ||(E(G(z1)) - E(G(z0)))/s - noise||^2
```

Use positive fixed s. Stopping gradients at the sampled targets makes this
auxiliary objective train E and G without directly moving particles to simplify
the targets. Other losses still train the particles. Start with a small weight;
the normalization exposes errors hidden by tiny s but can amplify gradients.

The hypothesis: G should use the neighborhood around each particle, and E should
recover that variation. On the 100-Gaussian task, does this improve mode width
without creating bridges? This is a particle-focused use of cycle consistency,
not an established new algorithm. It does not guarantee semantic coordinates;
E and G can encode imperceptible details or adopt complicated coordinates.

If an explicit `E(X) -> (k,u)` model is tried later, match the JOINT distribution
of (k,u) to uniform k with independent Gaussian u. Uniform component counts plus
a Gaussian pooled residual are insufficient: each component could have a very
different conditional residual distribution. Hard routing also needs an explicit
gradient estimator or alternating assignment scheme. Overlapping Gaussians make
the generating component unidentifiable from z, so requiring exact recovery of
the original k can impose an impossible target.

## Constraints that matter for this repository

- The toy observations are 2D but the default latent is 4D. A regular deterministic
  encoder from 2D cannot exactly fill a full-dimensional 4D MoG. A differentiable
  4D-to-2D decoder cannot have a local inverse in every latent direction either.
  Start all compared arms at latent dimension 2; retain a separate dimension-4
  control only if investigating this mismatch explicitly.
- Fixed s prevents variance from shrinking as an easy escape, but means can
  still coincide or move closer. Standardization and spread regularization do
  not guarantee distinct or well-used components. Monitor nearest-neighbor
  spacing, effective overlap, and per-component outputs.
- A nearest-center encoder assignment is not a true mixture-component label.
  Measure generated components using their known sampled IDs; report real-data
  responsibilities separately. Uniform prior weights need not imply one particle
  per semantic mode. Unequal data frequencies may need multiple particles per
  common mode or a later learned-weight model.
- Do not use a soft average of particle centers as if it were a MoG sample.
  Such averages can fall between components where generation rarely samples.
- Reconstruction helps represent real examples, but does not by itself establish
  good unconditional generation or rule out memorization.

## Initial comparison, if we proceed to experiments

These are proposed arms, not a results leaderboard. Use one shared seed; change
the mechanism, not the seed. Fix dimensions, G/D widths, K, s calibration,
training budget and evaluation samples across applicable arms. Report wall time
alongside steps because encoder and matching computations add cost.

| Priority | Arm | Question |
| --- | --- | --- |
| 1 | Existing MoG GAN, dimension 2 | Matched generative control |
| 2 | Add E + reconstruction + aggregate SW matching | Does inference improve coverage and sampling? |
| 3 | Add local center/offset round trips | Do neighborhoods preserve useful variation? |
| 4 | Freeze particle means in arm 2 | Does adapting the prior actually help? |

Use fresh toy samples for held-out reconstruction. Evaluate unconditional mode
coverage, high-quality fraction, balance, mode width and component purity with
the existing MoG metrics; also record reconstruction error, held-out latent
matching, and normalized offset recovery. The existing `kl_balance` is an
evaluation statistic over data-mode frequencies, not a VAE training KL.

After completion, publish a leaderboard with raw metrics, explanation of failures,
and next-step recommendations. Do not rank on reconstruction alone. If implemented,
write flushed `log.txt` and structured metrics in each run directory, document a
literal `tail -F` command, and keep evaluation RNG separate from training RNG.
No experiments were launched during the initial design discussion. The later
routing scout directly tests particle selection with random, predicted, bounded
predicted, or zero offsets; it does not yet implement the SW matching or local round trips
proposed above.
An additional control amplifies the predicted-offset gradient by 100 while
preserving its forward values and fixed sigma; the report records its results.
A further bounded-offset control penalizes aggregate hard particle-usage
imbalance with a soft routing gradient. A saved-checkpoint audit compares hard
and soft usage on 100,000 examples; see the report for the observed mismatch.
Two matched follow-ups restrict the backward routing surrogate to eight nearest
particles with a detached local bandwidth, with and without the same usage loss.
The hard forward path and fixed sigma are unchanged; the prespecified comparison
is recorded in `reports/mog-autoencoder/local_protocol.md`.
A frozen-checkpoint audit then searches all 400 decoded centers to isolate the
zero-offset reconstruction error from encoder selection. It finds substantial
selection gaps while the original bounded arm retains the best center set;
see `reports/mog-autoencoder/ORACLE.md` for the diagnostic leaderboard.

## Prior art and novelty boundary

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): stochastic
  variational inference and the reparameterized objective.
- [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644): match the aggregate
  encoder distribution to a prior using an adversary.
- [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558): reconstruction
  with aggregate distribution regularization.
- [Sliced-Wasserstein Autoencoder](https://arxiv.org/abs/1804.01947): use sliced
  Wasserstein matching for a generative autoencoder.
- [FlexAE](https://proceedings.mlr.press/v161/mondal21a.html): learned flexible
  priors and the importance of latent dimension.
- [Shape your Space](https://proceedings.neurips.cc/paper/2021/hash/3c057cb2b41f22c0e740974d7a428918-Abstract.html):
  Gaussian mixture regularization for deterministic autoencoders; a particularly
  close reference to examine before making novelty claims.
- [VQ-VAE](https://arxiv.org/abs/1711.00937): discrete codebook alternative;
  a single hard code with a deterministic decoder has only K possible outputs.

Recommendation: implement the deterministic MoG autoencoder baseline first;
then test whether local offset recovery adds measurable value. The defensible
research claim must come from that comparison, not simply from removing KL.
