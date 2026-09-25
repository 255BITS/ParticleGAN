# Replace paired reconstruction with the sliders error game

The user requested the [Anima concept-sliders formulation](https://huggingface.co/ntc-ai/anima-concept-sliders),
with a configurable scope, and selected **all heads** for this experiment.
`slider_scope: all` replaces action MSE and both state heads' MSE/contact BCE;
`slider_scope: action` replaces only action MSE. The latter is implemented and
tested, but is not a second research run.

Keep the previous-action model and full-label dataset:

```text
E(st, at-1, terrain) -> z -> G1 -> st
                         -> G2 -> at
                         -> G3 -> st+1

prior -> z -> G1 / G2 / G3
```

All Gs, E, prior, and discriminators start from scratch. The 1,024-particle MoG,
z32, width 128 networks, paired minibatch streams, seed24003, 2,500 updates,
batch 256, and GPU1 match the preceding L2 run. Joint D, action D, and shared
state D retain their previous adversarial objectives. No simulator calls or
trajectory unrolling occur in training. No extra expert data is collected.

## Paired-error adaptation

Let y be the normalized real 18-coordinate transition target and y_hat its
control-encoded prediction. Convert the four contact logits to probabilities
for this residual; the existing joint/marginal GAN paths still use sampled
binary contacts with straight-through gradients.

The source has frozen neutral/positive diffusion predictions. Here the fixed
neutral baseline is the training target mean mu, so delta_T = y - mu and
delta_S = y_hat - mu. This baseline cancels from their difference. Fit S only
on training records: per-coordinate sample std(delta_T), floored at 1e-4,
multiplied by median row RMS(delta_T / std). Thus median normalized target-edit
RMS is one. There is one normalization group, not diffusion-position groups.

```text
error = (y_hat - y) / S
real_error = sigma(k) * epsilon
fake_error = same sigma(k) * epsilon + error
epsilon ~ Normal(0, I)

L_R = mean softplus(R(fake_error) - R(real_error)) + lazy gradient cap
L_paired = mean softplus(R(real_error) - R(fake_error))
```

Use the unmodified upstream global-mix critic: eight width 48 tokens projected
from the full error vector, one attention layer with four heads, and a bounded
8*tanh(score/8) output. Tokens are learned mixtures of coordinates, not time steps.
R has no state, action-history, terrain, or target input beyond the noisy residual.
G does not receive epsilon. Real/fake share noise within a pair; D and G noise
streams are separate and independent of existing training streams.

The exact autograd input-gradient cap applies every fourth update, threshold 1,
coefficient 1, with multiplier 4. Noise begins at RMS(delta_T/S)/0.28 and follows
the source's geometric curve toward 0.03, with absolute hold 1.0. Use update k-1
as the schedule index and adapt its horizon to our 2,500 updates. For a start
above1, the hold prevents the actual noise from reaching0.03.

For `all`, optimize existing joint/marginal GAN loss + L_paired (weight1) +
the existing full-table MoG prior regularizer. No MSE/BCE optimization remains;
these values are logged only as diagnostics. The error critic has its own Adam
optimizer at 0.0009, with the repository's cosine schedule. Other optimizers,
EMA0.995, and the MoG routing/prior regularizer stay unchanged. We borrow the
paired-error formulation, not Anima's LoRA architecture, diffusion sampler,
1600-step optimizer recipe, or 64-particle-subset regularizer.

The MIT reference is vendored unmodified from
[core revision beaffeb](https://github.com/mikkel/sliders-conceptmod/tree/beaffeb3640c4554a7315998c04a5909f384b972/packages/concept-slider-core).
Its hash, origin, provenance, and license are stored alongside it. No Anima
weights or runtime dependencies are needed.

## Frozen comparison

Compare with the completed previous-action L2 run without retraining it. Verify
identical initial G/E/prior/joint/marginal parameters and real minibatch streams.
Reselect both models' saved updates 250/1000/2500 on the same 20 fresh validation
worlds 1391000–1391019, by landing rate, mean return, then earlier step. Score
the selected models on 50 fresh paired test worlds 1491000–1491049. Also score
the new final checkpoint if distinct. Test scores never select checkpoints.

Reevaluate the existing state-only joint GAN as a GAN reference and the original
imitation fine-tune as a separate non-GAN reference. The latter cannot rank on
the GAN-only board or choose its default. Existing references have different
data/training histories. Even the matched sliders/L2 comparison changes paired
supervision and adds critic capacity; it is not an equal-compute comparison.

Report landing counts/intervals, mean and paired returns, failures, offline
prediction metrics, critic/noise behavior, and compute. Run one full experiment,
with no seed repeats. Progress:

```bash
tail -F results/gym/lunar_lander_slider_gan/live.log
```
