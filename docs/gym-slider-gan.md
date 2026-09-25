# Slider-error supervision for Lunar Lander

The completed all-heads run landed **6/50**, versus **11/50** for the matched L2
GAN on the same fresh test worlds. The existing joint GAN landed 37/50 and remains
the validation-selected GAN baseline. See the
[results and diagnostics](../reports/gym/lunar_lander_slider_gan/READOUT.md).

This experiment replaces the previous-action GAN's paired reconstruction losses
with the [Anima concept-sliders paired-error game](https://huggingface.co/ntc-ai/anima-concept-sliders#the-paired-error-game).
The graph remains:

```text
E(st, at-1, terrain) -> z -> G1 -> st
                         -> G2 -> at
                         -> G3 -> st+1

prior -> z -> G1 / G2 / G3
playback: E(st, at-1, terrain) -> z -> G2 -> at -> actual simulator
```

An extra critic R compares Gaussian noise against the **same noise plus paired
prediction error**. G/E learn to remove that error through the critic. Existing
joint, action, and shared state discriminators continue training on generated
transitions. The MoG prior regularizer is retained once per generator update.

```text
error = (encoded prediction - paired target) / training-only scale
real_R = sigma * noise
fake_R = same sigma * noise + error

R loss = mean softplus(R(fake_R) - R(real_R)) + gradient cap
paired G/E loss = mean softplus(R(real_R) - R(fake_R))
```

The configuration controls which paired losses it replaces:

| `slider_scope` | Paired-error coordinates | Remaining paired reconstruction losses |
| --- | --- | --- |
| `all` (this run) | All 18 state/action/successor coordinates | None: neither MSE nor contact BCE |
| `action` | Two action coordinates | G1/G3 continuous MSE and contact BCE |

For `all`, contact logits become probabilities inside the error vector. The
other GAN critics still see sampled binary contacts. Logged MSE/BCE values are
diagnostics with no contribution to the all-heads optimization graph.

Normalization uses a fixed training-mean baseline in place of the source's
neutral diffusion teacher. Scale is per-coordinate sample standard deviation,
with a median-row-RMS adjustment; only the 9,297 expert training triples fit it.
The pinned source critic mixes the error into eight width 48 tokens, applies one
four-head attention layer, and bounds its score at 8. These tokens are mixtures
of coordinates, not trajectory steps. Noise follows the source's geometric
schedule with absolute hold 1.0, adapted to our 2,500-update budget. The exact
gradient cap runs every fourth update with the lazy multiplier 4.

We reuse the unmodified MIT
[shared reference implementation](https://github.com/mikkel/sliders-conceptmod/tree/beaffeb3640c4554a7315998c04a5909f384b972/packages/concept-slider-core),
including its license and provenance. We keep ParticleGAN's existing MoG router,
1,024 particles, optimizers, full-table prior regularizer, and EMA. This adapts
the error game; it does not install Anima, use its weights, or reproduce its
adapter/diffusion architecture. See the [frozen plan](gym-slider-gan-plan.md).

The [leaderboard](../reports/gym/lunar_lander_slider_gan/README.md) compares with
the saved previous-action L2 run using identical initial model weights and
training minibatch streams. Both sets of saved checkpoints are reselected on
the same fresh validation worlds. The existing joint GAN is another reference;
the imitation fine-tune remains a separate non-GAN reference. No L2 retraining
or seed-only repeats are needed.

Run the all-heads recipe:

```bash
.venv/bin/python -u experiments/evaluate_gym_slider_gan.py --freeze
.venv/bin/python -u experiments/train_gym_slider_gan.py \
  --config configs/gym/lunar_lander_slider_gan/sliders_all.yaml
.venv/bin/python -u experiments/evaluate_gym_slider_gan.py \
  --run-dir results/gym/lunar_lander_slider_gan/sliders_all
.venv/bin/python -u experiments/diagnose_gym_slider_gan.py
tail -F results/gym/lunar_lander_slider_gan/live.log
```

Use a fresh training output directory. `--slider-scope action` overrides the
scope for a future experiment; it needs a separate frozen evaluation protocol
because this report pins `all`. Both scopes have correctness tests; only `all`
is a completed-budget research run in this comparison. GPU experiments use GPU1.

Training records, checkpoints, and logs stay under ignored `results/gym/`.
Reports archive sources, normalizations, losses, hashes, and evaluation traces.
Playback still runs E/G2/prior; no discriminator is needed to choose actions.
