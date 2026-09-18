# Conditional variation without retraining or visual inspection

**Yes: the trained decoder produces numerically distinct variations around an
encoded image. A moderate noise level changes the output while mostly retaining
its proximity to that input's deterministic reconstruction.** This does not
establish distinct semantic content or a learned VAE posterior.

We used the final bounded model's EMA encoder, generator and particle table.
512 held-out CIFAR test inputs, eight draws per input, all 28 within-input pairs,
six sampling conditions. Total evaluation time **29.8 seconds**, peak allocated
memory **1.76 GiB**. No images were saved or visually inspected for this audit.

```text
E(X) -> (k, offset)
z_X = p[k] + fixed_sigma * bounded(offset)
fresh_noise -> G(z_X + a * fixed_sigma * fresh_noise)
```

Sigma remains 0.212616. `a` is an inference-time noise multiplier, not a learned
parameter. Newly perturbed codes are not clipped. One common tensor of Gaussian
draws is reused across noise strengths. This is a Monte Carlo evaluation of one
fixed network, not a training-seed sweep.

| Noise multiplier a | Distinct outputs /8 | Feature variation | Input MSE increase | Own reconstruction remains nearest |
|---|---:|---:|---:|---:|
| 0 | 1 | 0% | 0% | 100% |
| 0.25 | 8 | 9.6% | +1.5% | 99.98% |
| 0.5 | 8 | 23.1% | +5.8% | 99.34% |
| 1 | 8 | 45.4% | +22.6% | 82.23% |
| 2 | 8 | 72.8% | +80.7% | 28.27% |

Feature variation is the pairwise cosine distance between normalized Inception
pool2048 features, relative to the distance between **unrelated deterministic
reconstructions**. A value of 100% means equal average feature distance to that
reference; it does not mean 100% of images or semantic content changed. The
extractor is independent of the ResNet18 discriminator used in training.

At a=0.5, output pairs differ by RMS **11.07 pixel levels out of 255**.
Feature distance is 0.06833, versus 0.29565 between unrelated reconstructions
and 0.40414 between unrelated real images. Exact pixel duplicates are absent
for every nonzero condition, for all 512 inputs. Zero-noise outputs are exactly
identical after uint8 quantization, as expected for this deterministic encoder.

The nearest-reconstruction metric searches the 512 deterministic decoded
references in feature space. At a=0.5, 99.34% of individual draws retrieve their
own reference, and 95.31% of inputs retain it for all eight draws. These are
descriptive proximity metrics, not class/identity accuracy or proof that fine
details are preserved. The base reconstructions were already blurry in the
preceding experiment; this audit did not reassess them visually.

## A different question: sampling from the selected particle

```text
E(X) -> k
G(p[k] + fixed_sigma * fresh_noise)
```

This also produces eight distinct outputs, with feature variation 70.2% of the
unrelated-reconstruction reference. But reconstruction MSE becomes **0.36786**,
versus **0.06296** for the deterministic encoding: a **484.3% increase**. Only
0.49% of draws retrieve their input's original reconstruction. Selecting a
particle and replacing the learned offset is therefore a poor way to retain
the input in this trained model.

The feature distance to the original real image actually decreases in this
condition despite its much worse pixel fidelity (0.383 versus 0.432). This
illustrates why a single feature metric cannot certify input preservation or
semantic correctness. The full results expose both pixel and feature metrics.

## Recommendation and scope

Use **a=0.5 as a starting point** for moderate variation; a=0.25 is a more
conservative option. These choices follow this measured tradeoff, not a tuned
optimum or a statistical guarantee. Larger noise should not be interpreted as
automatically better diversity.

This confirms a useful inference-time knob for the existing network. The
encoder itself is still deterministic, the noise distribution is externally
chosen, and no stochastic posterior was trained. It does not resolve the late
GAN-training regression or establish a general VAE-like semantic variation
guarantee. No further training is necessary to use this sampling rule.

## Artifacts and checks

- [Full metric table](LEADERBOARD.md), [raw summary](summary.json),
  [protocol](protocol.json), [per-input measurements](per_input_metrics.npz).
- Measurement script: `experiments/measure_cifar_particle_variation.py`.
  Table/verification script: `experiments/analyze_cifar_particle_variation.py`.
- Log: `runs/cifar_particle_ae/variation.log`. The measurement refuses to
  overwrite an existing output directory.
- Two known-answer tests pass for pairwise distances and duplicate counting.
  The deterministic MSE reproduces the saved first-512 reconstruction errors
  within 1e-5; all reported means verify against per-input arrays using float32
  reduction tolerance. Full model state and checkpoint hashes are unchanged.
- These are results for one trained checkpoint and one fixed held-out subset.
  No seed sweep, retraining, generated-image FID, or visual judgment was added.
