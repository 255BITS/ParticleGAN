# Procedural image transfer tasks

These are development validation tasks for a generic GAN controller. The four
healthy cases affect ranking, not eligibility. Four explicitly difficult or
information-poor cases are diagnostics with zero selection weight. The existing
required suite remains separate. Failure here cannot silently turn a new task
into a required gate, and reference failure does not change its declared tier.

| Task | Tier | Factor or limitation | Steps |
| --- | --- | --- | ---: |
| img_stripes2 | ranking | Horizontal versus vertical central stripe | 600 |
| img_bars4 | ranking | Two horizontal and two vertical bar positions | 600 |
| img_blobs4 | ranking | Four corner patch positions | 600 |
| img_intensity2 | ranking | Central patch intensity 0.35 versus 0.85 | 600 |
| img_bars8 | diagnostic | Eight support modes in a short budget | 600 |
| img_tiny_generator | diagnostic | Width 2, one-dimensional generator latent | 480 |
| img_mean_discriminator | diagnostic | D sees only image mean; patch locations have identical means | 480 |
| img_uniform_generator | diagnostic | G cannot represent any nonuniform stripe | 480 |

All eight belong to `image_conv_transpose`. `img_residual_bars4` is the single
reserved `image_conv_residual_upsample` architecture family: width 16, nearest
neighbor upsampling and residual convolutions. Reference calibration left it
unevaluated; the later [frozen comparison](../../reports/transfer_suite/README.md)
reports its transfer results. The
reference CLI excludes it; the aggregate runner must freeze policy selection
before calling its episode API.

Each real image is a uniformly selected 8×8 grayscale template with independent
Gaussian pixel noise, standard deviation 0.01, clipped to [0,1]. Both networks
are unconditional: no label or template reaches them or the controller. The
standard generator maps an 8-dimensional latent through a linear layer to
12×2×2, two kernel4/stride2/padding1 transposed convolutions, then a kernel3
output convolution and sigmoid. Hidden activations are leaky ReLU 0.2. D has
kernel3/stride2/padding1 convolutions with 12 then 24 channels, leaky ReLU 0.2,
and a linear scalar head. There is no batch normalization, reconstruction
loss, template decoder or pretrained representation.

Training uses ParticleGAN's relativistic-pair logistic GAN loss, exact b_cap
input-gradient penalty (coefficient 3, kappa 1.25), and 32 learned latent particles
with ParticleRegularizer weight 0.05. G and prior share Adam; D has a separate
Adam. Both use LR 0.0017, betas (0,0.99), batch 32, and one update per step.
There is no particle L2. A feedback action runs after the current backward
and before each optimizer step; its held regularization action affects the
following loss. Fixed controls use the same models, data draws and update order.
The reference cosine holds until 60% then decays to 5% of initial LR. All GAN
initialization and training use seed 0 and one CPU thread; no seed sweeps.

Quality is nearest-template pixel RMSE≤0.10 (≤0.06 for intensity). Regions are
disjoint: every pair of templates is more than twice that radius apart. A mode
counts only if quality-qualified particles contribute at least half its uniform
target mass, so a sharp collapsed generator cannot pass coverage. Overall HQ
must be≥90%. We also report each mode's mass, mean RMSE and total variation from
uniform; total variation is descriptive, not an additional gate. Evaluation
enumerates all 32 uniformly weighted particles under an isolated RNG context.
This measures the exact finite prior, not continuous-latent coverage, noisy
texture fidelity or natural-image quality.

There are 24 equally spaced observations. Sustained success needs all modes and
HQ≥90% for at least the final five observations of a complete curve. Transient
passes and truncated runs cannot qualify. EMA 0.99 applies separately to both
G and prior and is reported separately; only live weights determine success.
The impossible uniform-generator and information-poor discriminator cases are
intentional diagnostics, not controller eligibility tests.

Run a reference calibration (fresh output directory or unrun task names):

```bash
python -m benchmarks.transfer_suite.image_tasks \
  --output /tmp/pr36-transfer-images-v1 --schedule cosine
```

The CLI writes the full spec declaration and source/runtime fingerprints before
training, then one JSON per attempt with all 24 measurements, live/EMA metrics,
loss checkpoints, action traces, timings and errors. It refuses to overwrite
results or mix changed source/spec declarations. The first stripe reference took
7.95 CPU seconds. The reference evidence below includes every attempt; no learned policy was fitted.


## Reference evidence

Thresholds, budgets and tiers were declared before these runs and remain unchanged.
Both ordinary references sustain **one of four ranking tasks** (blobs4), confirming
a solvable healthy case. No diagnostic case sustains. A sharp but severely
imbalanced stripe generator fails coverage even at HQ 100%. The other healthy
cases remain useful unresolved ranking challenges; they are not required gates.

Each cell is **live modes / total · HQ · sustained confirmation step**; a dash
means no qualifying final five-observation suffix.

| Task | Tier | Cosine | Constant |
| --- | --- | --- | --- |
| img_stripes2 | ranking | 1/2 · 100.0% · — | 1/2 · 100.0% · — |
| img_bars4 | ranking | 2/4 · 100.0% · — | 1/4 · 56.2% · — |
| img_blobs4 | ranking | 4/4 · 96.9% · 525 | 4/4 · 96.9% · 525 |
| img_intensity2 | ranking | 0/2 · 0.0% · — | 0/2 · 0.0% · — |
| img_bars8 | diagnostic | 5/8 · 75.0% · — | 5/8 · 75.0% · — |
| img_tiny_generator | diagnostic | 1/4 · 37.5% · — | 1/4 · 53.1% · — |
| img_mean_discriminator | diagnostic | 0/4 · 0.0% · — | 0/4 · 0.0% · — |
| img_uniform_generator | diagnostic | 0/2 · 0.0% · — | 0/2 · 0.0% · — |

Two additional fixed cosine rate ratios were declared before those eight runs:
G-half uses G 0.00085 / D 0.0017; D-half uses G 0.0017 / D 0.00085. These calibration-only
variants retain the same budgets, thresholds and data. Neither adds a sustained
pass. D-half improves intensity coverage to 2/2 but HQ 81.25% still fails. These
attempts do not change the task defaults or establish a controller winner.

| Ranking task | G-half | D-half |
| --- | --- | --- |
| img_stripes2 | 1/2 · 96.9% · — | 0/2 · 0.0% · — |
| img_bars4 | 2/4 · 96.9% · — | 1/4 · 25.0% · — |
| img_blobs4 | 1/4 · 18.8% · — | 2/4 · 50.0% · — |
| img_intensity2 | 0/2 · 0.0% · — | 2/2 · 81.2% · — |

All 24 episodes finished with complete curves and no numerical errors, using
125.8 total CPU wall seconds (single observations, not speed estimates).
The reserved family has zero evaluations in these calibration runs. Six contract tests cover disjoint
quality regions, collapse/blend rejection, deliberate architecture limitations,
RNG isolation, independent EMA, exact fixed/zero-feedback numerical parity,
post-backward action timing, and incomplete/transient convergence rejection.

Original JSON and logs are in `/tmp/pr36-transfer-images-v1/`,
`/tmp/pr36-transfer-images-ratios/`, and their matching `.log` files. The handoff
bundle `/tmp/pr36-transfer-images-artifacts/` contains deterministic `.json.gz`
files, original-byte SHA256s in `manifest.json`, and exact code plus calibration
scripts in `source.tar.gz`. Parent suite reports archive those machine records;
all failures are retained.
