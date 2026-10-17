The generated caption toy retains an untied-head accuracy win after matching
game normalization and cross-time latent correlation. Its frozen random host
still has much smaller parameter magnitudes than the retained pretrained host.
This separate public-API variant changes only the fresh frozen backbone's
per-tensor scalar initialization moments. It preserves the source-caption
student and positive-caption teacher through the same frozen nonlinear host.

Run the complete asset-free numerical test from the repository root:

```sh
CUDA_VISIBLE_DEVICES=0 python -m examples.e22_routed_caption_frozen_stats --run --out runs/caption-frozen-stats-v1
```

The command uses only the bundled scalar profile and public ParticleGAN API.
Historical `/ml2` paths in provenance are not runtime dependencies. It trains
ordinary BF16 LoRA, shared-Up BF16 particles and untied-Up BF16 particles for
512 updates each. Exit 0 means completed scientific PASS, 1 completed FAIL,
and 2 incomplete/error. Imported package identity is recorded and checked
within the run; compatible future APIs are not permanently allowlisted.

The profile embeds all 29 full shapes, source-to-toy names, float64 means and
population standard deviations from one immutable initial pure-base capsule.
The original distribution families remain:

```text
Linear weight/bias: Uniform(μ − √3 σ, μ + √3 σ)
Position:          Normal(μ, σ)
```

Fresh trainable parameters receive call-local public `initialize_` overrides
using the original full-name CPU streams, then freeze before disjoint
student/live-teacher copies and learned adapter initialization. The distribution
registry and learned Down/H/C/Up/query/critic/prior initialization stay unchanged.
No pretrained tensors, learned directions, spectra or covariance are copied.
Realized sample moments are descriptive; no correction forces exact moments,
actual token-mean power, scales or accuracy. Initial raw/normalized token-mean
and centered power and fixed-owner immutability are recorded before training.

Flow/caption inputs, six adapted projections, geometry, shared 128×4 particle
bank, sampled C with H/b/Up zeros, native D/G DV12, RpGAN/KA2 and feature-only row
guards are inherited unchanged. The untrained FIT coordinate-std normalization
rule retains correction 1 and the fixed `1e-8` nondegeneracy refusal; numerical
scales are recomputed consequences of the changed frozen host. Physical paired
TEST48 RMSE remains an offline metric and never enters training or row decisions.

The terminal gate requires untied particles to improve aggregate RMSE by at
least 0.1% against both controls, harm no source by more than `1e-6`, benefit
from codes by at least 0.1% overall and strictly on each of six sources, and
retain bank/router gradients on at least 90% of 511 post-first updates. All six
C and particle-Up norms must be finite positive. There is no historical-failure
requirement. PASS would reject this scalar calibration as sufficient to reproduce
the remaining transfer failure in this fixed generated fixture; FAIL would
preserve failed bounds without proving a unique layer or direction cause.

One physical GPU0 campaign has a 300-second startup-through-final-write budget,
including data construction, public ownership/restore/tangent prerequisites,
training and all captures. It adds no old-host observations. Render actual
retained clean states at fixed updates 0/64/128/256/384/512 separately on CPU:

```sh
python -m examples.render_e22_routed_caption_frozen_stats --run-directory runs/caption-frozen-stats-v1
```

Rendering has a 60-second budget and uses optional NumPy/Pillow>=10.1. The goal
GIF compares target zero and three actual residual maps on fixed six-source
cameras, with an initial-only physical-error color scale. Full TEST48 determines
the gate. Outputs and media paths are exclusive; raw arrays/logs stay local.

NEW CPU software prerequisites use reduced geometry only, with at most four
tiny public native updates for a 2+2 fresh-owner replay. They check distribution
oracles/destructive profile controls, shape/stream/Parameter/freeze ownership,
paired-host immutability, fixed normalization, public restore/tangent, unchanged
accuracy/source/code gates and raw media controls. Earlier tests are not rerun.
The full actual-caption and original full-Supra failures remain unresolved.
