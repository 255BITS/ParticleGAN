# Late-checkpoint stability of saved 100-Gaussian runs

This is a read-only audit of existing `grid100` samples. It compares the
lower-rate MLP run at the local ignored path `artifacts/toy100-accuracy/search-agent/batch2048/noise029_batch2048_floor005/grid100`,
the shared-v3 warmup MLP run at `artifacts/toy100-accuracy/search-agent/warmup-screen/warm_b2048_f2_g128_d3/warm_b2048_f2_g128_d3/grid100`,
and the affine-normal probe at `artifacts/toy100-accuracy/affine-normal/grid100`.
All three have seed 1234, 20,000 particles, batch 2048, 4,096-sample saved
frames, and final output-noise standard deviation 0.029. The two MLPs also
have the same depth, width, Fourier features, and latent dimension. Their
optimizer, penalty, input-noise, and output-warmup settings differ. The affine
probe changes the generator, latent dimension, and Fourier features, so these
comparisons diagnose behavior rather than isolate a causal hyperparameter.

At each checkpoint the runner seeds the evaluation latent-index stream the
same way and forks the global stream used by `OutputNoise`. At steps 6000–7000
all three have reached output-noise standard deviation 0.029, so paired
sample differences cancel the identical additive noise draw. I verified that
the saved target arrays match across both checkpoints and all three runs, and
that each final 20,000-draw archive begins with the corresponding 4,096-sample
frame. Paired displacement is therefore generator **plus particle-prior**
movement; these archives cannot attribute it to either part or to the
discriminator separately.

| Run | Updates | Live RMS displacement | Live p95 displacement | Nearest-mode switches / 4,096 | Conditional centroid drift RMS / target σ | Eligible modes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lower-rate MLP | 6000→6250 | 0.02477 | 0.04518 | 0 | 0.187 | 100 |
| Lower-rate MLP | 6250→6500 | 0.01951 | 0.03716 | 0 | 0.128 | 100 |
| Lower-rate MLP | 6500→6750 | 0.01531 | 0.02909 | 0 | 0.108 | 100 |
| Lower-rate MLP | 6750→7000 | 0.01283 | 0.02458 | 0 | 0.109 | 100 |
| Shared-v3 warmup MLP | 6000→6250 | 0.04708 | 0.08467 | 1 | 1.040 | 73 |
| Shared-v3 warmup MLP | 6250→6500 | 0.03195 | 0.05427 | 0 | 0.634 | 80 |
| Shared-v3 warmup MLP | 6500→6750 | 0.02726 | 0.04706 | 0 | 0.624 | 81 |
| Shared-v3 warmup MLP | 6750→7000 | 0.02140 | 0.03760 | 0 | 0.453 | 81 |
| Affine-normal, shared-v3 core | 6000→6250 | 0.01779 | 0.02999 | 2 | 0.226 | 89 |
| Affine-normal, shared-v3 core | 6250→6500 | 0.01290 | 0.02252 | 0 | 0.149 | 89 |
| Affine-normal, shared-v3 core | 6500→6750 | 0.01000 | 0.01730 | 0 | 0.143 | 89 |
| Affine-normal, shared-v3 core | 6750→7000 | 0.00874 | 0.01543 | 0 | 0.153 | 89 |

Displacement is the Euclidean distance between paired output points. The
centroid column takes points whose *same* nearest mode is within the original
three-σ quality radius at both checkpoints, computes their mean displacement
in each mode, then reports RMS across eligible modes divided by target σ=0.03.
Eligibility requires at least 21 paired points, the original 0.5% minimum
mode-mass fraction scaled to 4,096 draws. This is only a diagnostic sample
filter, not an added success threshold. Fewer eligible high-rate modes make
its centroid statistic conditional on its better-populated modes.

| Run | Live final modes / 100 | Live final HQ | Live final mass TV | EMA final modes / 100 | EMA final mass TV | Live 6000→7000 RMS movement | Nearest-mode switches / 4,096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lower-rate MLP | 100 | 0.98955 | 0.05250 | 100 | 0.05250 | 0.02170 | 0 |
| Shared-v3 warmup MLP | 85 | 0.98265 | 0.20100 | 86 | 0.20100 | 0.05074 | 1 |
| Affine-normal, shared-v3 core | 96 | 0.98170 | 0.11825 | 96 | 0.11825 | 0.02392 | 2 |

The shared-v3 MLP still moves 1.67 times as far per paired point as the
lower-rate MLP over the last 250 updates, and its conditional centroid drift
is about four times as large among eligible modes. Yet practically no sampled
points cross nearest-mode boundaries late in training. Its remaining failure
is mainly an established unequal *allocation* of particle mass, not rapid
late mode switching. EMA smooths its final center error from 0.273σ to
0.122σ, but both live and EMA retain mass TV 0.201 and fewer than 100 modes.
The affine generator has the smallest late motion and still misses four modes
with mass TV 0.118. Lowering output motion alone would therefore not prove
that the common recipe can repair mode mass. These observations motivate a
separate, target-agnostic check of when particle mode assignments and masses
become fixed, alongside any optimizer or architecture stabilization probe.
