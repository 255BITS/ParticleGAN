# PR #215 GS2 and PR #217 DV12 on the #155 frozen hosts

This comparison uses the same 13-task and native 100-Gaussian hosts as the
#155 research harness. It is separate from the [structural `st5`/`st6`
round](../structural100/README.md). Results below score the model's own
samples, including output noise when the recipe has it. The native verdict
requires sustained live accuracy and an independent 100k holdout.

## Source and protocol

| Arm | Source and adaptation | Initialization |
|---|---|---|
| #215 exact GS2 | [PR #215](https://github.com/255BITS/ParticleGAN/pull/215) commit `c138aff3a1fcdca76ee3e84567c28b6f9be7db12`; package SHA-256 `eb988dd9f211df8af9c3ade4ba90edc438eec6f71251545b467e6dc91ade184b` | Frozen host defaults; native critic Xavier |
| #215 QR | Same commit plus [QR adapter](pr215-qr-adapter.patch), patch SHA-256 `332b2e6728d4a2477b23a1b009369019b7871b8733385db3762026e76ecd791b`; package SHA-256 `0a5aae9479ab741dd0f9f0ebde86137fbde43109e33e128d4839c7ad6a1d9686` | #155 QR |
| #217 QR | [PR #217](https://github.com/255BITS/ParticleGAN/pull/217) commit `261fcfd1decba067b9c95123abf21e046dc32127`; exact package SHA-256 `90e96a276b90af0316de5960dfd967c3ee6f639fca161b0e78f7ef510627a849`; [QR/native adapter](pr217-qr-native-adapter.patch), patch SHA-256 `a9d3a922abf7ba7ceed576cee5b76c669512b4efac42d125699df25679114d14`, package SHA-256 `387a1265838966581b7e1858593ce9f292f376ba76d04f5f2c64e0666bf814c4` | #155 QR |

The #215 QR adapter applies #215's public orthogonal initializer to fresh
G/D and recipe-created priors before optimizers; it preserves the native
host's supplied prior. The GS2 loss, optimizers, rates, penalty, and sampling
remain at the source commit. The exact arm uses the source initializer. The
#217 source rejects the frozen native host's supplied prior because it lacks
`support_jitter=True`; its adapter adds the two zero buffers required by the
source implementation while preserving the host's latent fixture, parameters,
and random streams. It also initializes fresh G/D and recipe-created priors
with #217's public QR initializer. The DV12 controller, penalty, losses,
optimizers, noise, and update order remain at the source commit. These are
protocol adapters, so the adapted rows are labeled separately from exact
source behavior.

#217 moved image latent jitter into `prior.perturb()`. Its corrected image
task runs use the harness evaluation opt-in `image_prior_perturb=true`
so evaluation follows that source sampling path. The earlier #217 image
results that evaluated raw `prior.z` are excluded. The opt-in changes the
evaluator's draw, not the candidate package. Its run headers record the
option and the same adapted package digest.

The native #215 comparison uses 7,000 updates, `eval_output_noise=true`, and
the same frozen 100-Gaussian cards. The 13-task gate judges `img_intensity2`
at 1,200 updates and `stationary` at 7,500. For #215, the latter requires a
9,000-step recipe budget override solely to let the frozen host finish; the
constant rates do not change. Results, run paths, and source digests are in
[results.json](results.json).

## Frozen 13-task result

| Task | #215 exact / host init | #215 QR adapter | #217 QR/native adapter |
|---|---|---|---|
| mode_hold | FAIL | FAIL | PASS |
| img_intensity2 @1,200 | FAIL | PASS | PASS |
| img_blobs4 | PASS | FAIL | PASS |
| img_stripes2 | FAIL | PASS | PASS |
| img_bars4 | FAIL | PASS | PASS |
| vector_two_broad | PASS | PASS | PASS |
| vector_unequal_mass | FAIL | FAIL | PASS |
| vector_unequal_width | FAIL | FAIL | PASS |
| vector_anisotropic | FAIL | FAIL | PASS |
| vector_overlap | FAIL | FAIL | PASS |
| vector_spiral | PASS | PASS | PASS |
| ring_shift | PASS | PASS | PASS |
| stationary | PASS | PASS | PASS |
| **Total** | **5/13** | **7/13** | **13/13** |

The historical #155 `dv12-ams-rc3` reference passed 13/13 on this suite,
including the first 1,200 updates of its longer `img_intensity2` run. That
reference underlies the #217 default and is recorded in the older
[leaderboard](../README.md#earlier-leaderboard-september-27-evening-mixed-native-initialization).
The exact #217 source comparison above confirms 13/13 with the corrected
image evaluator. The historical result is context, not the exact-head score.
Its saved
package digest changed from
`f9851e15ca315d2c03641194262867812b822615ab8d94e5822ccda7afa357d0`
to `ca2feb43713a0eb8000fbdcf6a640d83d47ce275bbb3228a34ed271756f72506`
through the nearest-neighbor speed change documented at
`/ml2/hypergan/lrfree-20260926/reports/perturb-latent-speedup.md`.
Its checked runs had unchanged rates, metrics, sample clouds, and gates;
wall time and the package digest changed. This is a code-path parity check,
not a new 13-task score.

## Frozen native 100-Gaussian result

| Arm | grid100 | rotated100 | staggered100 | Native passes |
|---|---|---|---|---:|
| #215 exact / native Xavier D | FAIL, 100 modes / .982 precision | FAIL, 100 / .969 | FAIL, 100 / .977 | 0/3 |
| #215 QR adapter | FAIL, 100 / .986 | FAIL, 100 / .967 | FAIL, 98 / .978 | 0/3 |
| #217 QR/native adapter | FAIL, 99 / .889 | FAIL, 100 / .878 | FAIL, 100 / .884 | 0/3 |

All nine native gates have zero passing checks out of 34 and failing 100k
holdouts. Reaching 100 modes or high aggregate precision does not establish
per-mode shape accuracy. On grid, #217 reaches 99 modes with centre RMS
`.374σ`; on rotated it reaches 100 modes but precision is `.878` and centre
RMS `.220σ`. The QR arms share the #155 initializer; exact #215's native
arm with Xavier critic is a source control with a different trajectory. The
historical #155 DV12
reference also failed all three native gates under QR/noisy scoring, but its
supplied-prior path differs from exact #217. The exact-head adapted run above
resolves that provenance gap. The published #217 CLI Xavier figure uses
another initialization and should not be substituted for this QR table.

## High-dimensional nearest-search follow-up

Exact #217 commit `261fcfd` computes support-jitter's nearest-particle
distance in memory-bounded broadcast blocks. At 64 query rows, 65,536
particles, and latent dimension 128, its old-distance-matrix bound selects a
4 MiB temporary block and 517 center launches. The isolated
[benchmark script](pr217-nearest-block-bench.py) varied only the block floor
in one Python process on a shared RTX A6000 card. It imported the unchanged
exact #217 package; [machine results](pr217-nearest-block-bench.json) record
four rotated measurement cycles, source shape, checksums, and peak allocated
memory.

| Temporary block floor | Center blocks | Median search time | Peak extra allocation |
|---:|---:|---:|---:|
| 4 MiB, current | 517 | 27.69 ms | 4.03 MiB |
| 16 MiB | 130 | 16.58 ms | 16.12 MiB |
| **32 MiB, proposed** | **65** | **11.69 ms** | **32.50 MiB** |
| 64 MiB | 33 | 11.02 ms | 64.99 MiB |

All four outputs were bitwise equal on this seeded `z=128` table. The
[32 MiB proposal](pr217-highz-nearest-32m-proposal.patch) (SHA-256
`7165f48242efd31b3e7c0763231918649cc14f23e0cd39b1b54e72aad567a850`)
raises the CUDA floor only when `z_dim>32`; the native `z=2` branch retains
the exact old expression and therefore its bitwise behavior. The patch
applies to the exact and QR-adapted #217 packages, but was **not** applied to
any reported gate run or live source. Its 2.37× figure is nearest-search
latency on this shape under shared GPU load, not a measured full-step speedup.
Bitwise equality for this high-dimensional sample does not prove equality
for every dtype and shape; the low-dimensional native path is unchanged by
construction.

## Scope of other published numbers

The [#215/#202 17/26](https://github.com/255BITS/ParticleGAN/pull/202)
result was measured on a different 26-task transfer suite. Its image host
uses a transpose convolution with width 12, whereas this #155 13-task suite
uses a residual upsample host with width 16; task composition also differs.
The 17/26 figure is therefore context, not another count on these hosts.

The [#218 high-latent-dimension performance
regression](https://github.com/255BITS/ParticleGAN/pull/218) is separate from
these quality gates. A #217 high-`z` benchmark reported a drop from 16.1 to
9.4 steps/s after its nearest-neighbor change on a CIFAR-like setup, with
sampling concentrated in `_nearest_other`. These runs do not provide a
matched-host speed comparison between GS2 and DV12; package changes, latent
dimension, and workload matter. No relative speed claim follows from the
gate totals above.

## Reproduce and audit

The read-only harness is `/ml2/hypergan/lrfree-20260926`. Source checkouts
are `/ml2/hypergan/ParticleGAN-pr215-gs2-audit` and
`/ml2/hypergan/ParticleGAN-pr217-dv12-audit`; packages and run directories
are named in `results.json`. Each run's `result.json` records the package
digest, overrides, options, fixture, status, and holdout. The adapter patches
in this folder reconstruct the packages from those source commits. Check a
native 7k result's live metrics in `metrics.jsonl` at `step=7000`; use its
native verdict and holdout from `result.json` for the gate.
