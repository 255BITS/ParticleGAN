# Completed duration experiments

Both training runs and endpoint diagnostics finished; both GPUs are idle following the deconv scout. Current-source training certificates rechecked. Endpoint probes use the same real-feature SHA and report unchanged parent/frozen state, near-zero shuffled-label and identical-clone controls.

| Model | Step | FID50k | Density | Coverage | Sibling bits |
|---|---:|---:|---:|---:|---:|
| 16k CNN | 80000 | 15.7527 | 0.64686 | 64.26% | 3.11494/4 |
| 16k CNN | 160000 | 19.0584 | 0.59038 | 58.03% | 1.75446/4 |
| 32k CNN | 40000 | 16.3451 | 0.61684 | 62.17% | 3.48376/5 |
| 32k CNN | 80000 | 16.2461 | 0.59478 | 62.23% | 3.74238/5 |

16k FID at90/100/110/120/130/140/150/160k:15.7901,16.4609,17.3499,18.1365,18.6474,19.0182,19.0753,19.0584. Its80k parent remains the overall best15.7527. Longer training at unchanged rates did not help: independent density/coverage also declined, and the restricted decoder recovered fewer sibling bits. This is consistent with lost useful distinctions, but the probe does not identify whether prior motion, generator changes, or their interaction caused it. Bits can also describe artifacts and are not an exact information or mode count.

32k FID at50/60/70/80k:16.1271,16.4369,16.6663,16.2461. Best sampled32k checkpoint is50k16.1271. At80k it is0.4934 worse than16k at80k. Its earlier particle-count advantage at40k did not persist at80k. From40k to80k it gains a little decodability while coverage remains nearly constant and density declines; extra distinguishability need not mean better samples.

Recommendation: retain16k80k as the reference and do not blindly extend the same fixed-rate CNN recipe to200k. A checkpoint intervention on center motion or learning-rate schedule would be more informative than more identical training. Separately, the plain deconv scout remains improving at40k (FID26.2232); an extension there asks a different question. No new training or intervention queued.
