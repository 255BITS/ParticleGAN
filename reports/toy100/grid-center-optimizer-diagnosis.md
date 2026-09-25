# Late grid-center dynamics: κ=1.0 CI control and κ=1.176

The new κ=1.176 grid100 run passes **both the original gate and the accuracy gate in all five terminal 20,000-sample checks**, live and EMA. Its separate 100,000-sample holdout also passes (live precision .98266, center RMS .1034 target-σ). The terminal checks are:

| Step | Live precision | EMA precision | Live center RMS (σ units) | EMA center RMS (σ units) | Both gates, live / EMA |
| ---: | ---: | ---: | ---: | ---: | --- |
| 6000 | .97580 | .98250 | .144 | .129 | PASS / PASS |
| 6250 | .97825 | .98240 | .124 | .126 | PASS / PASS |
| 6500 | .98090 | .98215 | .131 | .123 | PASS / PASS |
| 6750 | .98225 | .98275 | .125 | .122 | PASS / PASS |
| 7000 | .98320 | .98330 | .130 | .124 | PASS / PASS |

I assigned each sample to its nearest analytic grid center, kept points within the fixed 3σ quality radius, estimated each component's center from its accepted samples, and fitted a common translation plus 2×2 linear displacement to the 100 center errors. Each mode has roughly 200 accepted samples (minimum 139 at the κ=1.176 final check). A true target draw at this sample size has about **.10σ center RMS from finite sampling alone**; the saved 16-draw oracle calibration measured .099σ on average. Plug-in covariance/count estimates give a .097σ sampling floor for the final generated clouds.

| Final 20k diagnostic | κ=1.0 CI live | κ=1.176 live |
| --- | ---: | ---: |
| Observed center RMS | .129σ | .130σ |
| Approximate center RMS after subtracting sampling variance | .085σ | .087σ |
| Common affine part, observed / variance-corrected | .050σ / .047σ | .034σ / .030σ |
| Affine share of variance-corrected center error | ~30% | ~12% |
| Live–EMA paired center gap, observed / paired sampling floor | .0466σ / .0095σ | .0358σ / .0093σ |
| Affine part of paired live–EMA gap, observed / paired floor | .0387σ / .0017σ | .0265σ / .0016σ |

Thus **absolute center error is mostly mode-specific**, especially at κ=1.176; a coherent displacement of the entire grid is small. The *difference between live and EMA* is more coherent: about 55% of its final squared per-mode center gap at κ=1.176 is affine, and that gap is larger than the paired sampling floor. Live and EMA use the same indexed latent/noise draws at each checkpoint (100% nearest-mode agreement), so the paired floor is the appropriate uncertainty scale for that comparison. Variance correction is a plug-in estimate, not a confidence interval.

The five checkpoints do **not** establish adversarial cycling. κ=1.176 live precision rises from .9758 to .9832 while its EMA gap narrows from .0067 to about .0001. Live conditional covariance trace bias moves in one direction, +.073 to −.036; EMA moves smoothly from −.030 to −.040. Adjacent live center changes are .054–.093σ RMS, but their common affine parts are only .028–.040σ; EMA changes are .026–.037σ with affine parts below .006σ. There is some nonmonotonic, small live affine motion, but no repeated cycle is resolved by these five 250-step observations.

A bounded **Optimistic Adam hypothesis** is that anticipating the next G/D gradient might reduce the transient live–EMA lag or the residual mode-specific chasing. These samples provide a measurable lag to target, not evidence that it is caused by a rotational game dynamic; changing the optimizer could also alter the already passing mass and width gates. A controlled same-seed optimizer comparison would need full frozen budgets and a fresh combined 22 verdict before claiming improvement.

Exact inputs: the κ=1.0 CI grid run (`artifacts/toy100-accuracy/ci-common22-network-floor010/artifacts/toy-suite-ci/toy100/grid100`) (final NPZ SHA-256 `529717fc…`) and the [retained κ=1.176 run](shared22/toy100/grid100) (final NPZ SHA-256 `93d1331e…`). All five quality-check NPZ hashes and both final NPZ hashes were verified against their summaries. The [machine-readable diagnosis](grid-center-optimizer-diagnosis.json) contains full source commits, file hashes, per-check metrics, fitted affine coefficients, and paired uncertainty estimates. The resolved configs differ only in `name` and κ; their archived native source versions differ in `config.py`, `models.py`, and `train.py`, so small between-run differences should not be assigned causally to κ alone. No training, seed, gate, or source was changed for this diagnosis.
