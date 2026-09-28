# Prior-rate coupling native100 near miss

This completed 7,000-update native grid100 test is the strongest directional result after the paired birth/death graft. It **fails** the frozen gate: noisy precision is slightly below the required 0.97 in both the final grid and independent 100k holdout. No rotated/staggered or 22-check score is claimed.

The sole source change relative to the committed paired-BD graft caps the applied prior stationarity scale at the lowest non-sigma generator scale. The prior table therefore cannot update faster in intrinsic settle time than the map that moves its samples. The recipe and BD code are byte-identical to the graft, with QR `batch_feature_zero` initialization, learnable output noise initialized at 0.02, no horizon (`total_steps: null`), and noisy sampling-law scoring.

| Frozen live measure | Result | Requirement |
|---|---:|---:|
| Modes | 100 | 100 |
| Noisy precision | .96875 | ≥ .97000 |
| Centre RMS / data σ | .12646 | ≤ .20 |
| Covariance eigenvalue ratios | .57226–1.32953 | .40–1.70 |
| Mass TV | .03370 | ≤ .06 |
| Radial KS | .03298 | ≤ .04 |
| Independent 100k precision | .96946 | ≥ .97000 |
| Frozen verdict | **FAIL 0/34** | final five + holdout |

Precision rose from .96415 at step 4,750 to .96875 at step 7,000. At the end, applied G/prior scales were both 1/64, with G/prior rates 6.640625e-5 / 1.328125e-4. The raw prior tester still reported scale 1. `_sigma_intrinsic_scale()` and `_output_sigma()` inspect that raw tester value, so sigma LR remained zero and output sigma stayed at .02. This is a semantic inconsistency in this candidate, and the next isolated experiment will make those controls inspect the effective applied scales. Whether that change improves the frozen result is untested here.

`training.patch` applies to the paired-BD graft training source with SHA256 119888571fea4784f2ec051831f93dbd3721d3e910b6c49475bfaa8087ba9b54. `source-sha256.json` records the resulting source, unchanged BD implementation and overrides hashes. The raw frozen result, per-check trace, fixture and noisy verdict are included for review.
