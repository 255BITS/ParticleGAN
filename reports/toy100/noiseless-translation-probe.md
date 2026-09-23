# Noiseless translation-only generator probe

Freezing the affine generator's identity matrix, while training its two translation parameters and the particle prior, improves the noiseless grid100 result but does not pass either frozen gate. The exact public-default optimizer/loss core and F5/H1600 config are unchanged. Model construction preserves the ordinary affine initialization's random draws; seed 1234, 20,000 particles, 7,000 updates, five terminal checks, and the separate 100,000-sample holdout remain fixed.

| Measurement | Final 20k | Independent 100k |
| --- | ---: | ---: |
| Precision | .97065 | .96974 |
| Mass TV | .04595 | .03817 |
| Center RMS / target σ | .30080 | .28057 |
| Covariance trace bias | −.05194 | −.06075 |
| Radial KS | .01464 | .01387 |

The final sample covers all 100 modes, but center error exceeds the .20 limit. The original conditional shape checks also fail: final component covariance eigenvalue ratios span .1941–1.9473 (required .4–1.7), and radial median ratios span .3712–1.6091 (required .65–1.4). All five terminal checks fail. Passing pooled mass and average shape statistics cannot replace those conditional checks.

This is an explicitly ineligible scratch model, not a common-22 result. Its source was frozen at `5bb6491`; config SHA-256 is `51f6bd91a60e523708b51a36bd2f38907a28bc138a1d0d1b366e6a6a200586b0`. The local archive `artifacts/toy100-accuracy/noiseless-translation-v1/` retains its custom trainer receipt, executable source, samples, holdout, and failed gates. All 54 files matched the RAM originals by SHA-256, the relocated source archive verified, and independent relocated regrading reproduced FAIL. The [ordinary affine controls](affine-noiseless-native.md) and [fresh older-19 replay](affine-noiseless-f5-transfer.md) provide separate comparisons.
