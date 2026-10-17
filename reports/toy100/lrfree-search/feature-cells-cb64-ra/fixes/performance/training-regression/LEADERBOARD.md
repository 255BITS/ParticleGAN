# Saved learned-state regression

## CUDA outcome recorded by the frozen learned harness

These are the original CUDA measurements, not a CPU quality retest. All three toy variants fail the unchanged P≥.9, 25-mode, TV≤.1 gate. MNIST has no invented acceptance threshold.

| Variant | Toy P | Modes | TV | Toy seconds | MNIST active FD | Active P | Active R | Class TV | MNIST seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E22 | 0.7155 | 25 | 0.2845 | 72.34 | 0.54449 | 0.8691 | 0.8472 | 0.0364 | 90.96 |
| CB64-RA | 0.6233 | 24 | 0.3780 | 80.79 | 1.48883 | 0.8525 | 0.3228 | 0.1244 | 103.26 |
| CB64-RA2 | 0.1566 | 1 | 0.8434 | 90.52 | 0.81414 | 0.8418 | 0.7139 | 0.0514 | 107.98 |

## Same-input conditional gradient comparison

Fast training G/D weights, first128 prior rows, that update’s original generator real batch, and the same saved fold128/fold2 noise. Columns compare the existing latent kernel against zero latent jitter; both include the same toy output noise. No critic or optimizer update is performed. These 128-row coverages are diagnostic observations, not the 8192-sample acceptance run.

| Variant | Step | G gradient cosine | Prior gradient cosine | Opposed prior rows /128 | Zero-jitter P / modes | Existing-kernel P / modes |
|---|---:|---:|---:|---:|---:|---:|
| E22 | 1000 | 0.9846 | 0.8732 | 5 | 0.6797 / 23 | 0.5938 / 23 |
| E22 | 2000 | 0.9283 | 0.6632 | 19 | 0.7266 / 22 | 0.5859 / 22 |
| CB64-RA | 1000 | 0.9989 | 0.9880 | 0 | 0.3984 / 13 | 0.3906 / 13 |
| CB64-RA | 2000 | 0.9998 | 0.9986 | 0 | 0.5000 / 18 | 0.5156 / 18 |
| CB64-RA2 | 1000 | 0.8055 | 0.4466 | 31 | 0.1719 / 7 | 0.0859 / 2 |
| CB64-RA2 | 2000 | 0.0113 | 0.1485 | 60 | 0.3047 / 10 | 0.1016 / 5 |

## Same-snapshot count decomposition

Existing score, null and Q=.05 remain fixed. Pointwise support here means the current p>Q parent eligibility. Its boundary uses odd calibration, so these augmented counts are descriptive and are not a valid new fixed-partition test. An eventual count boundary must be fitted on even real rows alone.

| Step | Real eligible | Emitted fake eligible | Clean table eligible | Emitted / table BH flags | Count discoveries /64 | Emitted flags without count discovery | Table flags without count discovery | Full / augmented TV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.9531 | 0.1016 | 0.1250 | 913 / 887 | 6 | 788 | 805 | 0.3096 / 0.8604 |
| 2000 | 0.9531 | 0.1572 | 0.2861 | 860 / 710 | 5 | 820 | 688 | 0.2734 / 0.8008 |
