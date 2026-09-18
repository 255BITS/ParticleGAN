# Particle information diagnostics

7/7 certified; frozen checkpoints unchanged.

Fixed Inception features, equal real/generated budgets, same cached real reference. Existing checkpoint FID50k is shown separately; this probe does not recompute FID.

| Checkpoint | FID50k | Density | Coverage | Feature variance / real | Probe minutes |
|---|---:|---:|---:|---:|---:|
| p1024_20k | 19.8932 | 0.6281 | 0.5813 | 1.0678 | 0.55 |
| p4096_20k | 18.5207 | 0.6344 | 0.6028 | 1.0583 | 0.93 |
| p8192_20k | 18.1602 | 0.6352 | 0.6029 | 1.0710 | 1.17 |
| p16384_20k | 18.0136 | 0.6294 | 0.5943 | 1.0742 | 2.10 |
| p4096_35k | 16.5033 | 0.6303 | 0.6258 | 1.0644 | 0.72 |
| p1024_40k | 26.0216 | 0.4662 | 0.4514 | 1.0960 | 0.39 |
| p4096_40k | 17.2350 | 0.6148 | 0.6139 | 1.0689 | 0.72 |

| Checkpoint | Decodable bits | Available bits | Shuffled-label bits | Between-sibling variance | Within-child variance |
|---|---:|---:|---:|---:|---:|---:|
| p1024_20k | 0.0000 | 0 | 0.0000 | 0.00% | 33.15% |
| p4096_20k | 1.1166 | 2 | -0.0004 | 6.25% | 31.45% |
| p8192_20k | 1.7195 | 3 | -0.0017 | 8.45% | 32.25% |
| p16384_20k | 2.2756 | 4 | -0.0011 | 10.14% | 32.62% |
| p4096_35k | 1.4938 | 2 | -0.0017 | 10.89% | 31.50% |
| p1024_40k | 0.0000 | 0 | 0.0000 | 0.00% | 33.32% |
| p4096_40k | 1.4854 | 2 | -0.0007 | 12.29% | 31.41% |

Bits are a held-out, restricted-decoder estimate of a conditional mutual-information lower bound, not exact entropy or semantic coverage. Negative estimates are retained. The synthetic identical-clone control gives zero additional bits. Shuffled labels measure the null behavior with independent train/validation/test assignments.

Density need not be bounded by1 and higher density alone does not demonstrate quality. Coverage depends on extractor, neighborhood k and sample counts. ANOVA fractions are finite-sample descriptive measurements; estimated child means include sampling noise. More distinguishable features can reflect artifacts.

The compression hypothesis predicts additional decodable information accompanied by broader real-feature coverage without deterioration in fidelity. These metrics diagnose that pattern; FID50k remains the benchmark target.
