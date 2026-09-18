# Particle information diagnostics

SMOKE ONLY: reduced sample counts; no benchmark interpretation.
1/1 certified; frozen checkpoints unchanged.

Fixed Inception features, equal real/generated budgets, same cached real reference. Existing checkpoint FID50k is shown separately; this probe does not recompute FID.

| Checkpoint | FID50k | Density | Coverage | Feature variance / real | Probe minutes |
|---|---:|---:|---:|---:|---:|
| p4096_20k | 18.5207 | 0.7682 | 0.6875 | 1.0672 | 0.06 |

| Checkpoint | Decodable bits | Available bits | Shuffled-label bits | Between-sibling variance | Within-child variance |
|---|---:|---:|---:|---:|---:|---:|
| p4096_20k | 0.2962 | 2 | -0.0554 | 13.46% | 26.15% |

Bits are a held-out, restricted-decoder estimate of a conditional mutual-information lower bound, not exact entropy or semantic coverage. Negative estimates are retained. The synthetic identical-clone control gives zero additional bits. Shuffled labels measure the null behavior with independent train/validation/test assignments.

Density need not be bounded by1 and higher density alone does not demonstrate quality. Coverage depends on extractor, neighborhood k and sample counts. ANOVA fractions are finite-sample descriptive measurements; estimated child means include sampling noise. More distinguishable features can reflect artifacts.

The compression hypothesis predicts additional decodable information accompanied by broader real-feature coverage without deterioration in fidelity. These metrics diagnose that pattern; FID50k remains the benchmark target.
