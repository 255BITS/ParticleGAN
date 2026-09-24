# Noiseless F5 prior-regularization transfer screen

Four predeclared global `prior_reg` values were run on the same ten frozen
older hosts. **None passed all ten.** The strongest row, 0.04, passed 8/10;
it therefore did not qualify for a fresh all-19 replay or any native
promotion.

| `prior_reg` | Strict passes / 10 | Failed frozen hosts |
| ---: | ---: | :--- |
| 0.000 | 5/10 | `trajectory`, `mode_hold`, `vector_unequal_mass`, `vector_unequal_width`, `vector_overlap` |
| 0.010 | 6/10 | `mode_hold`, `vector_unequal_width`, `img_stripes2`, `img_bars4` |
| 0.025 | 5/10 | `vector_unequal_mass`, `vector_unequal_width`, `vector_overlap`, `img_stripes2`, `img_blobs4` |
| 0.040 | 8/10 | `vector_unequal_width`, `img_blobs4` |

At 0.04, the width host's final component minimum eigenvalue ratio was
0.128 versus its frozen 0.15 minimum. The blobs host passed its final
observation but only four consecutive terminal checks, one short of the
required five. `mode_hold` and unequal mass passed at this setting. These
are host verdicts from original budgets, seeds, data, resources, and
thresholds, not a relaxed transfer score.

`prior_reg` weights the learned particle prior's VICReg-like variance-floor
and off-diagonal covariance penalties. This screen reduced that global
weight; it did not change an L2 contraction term. The exact
base config (archived `priorreg-f5-configs/base.json`) matches SHA-256
`51f6bd91a60e523708b51a36bd2f38907a28bc138a1d0d1b366e6a6a200586b0`.
Each row config (archived `priorreg-f5-configs`) changes only `prior_reg` and `name`.
The predeclared manifest (archived `priorreg-f5-manifest.json`), SHA-256
`cbbe05c30f2c800db680fbba4067070575acadd7c075c5ee7013e436bbeafcfb`,
binds all four config hashes and the ten named hosts. The run used native and
transfer source bytes from commit `1c1a0865fe605c9f212d832c06596c12510f937e`;
only report/config files were added in scratch commit
`683fa48bce89456e95c136c6077199031ebf5e6f`.

Raw evidence is retained locally at
`artifacts/toy100-accuracy/priorreg-f5-transfer-v1-683fa48`. All 68
files matched their RAM originals by SHA-256. The strict combined-gate
episode checker regraded every relocated source/config/optimizer receipt and
all 40 host verdicts as 5/10, 6/10, 5/10, and 8/10. This is a **ten-host
subset**, not all-19 or common-22 evidence.
