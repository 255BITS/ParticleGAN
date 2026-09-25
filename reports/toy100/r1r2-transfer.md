# RpGAN with zero-centered R1+R2: bounded transfer screen

**No row qualified for a ten-host or full-19 promotion.** Eight predeclared
global configurations were screened in the measured fail-fast host order.
`PASS` below means the original frozen host gate passed at its original
budget; the other hosts in that row were deliberately untested.

| Base recipe | R1+R2 coefficient γ | Strict passes / attempted | First failing host | Final diagnostic |
| :--- | ---: | ---: | :--- | :--- |
| Shared candidate | 0.01 | 3/4 | `mode_hold` | 7/8 modes, HQ 0.993 |
| Shared candidate | 0.1 | 0/1 | `trajectory` | identity MSE 0.2006 |
| Shared candidate | 1 | 0/1 | `trajectory` | identity MSE 0.5061 |
| Shared candidate | 6 | 0/1 | `trajectory` | identity MSE 0.5300 |
| Noiseless F5 | 0.01 | 3/4 | `mode_hold` | 4/8 modes, HQ 1.000 |
| Noiseless F5 | 0.1 | 3/4 | `mode_hold` | 7/8 modes, HQ 1.000 |
| Noiseless F5 | 1 | 0/1 | `trajectory` | identity MSE 0.5011 |
| Noiseless F5 | 6 | 0/1 | `trajectory` | identity MSE 0.5250 |

The [R3GAN paper](https://arxiv.org/html/2501.05441v1) motivates paired
relativistic logistic GAN loss together with zero-centered real and fake
gradient penalties, each weighted by γ/2. The local `GANLoss(logistic, rp)`
has the same paired game under the opposite critic-sign convention. The
`a_r1r2` arm computes exactly γ/2 times the mean squared critic-input
gradient on real samples **plus** the corresponding fake-sample term; the
repository's numeric equivalence test passed. We changed only that arm,
γ ∈ {0.01, 0.1, 1, 6}, and each row's display name. `reg_kappa` stayed at
its base value and is ignored by this zero-centered arm. These experiments
test the loss and penalty in the existing particle-GAN hosts. They do not
replicate the paper's modern generator/discriminator architecture or support
its convergence conclusion for these tasks.

The first four frozen hosts were `trajectory`, `residual_student`,
`img_stripes2`, and `mode_hold`; the remaining order was `img_bars4`,
`vector_overlap`, `img_blobs4`, `img_intensity2`, `vector_unequal_mass`, and
`vector_unequal_width`. This order was selected before training from the
[measured stage-cost audit](fail-fast-stage-order.md). Each attempted host
ran its complete original seed-0 budget, resources, 24 observations, and
thresholds. The stage after the first failure was marked `SKIPPED`, never
counted as a success. Since no row passed the first four, no fresh all-19
replay or native 100-mode training was run. This screen is **not** evidence
for a common 22/22 recipe.

The frozen manifest (archived `r1r2-manifest.json`) has SHA-256
`a1f565a5c9f4142825d74ae9085086cf6144dcf96122d516c8a9a24d6131ecec`.
The eight candidate configs (archived `r1r2-configs`) bind exact base hashes and
only the declared changes; source bytes for all benchmark/training modules
are from commit `1c1a0865fe605c9f212d832c06596c12510f937e`.
The fail-fast runner (archived `r1r2_screen.py`) was predeclared at scratch commit
`66bb434`; orchestration-only commit `8b590f5` fixed its import path and
resumed the already completed first host without retraining or changing a
candidate/source file. Raw local artifacts are retained at
`artifacts/toy100-accuracy/r1r2-screen-v1-66bb434`. All 138 original RAM
files matched their disk copies by SHA-256; the 148-file copy into the main
artifact directory also matched by SHA-256. The strict combined-gate episode
checker independently regraded every relocated source, config, action,
optimizer, noise, and frozen-host verdict.
