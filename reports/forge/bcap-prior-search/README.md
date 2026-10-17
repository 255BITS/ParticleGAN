# BCAP prior repair searches

The initial mean/covariance tether at global strength **0.01** repairs **mode_hold**, increasing the terminal passing suffix from **3 to 7 / ≥5**. It loses **word retention**, passing **23/25** mandatory checks versus the leader's **25/25**. No tested recipe repairs a Tier 2 failure while preserving every leader Tier 2 pass.

Completed all 15 registered global recipes. 4 passed the six Tier 1 gates and reached Tier 2; 11 were blocked before Tier 2. Matched Tier 2 FAIL→PASS task/recipe pairs: 1. Recipes meeting the full registered repair criterion: none.

[Interpretation and recommendations](findings.md) explain the sustained mode-hold repair, its word-retention tradeoff, and the remaining native density errors.

Five independent feature PRs compare three positive global strengths each against a fresh, unchanged BCAP leader control. Protocol seed 0, public deterministic initialization, revision-9 tasks, original architectures/data/prior capacity/sampling/update budgets/evaluation cadence and gates are held fixed. The recipe delta is prior regularization or the explicit proximal prior update.

The complete `recipe-prior-refactor-bcap-pure-v1` leader card is the baseline in all five: non-saturating BCAP, direction blend, zero-momentum DualNorm with smoothing 0.001, base learning rate 0.012, discriminator multiplier 1.5 and prior multiplier 2.5, with no EMA or additional output noise. Native MixtureOfGaussians retains intrinsic sigma 0.025. The full host-adaptation and recipe overrides are copied; using a preset name alone would not establish this baseline.

Each arm must pass all six required Tier 1 tasks before any Tier 2 work. Every eligible arm completes its runnable Tier 2 peers. Unreached tasks retain their targets and unavailable scores. No seed studies, scientific retries, per-task recipes, Tier 3 runs or default adoption are included.

Task verdicts use their declared temporal rules. Acquisition smoke requires an earlier independently confirmed passing state; its final endpoint can differ. Retention and density tasks require their complete hold or terminal windows. Detailed reports show these gate counts beside endpoint scores, with compact worst-check witnesses for failed mandatory word holds. Optional clock probes use their own smaller recipe and do not establish invariance of the new prior feature.

The original word-hold task resumes each recipe's own earliest independently confirmed passing prefix, with its model, optimizer and named streams restored. Prefix lengths can differ; those holds compare the declared own-prefix protocol. The ring task retains its original adaptive stopping rule and maximum budget. Audits record actual consumed update counts and distinguish these cases from equal-length stream comparisons.

The registered repair criterion requires all six Tier 1 passes, at least one matched Tier 2 FAIL→PASS, and preservation of every control Tier 2 pass. One whole recipe per mechanism is displayed below: meeting that criterion first, then most Tier 1 passes, most Tier 2 passes, fewer regressions/unresolved control passes, and lower strength. All three searched strengths and every task measurement, score and target are in each linked detailed report. These are source-bound comparisons; the sole generated [current leaderboard](../technique-inventory.md) retains its published cohorts.

Counts below are passes / measured PASS-or-FAIL results / required tasks. Blocked tasks are unavailable measurements, and do not count as measured failures.

| Mechanism / feature PR | Strength | Tier 1 | Tier 2 | Matched FAIL→PASS | Lost control passes | Unresolved control passes |
| --- | --- | --- | --- | --- | --- | --- |
| [proximal](https://github.com/255BITS/ParticleGAN/pull/387) / [all measurements](proximal/README.md) | `prior_proximal_strength=1` | 5/6/6 | 0/0/21 | none | none | 9 |
| [coulomb](https://github.com/255BITS/ParticleGAN/pull/388) / [all measurements](coulomb/README.md) | `prior_reg=0.01` | 6/6/6 | 8/21/21 | none | five_word_joint_hold | 0 |
| [sigreg](https://github.com/255BITS/ParticleGAN/pull/389) / [all measurements](sigreg/README.md) | `prior_reg=0.001` | 5/6/6 | 0/0/21 | none | none | 9 |
| [covariance](https://github.com/255BITS/ParticleGAN/pull/390) / [all measurements](covariance/README.md) | `prior_reg=0.01` | 6/6/6 | 9/21/21 | mode_hold | five_word_joint_hold | 0 |
| [mmd](https://github.com/255BITS/ParticleGAN/pull/391) / [all measurements](mmd/README.md) | `prior_reg=0.1` | 6/6/6 | 8/21/21 | none | five_word_joint_hold | 0 |

## What the five changes test

- **proximal**: A position spring restores sampled prior rows toward their initial positions after the normalized BCAP step. This tests magnitude-sensitive prior motion while allowing the generator and critic to keep learning.
- **coulomb**: Softened inverse-distance repulsion follows a deterministic evenly spaced cyclic-offset rule on the supplied rows, with at most eight offsets and confinement to prevent expansion. Numeric offsets depend on the supplied row count; rows select neighbors by index within that input. The bounded graph tests spacing at linear cost.
- **sigreg**: Projected Gaussian characteristic-function matching penalizes distributional departures beyond covariance. This tests whether a Gaussian latent law gives the generator a more stable input geometry.
- **covariance**: A fixed-initial mean and full-covariance tether resists translation, contraction and anisotropy without imposing a new unit-scale prior. This tests preservation of the task's original global geometry.
- **mmd**: A multiscale IMQ distribution tether compares a finite deterministic panel with its initial law. It permits particle exchange while detecting changes beyond second moments; only selected current panel rows receive this regularizer's gradient.

The leader uses smoothed prior-row normalization: `g / sqrt(||g||² + 0.001²)`, applied only to the unique sampled rows. Unsampled rows stay fixed even when a regularizer creates dense gradients. A coefficient changes the combined gradient's direction and magnitude; the resulting displacement is nonlinear and saturates for large gradients. The proximal update acts after that normalization; the other four regularizers enter the prior loss. Each PR declares this trainer delta explicitly. These objectives measure latent locations; generated component density is graded independently by the original task gates.

The baseline GANTrainer supplies all locations to its regularizer for tables of at most 1,024 rows, and unique generator-sampled locations for larger tables. The native tasks have 20,000 locations. Fixed initial reference statistics/panels remain tied to the full initialization; current inputs follow each host's retained policy. Whole-table saved-state geometry and moment diagnostics are separate from historical batch objectives or forces.

## All registered strengths

| Mechanism | Global settings | Tier 1 | Tier 2 | Repairs | Regressions | Unresolved control passes |
| --- | --- | --- | --- | --- | --- | --- |
| proximal | `prior_proximal_strength=0.1` | 4/6/6 | 0/0/21 | none | none | 9 |
| proximal | `prior_proximal_strength=1` | 5/6/6 | 0/0/21 | none | none | 9 |
| proximal | `prior_proximal_strength=10` | 4/6/6 | 0/0/21 | none | none | 9 |
| coulomb | `prior_reg=0.001` | 5/6/6 | 0/0/21 | none | none | 9 |
| coulomb | `prior_reg=0.01` | 6/6/6 | 8/21/21 | none | five_word_joint_hold | 0 |
| coulomb | `prior_reg=0.1` | 6/6/6 | 8/21/21 | none | vector_two_broad | 0 |
| sigreg | `prior_reg=0.0001` | 4/6/6 | 0/0/21 | none | none | 9 |
| sigreg | `prior_reg=0.001` | 5/6/6 | 0/0/21 | none | none | 9 |
| sigreg | `prior_reg=0.01` | 4/6/6 | 0/0/21 | none | none | 9 |
| covariance | `prior_reg=0.001` | 5/6/6 | 0/0/21 | none | none | 9 |
| covariance | `prior_reg=0.01` | 6/6/6 | 9/21/21 | mode_hold | five_word_joint_hold | 0 |
| covariance | `prior_reg=0.1` | 5/6/6 | 0/0/21 | none | none | 9 |
| mmd | `prior_reg=0.1` | 6/6/6 | 8/21/21 | none | five_word_joint_hold | 0 |
| mmd | `prior_reg=1` | 5/6/6 | 0/0/21 | none | none | 9 |
| mmd | `prior_reg=10` | 4/6/6 | 0/0/21 | none | none | 9 |

## Tier 2 gates at a glance

Each cell shows its own matched control → the selected whole recipe. An endpoint improvement alone earns no repair credit.

| Task | proximal | coulomb | sigreg | covariance | mmd |
| --- | --- | --- | --- | --- | --- |
| cover_leftover | PASS → BLOCKED | PASS → PASS | PASS → BLOCKED | PASS → PASS | PASS → PASS |
| five_word_joint_hold | PASS → BLOCKED | PASS → FAIL | PASS → BLOCKED | PASS → FAIL | PASS → FAIL |
| gaussian1d_stability | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → FAIL | FAIL → FAIL |
| grid100 | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → FAIL | FAIL → FAIL |
| img_bars4 | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → FAIL | FAIL → FAIL |
| img_blobs4 | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → FAIL | FAIL → FAIL |
| img_intensity2 | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → FAIL | FAIL → FAIL |
| img_stripes2 | PASS → BLOCKED | PASS → PASS | PASS → BLOCKED | PASS → PASS | PASS → PASS |
| mid_scale_identity | PASS → BLOCKED | PASS → PASS | PASS → BLOCKED | PASS → PASS | PASS → PASS |
| mode_hold | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → PASS | FAIL → FAIL |
| residual_student | PASS → BLOCKED | PASS → PASS | PASS → BLOCKED | PASS → PASS | PASS → PASS |
| rotated100 | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → FAIL | FAIL → FAIL |
| staggered100 | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → FAIL | FAIL → FAIL |
| trajectory | PASS → BLOCKED | PASS → PASS | PASS → BLOCKED | PASS → PASS | PASS → PASS |
| unipolar | PASS → BLOCKED | PASS → PASS | PASS → BLOCKED | PASS → PASS | PASS → PASS |
| vector_anisotropic | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → FAIL | FAIL → FAIL |
| vector_overlap | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → FAIL | FAIL → FAIL |
| vector_spiral | PASS → BLOCKED | PASS → PASS | PASS → BLOCKED | PASS → PASS | PASS → PASS |
| vector_two_broad | PASS → BLOCKED | PASS → PASS | PASS → BLOCKED | PASS → PASS | PASS → PASS |
| vector_unequal_mass | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → FAIL | FAIL → FAIL |
| vector_unequal_width | FAIL → BLOCKED | FAIL → FAIL | FAIL → BLOCKED | FAIL → FAIL | FAIL → FAIL |

## The 100-Gaussian density gap

All measured positive native recipes appear below, using each original independent 100,000-draw live holdout. Modes are recomputed from those same hashed draws using the frozen coverage scorer; density measurements retain the stricter accuracy targets. These endpoint packets explain the results and add no qualification credit. The shared control is shown once; the baseline audit checks all five controls independently. Unavailable component statistics remain unavailable.

| Recipe | Task | Original gate | Precision / ≥0.97 | Modes / ==100 | Mass TV / ≤0.06 | Radial KS / ≤0.04 | Absolute covariance trace bias / ≤0.1 | Center RMS in sigma / ≤0.2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coulomb {'prior_reg': 0.01} | grid100 | FAIL | 0.2803 | 14 | 0.1423 | 0.417246 | 0.475674 | 1.29296 |
| coulomb {'prior_reg': 0.01} | rotated100 | FAIL | 0.28116 | 16 | 0.13708 | 0.39392 | 0.506447 | 1.30582 |
| coulomb {'prior_reg': 0.01} | staggered100 | FAIL | 0.32204 | 16 | 0.12469 | 0.391154 | 0.498203 | 1.2218 |
| coulomb {'prior_reg': 0.1} | grid100 | FAIL | 0.27582 | 16 | 0.16482 | 0.419203 | 0.265095 | 1.55346 |
| coulomb {'prior_reg': 0.1} | rotated100 | FAIL | 0.28306 | 14 | 0.13717 | unavailable | unavailable | unavailable |
| coulomb {'prior_reg': 0.1} | staggered100 | FAIL | 0.34003 | 22 | 0.11494 | 0.381505 | 0.463829 | 1.20381 |
| BCAP leader control | grid100 | FAIL | 0.24072 | 8 | 0.14718 | 0.490883 | 0.371385 | 1.52031 |
| BCAP leader control | rotated100 | FAIL | 0.25552 | 14 | 0.14325 | unavailable | unavailable | unavailable |
| BCAP leader control | staggered100 | FAIL | 0.30168 | 19 | 0.12277 | unavailable | unavailable | unavailable |
| covariance {'prior_reg': 0.01} | grid100 | FAIL | 0.29935 | 17 | 0.15976 | 0.411098 | 0.462706 | 1.31193 |
| covariance {'prior_reg': 0.01} | rotated100 | FAIL | 0.2257 | 10 | 0.13697 | 0.479056 | 0.498164 | 1.3965 |
| covariance {'prior_reg': 0.01} | staggered100 | FAIL | 0.34371 | 17 | 0.12345 | 0.38768 | 0.487619 | 1.20339 |
| mmd {'prior_reg': 0.1} | grid100 | FAIL | 0.24413 | 11 | 0.14798 | 0.454284 | 0.393986 | 1.45937 |
| mmd {'prior_reg': 0.1} | rotated100 | FAIL | 0.23907 | 12 | 0.13479 | 0.464625 | 0.466392 | 1.39095 |
| mmd {'prior_reg': 0.1} | staggered100 | FAIL | 0.34425 | 22 | 0.12414 | 0.366605 | 0.53531 | 1.10354 |

## Measured failure checks

These are the failed numerical checks for the selected whole recipes above. Endpoint checks explain the recorded endpoint; the sustained/confirmation rows retain the task's actual temporal gate. Original mandatory word-hold and audited terminal-window witnesses include their observation step and hash; they explain the existing verdict and earn no new qualification credit. Unavailable and blocked tasks are shown in the full reports with their original targets. Passing acquisition tasks with a failing final endpoint are excluded here because their declared earlier-confirmation gate passed.

### proximal: {'prior_proximal_strength': 1}

[Every task and measurement](proximal/README.md), including passing checks and diagnostics.

| Failed task | Measurement cohort | Measurement | Score | Target |
| --- | --- | --- | ---: | --- |
| ring16_acquisition | Sustained gate | terminal passing suffix | 0 | >= 5 |
| ring16_acquisition | Recorded endpoint | hq | 0.729492 | >= 0.85 |
| ring16_acquisition | Recorded endpoint | component_covariance_error | 3.36332 | <= 0.85 |
### coulomb: {'prior_reg': 0.01}

[Every task and measurement](coulomb/README.md), including passing checks and diagnostics.

| Failed task | Measurement cohort | Measurement | Score | Target |
| --- | --- | --- | ---: | --- |
| gaussian1d_stability | Sustained gate | stationary passing checks | 9 | == 72 |
| gaussian1d_stability | Sustained gate | deadline reacquisition | false | == true |
| gaussian1d_stability | Sustained gate | shifted hold passing checks | 1 | == 24 |
| gaussian1d_stability | Recorded endpoint | mean_error_sigma | 0.370123 | <= 0.2 |
| gaussian1d_stability | Recorded endpoint | cdf_ks | 0.125184 | <= 0.05 |
| five_word_joint_hold | Sustained gate | passing primary checks | 24 | == 25 |
| five_word_joint_hold | Sustained gate | confirmed checks | 24 | == 25 |
| five_word_joint_hold | Worst mandatory word-hold primary checks (saved failure witnesses); original step 5001, observation [f73d1a8e](coulomb/results.json) | quality_fraction | 0.782227 | >= 0.95 |
| five_word_joint_hold | Worst mandatory word-hold primary checks (saved failure witnesses); original step 5001, observation [f73d1a8e](coulomb/results.json) | modes | 4 | == 5 |
| five_word_joint_hold | Worst mandatory word-hold primary checks (saved failure witnesses); original step 5001, observation [f73d1a8e](coulomb/results.json) | mass_tv | 0.235742 | <= 0.1 |
| five_word_joint_hold | Worst mandatory word-hold primary checks (saved failure witnesses); original step 5001, observation [f73d1a8e](coulomb/results.json) | reconstruction_exact | 0 | == 1 |
| five_word_joint_hold | Worst mandatory word-hold primary checks (saved failure witnesses); original step 5001, observation [f73d1a8e](coulomb/results.json) | minimum_reconstruction_token_probability | 0.345344 | >= 0.9 |
| five_word_joint_hold | Worst mandatory word-hold independent confirmation checks (saved failure witnesses); original step 5001, observation [5e8ed52a](coulomb/results.json) | quality_fraction | 0.796875 | >= 0.95 |
| five_word_joint_hold | Worst mandatory word-hold independent confirmation checks (saved failure witnesses); original step 5001, observation [5e8ed52a](coulomb/results.json) | modes | 4 | == 5 |
| five_word_joint_hold | Worst mandatory word-hold independent confirmation checks (saved failure witnesses); original step 5001, observation [5e8ed52a](coulomb/results.json) | mass_tv | 0.209375 | <= 0.1 |
| five_word_joint_hold | Worst mandatory word-hold independent confirmation checks (saved failure witnesses); original step 5001, observation [5e8ed52a](coulomb/results.json) | reconstruction_exact | 0 | == 1 |
| five_word_joint_hold | Worst mandatory word-hold independent confirmation checks (saved failure witnesses); original step 5001, observation [5e8ed52a](coulomb/results.json) | minimum_reconstruction_token_probability | 0.345344 | >= 0.9 |
| mode_hold | Sustained gate | terminal passing suffix | 0 | >= 5 |
| mode_hold | Recorded endpoint | modes | 7 | >= 8 |
| vector_unequal_mass | Sustained gate | terminal passing suffix | 0 | >= 5 |
| vector_unequal_mass | Recorded endpoint | component_covariance_error | 1.21768 | <= 0.85 |
| vector_unequal_mass | Recorded endpoint | component_min_eigen_ratio | 0.0635782 | >= 0.15 |
| vector_unequal_width | Sustained gate | terminal passing suffix | 0 | >= 5 |
| vector_unequal_width | Recorded endpoint | sw1_normalized | 0.285169 | <= 0.18 |
| vector_unequal_width | Recorded endpoint | mass_tv | 0.301025 | <= 0.15 |
| vector_unequal_width | Recorded endpoint | component_covariance_error | 9.37526 | <= 0.85 |
| vector_anisotropic | Sustained gate | terminal passing suffix | 0 | >= 5 |
| vector_anisotropic | Recorded endpoint | sw1_normalized | 0.348019 | <= 0.18 |
| vector_anisotropic | Recorded endpoint | mass_tv | 0.289876 | <= 0.15 |
| vector_anisotropic | Recorded endpoint | component_covariance_error | 17.7193 | <= 0.85 |
| vector_overlap | Sustained gate | terminal passing suffix | 0 | >= 5 |
| vector_overlap | Recorded endpoint | sw1_normalized | 0.25137 | <= 0.18 |
| vector_overlap | Recorded endpoint | mean_error | 0.411998 | <= 0.15 |
| img_bars4 | Sustained gate | terminal passing suffix | 0 | >= 5 |
| img_bars4 | Recorded endpoint | modes | 2 | >= 4 |
| img_blobs4 | Sustained gate | terminal passing suffix | 0 | >= 5 |
| img_blobs4 | Recorded endpoint | modes | 3 | >= 4 |
| img_intensity2 | Sustained gate | terminal passing suffix | 0 | >= 5 |
| img_intensity2 | Recorded endpoint | hq | 0.5625 | >= 0.9 |
| grid100 | Sustained gate | coverage decision | false | == true |
| grid100 | Sustained gate | terminal live accuracy passing checks | 0 | == 5 |
| grid100 | Live endpoint coverage, 20000 saved draws | precision | 0.28515 | >= 0.97 |
| grid100 | Live endpoint coverage, 20000 saved draws | modes | 13 | == 100 |
| grid100 | Live endpoint coverage, 20000 saved draws | min_hq_mode_mass | 5e-05 | >= 0.005 |
| grid100 | Live endpoint coverage, 20000 saved draws | mass_tv | 0.13975 | <= 0.1 |
| grid100 | Live endpoint coverage, 20000 saved draws | max_mode_mass | 0.0212 | <= 0.02 |
| grid100 | Live endpoint coverage, 20000 saved draws | min_cov_eig_ratio | 0 | >= 0.4 |
| grid100 | Live endpoint coverage, 20000 saved draws | max_cov_eig_ratio | 4.02049 | <= 1.7 |
| grid100 | Live endpoint coverage, 20000 saved draws | max_radial_median_ratio | 2.35205 | <= 1.4 |
| grid100 | Live endpoint coverage, 20000 saved draws | passed | false | == true |
| grid100 | Live endpoint accuracy, 20000 saved draws | precision | 0.28515 | >= 0.97 |
| grid100 | Live endpoint accuracy, 20000 saved draws | mass_tv | 0.13975 | <= 0.06 |
| grid100 | Live endpoint accuracy, 20000 saved draws | accuracy_pass | false | == true |
| grid100 | Live endpoint accuracy, 20000 saved draws | frozen_pass | false | == true |
| grid100 | Live endpoint accuracy, 20000 saved draws | passed | false | == true |
| grid100 | Independent live holdout accuracy, 100000 saved draws | precision | 0.2803 | >= 0.97 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | abs_cov_trace_bias | 0.475674 | <= 0.1 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | center_rms_sigma | 1.29296 | <= 0.2 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | mass_tv | 0.1423 | <= 0.06 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | radial_ks | 0.417246 | <= 0.04 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | accuracy_pass | false | == true |
| grid100 | Independent live holdout accuracy, 100000 saved draws | frozen_pass | false | == true |
| grid100 | Independent live holdout accuracy, 100000 saved draws | passed | false | == true |
| grid100 | Independent holdout coverage recomputed from the same saved draws | precision | 0.28029 | >= 0.97 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | modes | 14 | == 100 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | min_hq_mode_mass | 4e-05 | >= 0.005 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | mass_tv | 0.1423 | <= 0.1 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | max_mode_mass | 0.02146 | <= 0.02 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | min_cov_eig_ratio | 0.0937262 | >= 0.4 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | max_cov_eig_ratio | 2.29697 | <= 1.7 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | max_radial_median_ratio | 2.26782 | <= 1.4 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | passed | false | == true |
| rotated100 | Sustained gate | coverage decision | false | == true |
| rotated100 | Sustained gate | terminal live accuracy passing checks | 0 | == 5 |
| rotated100 | Live endpoint coverage, 20000 saved draws | precision | 0.28055 | >= 0.97 |
| rotated100 | Live endpoint coverage, 20000 saved draws | modes | 16 | == 100 |
| rotated100 | Live endpoint coverage, 20000 saved draws | min_hq_mode_mass | 5e-05 | >= 0.005 |
| rotated100 | Live endpoint coverage, 20000 saved draws | mass_tv | 0.1343 | <= 0.1 |
| rotated100 | Live endpoint coverage, 20000 saved draws | max_mode_mass | 0.0233 | <= 0.02 |
| rotated100 | Live endpoint coverage, 20000 saved draws | min_cov_eig_ratio | 0 | >= 0.4 |
| rotated100 | Live endpoint coverage, 20000 saved draws | max_cov_eig_ratio | 3.34597 | <= 1.7 |
| rotated100 | Live endpoint coverage, 20000 saved draws | max_radial_median_ratio | 2.52485 | <= 1.4 |
| rotated100 | Live endpoint coverage, 20000 saved draws | passed | false | == true |
| rotated100 | Live endpoint accuracy, 20000 saved draws | precision | 0.2805 | >= 0.97 |
| rotated100 | Live endpoint accuracy, 20000 saved draws | mass_tv | 0.1343 | <= 0.06 |
| rotated100 | Live endpoint accuracy, 20000 saved draws | accuracy_pass | false | == true |
| rotated100 | Live endpoint accuracy, 20000 saved draws | frozen_pass | false | == true |
| rotated100 | Live endpoint accuracy, 20000 saved draws | passed | false | == true |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | precision | 0.28116 | >= 0.97 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | abs_cov_trace_bias | 0.506447 | <= 0.1 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | center_rms_sigma | 1.30582 | <= 0.2 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | mass_tv | 0.13708 | <= 0.06 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | radial_ks | 0.39392 | <= 0.04 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | accuracy_pass | false | == true |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | frozen_pass | false | == true |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | passed | false | == true |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | precision | 0.28117 | >= 0.97 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | modes | 16 | == 100 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | min_hq_mode_mass | 0.00013 | >= 0.005 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | mass_tv | 0.13708 | <= 0.1 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | max_mode_mass | 0.02221 | <= 0.02 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | min_cov_eig_ratio | 0.00404388 | >= 0.4 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | max_cov_eig_ratio | 2.63969 | <= 1.7 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | max_radial_median_ratio | 2.43032 | <= 1.4 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | passed | false | == true |
| staggered100 | Sustained gate | coverage decision | false | == true |
| staggered100 | Sustained gate | terminal live accuracy passing checks | 0 | == 5 |
| staggered100 | Live endpoint coverage, 20000 saved draws | precision | 0.32065 | >= 0.97 |
| staggered100 | Live endpoint coverage, 20000 saved draws | modes | 16 | == 100 |
| staggered100 | Live endpoint coverage, 20000 saved draws | min_hq_mode_mass | 0.00015 | >= 0.005 |
| staggered100 | Live endpoint coverage, 20000 saved draws | mass_tv | 0.13065 | <= 0.1 |
| staggered100 | Live endpoint coverage, 20000 saved draws | max_mode_mass | 0.0246 | <= 0.02 |
| staggered100 | Live endpoint coverage, 20000 saved draws | min_cov_eig_ratio | 0.000770933 | >= 0.4 |
| staggered100 | Live endpoint coverage, 20000 saved draws | max_cov_eig_ratio | 2.85381 | <= 1.7 |
| staggered100 | Live endpoint coverage, 20000 saved draws | max_radial_median_ratio | 2.33324 | <= 1.4 |
| staggered100 | Live endpoint coverage, 20000 saved draws | passed | false | == true |
| staggered100 | Live endpoint accuracy, 20000 saved draws | precision | 0.32065 | >= 0.97 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | abs_cov_trace_bias | 0.458409 | <= 0.1 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | center_rms_sigma | 1.25392 | <= 0.2 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | mass_tv | 0.13065 | <= 0.06 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | radial_ks | 0.389148 | <= 0.04 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | accuracy_pass | false | == true |
| staggered100 | Live endpoint accuracy, 20000 saved draws | frozen_pass | false | == true |
| staggered100 | Live endpoint accuracy, 20000 saved draws | passed | false | == true |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | precision | 0.32204 | >= 0.97 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | abs_cov_trace_bias | 0.498203 | <= 0.1 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | center_rms_sigma | 1.2218 | <= 0.2 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | mass_tv | 0.12469 | <= 0.06 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | radial_ks | 0.391154 | <= 0.04 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | accuracy_pass | false | == true |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | frozen_pass | false | == true |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | passed | false | == true |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | precision | 0.32204 | >= 0.97 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | modes | 16 | == 100 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | min_hq_mode_mass | 0.00024 | >= 0.005 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | mass_tv | 0.12469 | <= 0.1 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | max_mode_mass | 0.02449 | <= 0.02 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | min_cov_eig_ratio | 0.219406 | >= 0.4 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | max_cov_eig_ratio | 2.64344 | <= 1.7 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | max_radial_median_ratio | 2.2044 | <= 1.4 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | passed | false | == true |
### sigreg: {'prior_reg': 0.001}

[Every task and measurement](sigreg/README.md), including passing checks and diagnostics.

| Failed task | Measurement cohort | Measurement | Score | Target |
| --- | --- | --- | ---: | --- |
| ring16_acquisition | Sustained gate | terminal passing suffix | 0 | >= 5 |
| ring16_acquisition | Recorded endpoint | component_covariance_error | 0.884537 | <= 0.85 |
### covariance: {'prior_reg': 0.01}

[Every task and measurement](covariance/README.md), including passing checks and diagnostics.

| Failed task | Measurement cohort | Measurement | Score | Target |
| --- | --- | --- | ---: | --- |
| gaussian1d_stability | Sustained gate | stationary passing checks | 0 | == 72 |
| gaussian1d_stability | Sustained gate | deadline reacquisition | false | == true |
| gaussian1d_stability | Sustained gate | shifted hold passing checks | 0 | == 24 |
| gaussian1d_stability | Recorded endpoint | std_ratio | 1.22242 | >= 0.8 and <= 1.2 |
| gaussian1d_stability | Recorded endpoint | cdf_ks | 0.13339 | <= 0.05 |
| five_word_joint_hold | Sustained gate | passing primary checks | 23 | == 25 |
| five_word_joint_hold | Sustained gate | confirmed checks | 23 | == 25 |
| five_word_joint_hold | Worst mandatory word-hold primary checks (saved failure witnesses); original step 5167, observation [16738db1](covariance/results.json) | modes | 4 | == 5 |
| five_word_joint_hold | Worst mandatory word-hold primary checks (saved failure witnesses); original step 5167, observation [16738db1](covariance/results.json) | mass_tv | 0.203516 | <= 0.1 |
| five_word_joint_hold | Worst mandatory word-hold primary checks (saved failure witnesses); original step 6167, observation [f6bcfd6c](covariance/results.json) | minimum_reconstruction_token_probability | 0.796768 | >= 0.9 |
| five_word_joint_hold | Worst mandatory word-hold independent confirmation checks (saved failure witnesses); original step 5167, observation [cc6814fa](covariance/results.json) | modes | 4 | == 5 |
| five_word_joint_hold | Worst mandatory word-hold independent confirmation checks (saved failure witnesses); original step 5167, observation [cc6814fa](covariance/results.json) | mass_tv | 0.204687 | <= 0.1 |
| five_word_joint_hold | Worst mandatory word-hold independent confirmation checks (saved failure witnesses); original step 6167, observation [4aa6fd3a](covariance/results.json) | minimum_reconstruction_token_probability | 0.796768 | >= 0.9 |
| vector_unequal_mass | Sustained gate | terminal passing suffix | 0 | >= 5 |
| vector_unequal_mass | Recorded endpoint | component_covariance_error | 4.10357 | <= 0.85 |
| vector_unequal_mass | Recorded endpoint | component_min_eigen_ratio | 0.00366343 | >= 0.15 |
| vector_unequal_width | Sustained gate | terminal passing suffix | 0 | >= 5 |
| vector_unequal_width | Recorded endpoint | sw1_normalized | 0.308444 | <= 0.18 |
| vector_unequal_width | Recorded endpoint | mass_tv | 0.300049 | <= 0.15 |
| vector_unequal_width | Recorded endpoint | component_covariance_error | 6.73499 | <= 0.85 |
| vector_unequal_width | Recorded endpoint | component_min_eigen_ratio | 0.108822 | >= 0.15 |
| vector_anisotropic | Sustained gate | terminal passing suffix | 0 | >= 5 |
| vector_anisotropic | Recorded endpoint | sw1_normalized | 0.291984 | <= 0.18 |
| vector_anisotropic | Recorded endpoint | mass_tv | 0.299967 | <= 0.15 |
| vector_anisotropic | Recorded endpoint | component_covariance_error | 1.04532 | <= 0.85 |
| vector_overlap | Sustained gate | terminal passing suffix | 1 | >= 5 |
| img_bars4 | Sustained gate | terminal passing suffix | 0 | >= 5 |
| img_bars4 | Recorded endpoint | modes | 2 | >= 4 |
| img_blobs4 | Sustained gate | terminal passing suffix | 0 | >= 5 |
| img_blobs4 | Recorded endpoint | modes | 3 | >= 4 |
| img_blobs4 | Recorded endpoint | hq | 0.8125 | >= 0.9 |
| img_intensity2 | Sustained gate | terminal passing suffix | 0 | >= 5 |
| img_intensity2 | Recorded endpoint | hq | 0.71875 | >= 0.9 |
| grid100 | Sustained gate | coverage decision | false | == true |
| grid100 | Sustained gate | terminal live accuracy passing checks | 0 | == 5 |
| grid100 | Live endpoint coverage, 20000 saved draws | precision | 0.30345 | >= 0.97 |
| grid100 | Live endpoint coverage, 20000 saved draws | modes | 17 | == 100 |
| grid100 | Live endpoint coverage, 20000 saved draws | min_hq_mode_mass | 0.0002 | >= 0.005 |
| grid100 | Live endpoint coverage, 20000 saved draws | mass_tv | 0.16115 | <= 0.1 |
| grid100 | Live endpoint coverage, 20000 saved draws | max_mode_mass | 0.02975 | <= 0.02 |
| grid100 | Live endpoint coverage, 20000 saved draws | min_cov_eig_ratio | 0.0429869 | >= 0.4 |
| grid100 | Live endpoint coverage, 20000 saved draws | max_cov_eig_ratio | 2.67322 | <= 1.7 |
| grid100 | Live endpoint coverage, 20000 saved draws | max_radial_median_ratio | 2.37525 | <= 1.4 |
| grid100 | Live endpoint coverage, 20000 saved draws | passed | false | == true |
| grid100 | Live endpoint accuracy, 20000 saved draws | precision | 0.30345 | >= 0.97 |
| grid100 | Live endpoint accuracy, 20000 saved draws | abs_cov_trace_bias | 0.43602 | <= 0.1 |
| grid100 | Live endpoint accuracy, 20000 saved draws | center_rms_sigma | 1.32115 | <= 0.2 |
| grid100 | Live endpoint accuracy, 20000 saved draws | mass_tv | 0.16115 | <= 0.06 |
| grid100 | Live endpoint accuracy, 20000 saved draws | radial_ks | 0.408031 | <= 0.04 |
| grid100 | Live endpoint accuracy, 20000 saved draws | accuracy_pass | false | == true |
| grid100 | Live endpoint accuracy, 20000 saved draws | frozen_pass | false | == true |
| grid100 | Live endpoint accuracy, 20000 saved draws | passed | false | == true |
| grid100 | Independent live holdout accuracy, 100000 saved draws | precision | 0.29935 | >= 0.97 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | abs_cov_trace_bias | 0.462706 | <= 0.1 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | center_rms_sigma | 1.31193 | <= 0.2 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | mass_tv | 0.15976 | <= 0.06 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | radial_ks | 0.411098 | <= 0.04 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | accuracy_pass | false | == true |
| grid100 | Independent live holdout accuracy, 100000 saved draws | frozen_pass | false | == true |
| grid100 | Independent live holdout accuracy, 100000 saved draws | passed | false | == true |
| grid100 | Independent holdout coverage recomputed from the same saved draws | precision | 0.29934 | >= 0.97 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | modes | 17 | == 100 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | min_hq_mode_mass | 0.00032 | >= 0.005 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | mass_tv | 0.15976 | <= 0.1 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | max_mode_mass | 0.02966 | <= 0.02 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | min_cov_eig_ratio | 0.0927582 | >= 0.4 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | max_cov_eig_ratio | 2.37681 | <= 1.7 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | max_radial_median_ratio | 2.2375 | <= 1.4 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | passed | false | == true |
| rotated100 | Sustained gate | coverage decision | false | == true |
| rotated100 | Sustained gate | terminal live accuracy passing checks | 0 | == 5 |
| rotated100 | Live endpoint coverage, 20000 saved draws | precision | 0.2246 | >= 0.97 |
| rotated100 | Live endpoint coverage, 20000 saved draws | modes | 8 | == 100 |
| rotated100 | Live endpoint coverage, 20000 saved draws | min_hq_mode_mass | 0.0001 | >= 0.005 |
| rotated100 | Live endpoint coverage, 20000 saved draws | mass_tv | 0.13485 | <= 0.1 |
| rotated100 | Live endpoint coverage, 20000 saved draws | max_mode_mass | 0.02365 | <= 0.02 |
| rotated100 | Live endpoint coverage, 20000 saved draws | min_cov_eig_ratio | 0 | >= 0.4 |
| rotated100 | Live endpoint coverage, 20000 saved draws | max_cov_eig_ratio | 2.99274 | <= 1.7 |
| rotated100 | Live endpoint coverage, 20000 saved draws | max_radial_median_ratio | 2.37765 | <= 1.4 |
| rotated100 | Live endpoint coverage, 20000 saved draws | passed | false | == true |
| rotated100 | Live endpoint accuracy, 20000 saved draws | precision | 0.2246 | >= 0.97 |
| rotated100 | Live endpoint accuracy, 20000 saved draws | abs_cov_trace_bias | 0.40832 | <= 0.1 |
| rotated100 | Live endpoint accuracy, 20000 saved draws | center_rms_sigma | 1.41948 | <= 0.2 |
| rotated100 | Live endpoint accuracy, 20000 saved draws | mass_tv | 0.13485 | <= 0.06 |
| rotated100 | Live endpoint accuracy, 20000 saved draws | radial_ks | 0.474401 | <= 0.04 |
| rotated100 | Live endpoint accuracy, 20000 saved draws | accuracy_pass | false | == true |
| rotated100 | Live endpoint accuracy, 20000 saved draws | frozen_pass | false | == true |
| rotated100 | Live endpoint accuracy, 20000 saved draws | passed | false | == true |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | precision | 0.2257 | >= 0.97 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | abs_cov_trace_bias | 0.498164 | <= 0.1 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | center_rms_sigma | 1.3965 | <= 0.2 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | mass_tv | 0.13697 | <= 0.06 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | radial_ks | 0.479056 | <= 0.04 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | accuracy_pass | false | == true |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | frozen_pass | false | == true |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | passed | false | == true |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | precision | 0.22569 | >= 0.97 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | modes | 10 | == 100 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | min_hq_mode_mass | 0.00014 | >= 0.005 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | mass_tv | 0.13697 | <= 0.1 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | max_mode_mass | 0.02335 | <= 0.02 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | min_cov_eig_ratio | 0.056588 | >= 0.4 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | max_cov_eig_ratio | 2.42662 | <= 1.7 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | max_radial_median_ratio | 2.32489 | <= 1.4 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | passed | false | == true |
| staggered100 | Sustained gate | coverage decision | false | == true |
| staggered100 | Sustained gate | terminal live accuracy passing checks | 0 | == 5 |
| staggered100 | Live endpoint coverage, 20000 saved draws | precision | 0.34075 | >= 0.97 |
| staggered100 | Live endpoint coverage, 20000 saved draws | modes | 18 | == 100 |
| staggered100 | Live endpoint coverage, 20000 saved draws | min_hq_mode_mass | 0.0002 | >= 0.005 |
| staggered100 | Live endpoint coverage, 20000 saved draws | mass_tv | 0.12895 | <= 0.1 |
| staggered100 | Live endpoint coverage, 20000 saved draws | max_mode_mass | 0.02295 | <= 0.02 |
| staggered100 | Live endpoint coverage, 20000 saved draws | min_cov_eig_ratio | 0.0159732 | >= 0.4 |
| staggered100 | Live endpoint coverage, 20000 saved draws | max_cov_eig_ratio | 3.35805 | <= 1.7 |
| staggered100 | Live endpoint coverage, 20000 saved draws | max_radial_median_ratio | 2.14426 | <= 1.4 |
| staggered100 | Live endpoint coverage, 20000 saved draws | passed | false | == true |
| staggered100 | Live endpoint accuracy, 20000 saved draws | precision | 0.34075 | >= 0.97 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | abs_cov_trace_bias | 0.477489 | <= 0.1 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | center_rms_sigma | 1.22672 | <= 0.2 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | mass_tv | 0.12895 | <= 0.06 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | radial_ks | 0.395368 | <= 0.04 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | accuracy_pass | false | == true |
| staggered100 | Live endpoint accuracy, 20000 saved draws | frozen_pass | false | == true |
| staggered100 | Live endpoint accuracy, 20000 saved draws | passed | false | == true |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | precision | 0.34371 | >= 0.97 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | abs_cov_trace_bias | 0.487619 | <= 0.1 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | center_rms_sigma | 1.20339 | <= 0.2 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | mass_tv | 0.12345 | <= 0.06 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | radial_ks | 0.38768 | <= 0.04 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | accuracy_pass | false | == true |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | frozen_pass | false | == true |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | passed | false | == true |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | precision | 0.3437 | >= 0.97 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | modes | 17 | == 100 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | min_hq_mode_mass | 0.00039 | >= 0.005 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | mass_tv | 0.12345 | <= 0.1 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | max_mode_mass | 0.02311 | <= 0.02 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | min_cov_eig_ratio | 0.0349185 | >= 0.4 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | max_cov_eig_ratio | 2.38622 | <= 1.7 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | max_radial_median_ratio | 2.18468 | <= 1.4 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | passed | false | == true |
### mmd: {'prior_reg': 0.1}

[Every task and measurement](mmd/README.md), including passing checks and diagnostics.

| Failed task | Measurement cohort | Measurement | Score | Target |
| --- | --- | --- | ---: | --- |
| gaussian1d_stability | Sustained gate | stationary passing checks | 3 | == 72 |
| gaussian1d_stability | Sustained gate | deadline reacquisition | false | == true |
| gaussian1d_stability | Sustained gate | shifted hold passing checks | 0 | == 24 |
| gaussian1d_stability | Recorded endpoint | cdf_ks | 0.0733304 | <= 0.05 |
| five_word_joint_hold | Sustained gate | passing primary checks | 24 | == 25 |
| five_word_joint_hold | Sustained gate | confirmed checks | 24 | == 25 |
| five_word_joint_hold | Worst mandatory word-hold primary checks (saved failure witnesses); original step 2001, observation [e5f82f05](mmd/results.json) | reconstruction_exact | 0 | == 1 |
| five_word_joint_hold | Worst mandatory word-hold primary checks (saved failure witnesses); original step 2001, observation [e5f82f05](mmd/results.json) | minimum_reconstruction_token_probability | 0.19466 | >= 0.9 |
| five_word_joint_hold | Worst mandatory word-hold independent confirmation checks (saved failure witnesses); original step 2001, observation [e9f91a95](mmd/results.json) | reconstruction_exact | 0 | == 1 |
| five_word_joint_hold | Worst mandatory word-hold independent confirmation checks (saved failure witnesses); original step 2001, observation [e9f91a95](mmd/results.json) | minimum_reconstruction_token_probability | 0.19466 | >= 0.9 |
| mode_hold | Sustained gate | terminal passing suffix | 0 | >= 5 |
| mode_hold | Recorded endpoint | hq | 0.881836 | >= 0.9 |
| mode_hold | Original required terminal failure witness (saved, no new credit); original step 1150, observation [3244c95e](mmd/audit.json) | hq | 0.875488 | >= 0.9 |
| vector_unequal_mass | Sustained gate | terminal passing suffix | 0 | >= 5 |
| vector_unequal_mass | Recorded endpoint | component_min_eigen_ratio | 0.0138622 | >= 0.15 |
| vector_unequal_mass | Original required terminal failure witness (saved, no new credit); original step 1000, observation [f534c3cb](mmd/audit.json) | component_covariance_error | 0.992828 | <= 0.85 |
| vector_unequal_mass | Original required terminal failure witness (saved, no new credit); original step 1000, observation [f534c3cb](mmd/audit.json) | component_min_eigen_ratio | 0.0267888 | >= 0.15 |
| vector_unequal_width | Sustained gate | terminal passing suffix | 0 | >= 5 |
| vector_unequal_width | Recorded endpoint | sw1_normalized | 0.264519 | <= 0.18 |
| vector_unequal_width | Recorded endpoint | mass_tv | 0.258789 | <= 0.15 |
| vector_unequal_width | Recorded endpoint | component_covariance_error | 7.86101 | <= 0.85 |
| vector_unequal_width | Original required terminal failure witness (saved, no new credit); original step 1000, observation [1a741ed9](mmd/audit.json) | sw1_normalized | 0.27316 | <= 0.18 |
| vector_unequal_width | Original required terminal failure witness (saved, no new credit); original step 1000, observation [1a741ed9](mmd/audit.json) | mass_tv | 0.260742 | <= 0.15 |
| vector_unequal_width | Original required terminal failure witness (saved, no new credit); original step 1000, observation [1a741ed9](mmd/audit.json) | component_covariance_error | 11.3894 | <= 0.85 |
| vector_unequal_width | Original required terminal failure witness (saved, no new credit); original step 1000, observation [1a741ed9](mmd/audit.json) | component_min_eigen_ratio | 0.0916025 | >= 0.15 |
| vector_anisotropic | Sustained gate | terminal passing suffix | 0 | >= 5 |
| vector_anisotropic | Recorded endpoint | sw1_normalized | 0.274875 | <= 0.18 |
| vector_anisotropic | Recorded endpoint | mass_tv | 0.264323 | <= 0.15 |
| vector_anisotropic | Recorded endpoint | component_covariance_error | 2.6919 | <= 0.85 |
| vector_anisotropic | Original required terminal failure witness (saved, no new credit); original step 1000, observation [16e1e937](mmd/audit.json) | sw1_normalized | 0.273368 | <= 0.18 |
| vector_anisotropic | Original required terminal failure witness (saved, no new credit); original step 1000, observation [16e1e937](mmd/audit.json) | mass_tv | 0.250895 | <= 0.15 |
| vector_anisotropic | Original required terminal failure witness (saved, no new credit); original step 1000, observation [16e1e937](mmd/audit.json) | component_covariance_error | 2.42206 | <= 0.85 |
| vector_overlap | Sustained gate | terminal passing suffix | 0 | >= 5 |
| vector_overlap | Recorded endpoint | mean_error | 0.15257 | <= 0.15 |
| vector_overlap | Original required terminal failure witness (saved, no new credit); original step 1000, observation [eed6d28b](mmd/audit.json) | mean_error | 0.19106 | <= 0.15 |
| img_bars4 | Sustained gate | terminal passing suffix | 0 | >= 5 |
| img_bars4 | Recorded endpoint | modes | 2 | >= 4 |
| img_bars4 | Original required terminal failure witness (saved, no new credit); original step 500, observation [3edaed35](mmd/audit.json) | modes | 2 | >= 4 |
| img_blobs4 | Sustained gate | terminal passing suffix | 0 | >= 5 |
| img_blobs4 | Recorded endpoint | modes | 3 | >= 4 |
| img_blobs4 | Original required terminal failure witness (saved, no new credit); original step 500, observation [d6038028](mmd/audit.json) | modes | 3 | >= 4 |
| img_intensity2 | Sustained gate | terminal passing suffix | 0 | >= 5 |
| img_intensity2 | Recorded endpoint | hq | 0.875 | >= 0.9 |
| img_intensity2 | Original required terminal failure witness (saved, no new credit); original step 500, observation [123522c8](mmd/audit.json) | modes | 1 | >= 2 |
| img_intensity2 | Original required terminal failure witness (saved, no new credit); original step 500, observation [123522c8](mmd/audit.json) | hq | 0.59375 | >= 0.9 |
| grid100 | Sustained gate | coverage decision | false | == true |
| grid100 | Sustained gate | terminal live accuracy passing checks | 0 | == 5 |
| grid100 | Live endpoint coverage, 20000 saved draws | precision | 0.2477 | >= 0.97 |
| grid100 | Live endpoint coverage, 20000 saved draws | modes | 12 | == 100 |
| grid100 | Live endpoint coverage, 20000 saved draws | min_hq_mode_mass | 5e-05 | >= 0.005 |
| grid100 | Live endpoint coverage, 20000 saved draws | mass_tv | 0.1493 | <= 0.1 |
| grid100 | Live endpoint coverage, 20000 saved draws | max_mode_mass | 0.0227 | <= 0.02 |
| grid100 | Live endpoint coverage, 20000 saved draws | min_cov_eig_ratio | 0 | >= 0.4 |
| grid100 | Live endpoint coverage, 20000 saved draws | max_cov_eig_ratio | 3.05298 | <= 1.7 |
| grid100 | Live endpoint coverage, 20000 saved draws | max_radial_median_ratio | 2.47669 | <= 1.4 |
| grid100 | Live endpoint coverage, 20000 saved draws | passed | false | == true |
| grid100 | Live endpoint accuracy, 20000 saved draws | precision | 0.2477 | >= 0.97 |
| grid100 | Live endpoint accuracy, 20000 saved draws | mass_tv | 0.1493 | <= 0.06 |
| grid100 | Live endpoint accuracy, 20000 saved draws | accuracy_pass | false | == true |
| grid100 | Live endpoint accuracy, 20000 saved draws | frozen_pass | false | == true |
| grid100 | Live endpoint accuracy, 20000 saved draws | passed | false | == true |
| grid100 | Independent live holdout accuracy, 100000 saved draws | precision | 0.24413 | >= 0.97 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | abs_cov_trace_bias | 0.393986 | <= 0.1 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | center_rms_sigma | 1.45937 | <= 0.2 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | mass_tv | 0.14798 | <= 0.06 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | radial_ks | 0.454284 | <= 0.04 |
| grid100 | Independent live holdout accuracy, 100000 saved draws | accuracy_pass | false | == true |
| grid100 | Independent live holdout accuracy, 100000 saved draws | frozen_pass | false | == true |
| grid100 | Independent live holdout accuracy, 100000 saved draws | passed | false | == true |
| grid100 | Independent holdout coverage recomputed from the same saved draws | precision | 0.24409 | >= 0.97 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | modes | 11 | == 100 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | min_hq_mode_mass | 7e-05 | >= 0.005 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | mass_tv | 0.14798 | <= 0.1 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | max_mode_mass | 0.02196 | <= 0.02 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | min_cov_eig_ratio | 0.0252797 | >= 0.4 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | max_cov_eig_ratio | 2.62537 | <= 1.7 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | max_radial_median_ratio | 2.35915 | <= 1.4 |
| grid100 | Independent holdout coverage recomputed from the same saved draws | passed | false | == true |
| rotated100 | Sustained gate | coverage decision | false | == true |
| rotated100 | Sustained gate | terminal live accuracy passing checks | 0 | == 5 |
| rotated100 | Live endpoint coverage, 20000 saved draws | precision | 0.23825 | >= 0.97 |
| rotated100 | Live endpoint coverage, 20000 saved draws | modes | 12 | == 100 |
| rotated100 | Live endpoint coverage, 20000 saved draws | min_hq_mode_mass | 5e-05 | >= 0.005 |
| rotated100 | Live endpoint coverage, 20000 saved draws | mass_tv | 0.13245 | <= 0.1 |
| rotated100 | Live endpoint coverage, 20000 saved draws | max_mode_mass | 0.02455 | <= 0.02 |
| rotated100 | Live endpoint coverage, 20000 saved draws | min_cov_eig_ratio | 0 | >= 0.4 |
| rotated100 | Live endpoint coverage, 20000 saved draws | max_cov_eig_ratio | 3.89194 | <= 1.7 |
| rotated100 | Live endpoint coverage, 20000 saved draws | max_radial_median_ratio | 2.50343 | <= 1.4 |
| rotated100 | Live endpoint coverage, 20000 saved draws | passed | false | == true |
| rotated100 | Live endpoint accuracy, 20000 saved draws | precision | 0.2383 | >= 0.97 |
| rotated100 | Live endpoint accuracy, 20000 saved draws | mass_tv | 0.13245 | <= 0.06 |
| rotated100 | Live endpoint accuracy, 20000 saved draws | accuracy_pass | false | == true |
| rotated100 | Live endpoint accuracy, 20000 saved draws | frozen_pass | false | == true |
| rotated100 | Live endpoint accuracy, 20000 saved draws | passed | false | == true |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | precision | 0.23907 | >= 0.97 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | abs_cov_trace_bias | 0.466392 | <= 0.1 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | center_rms_sigma | 1.39095 | <= 0.2 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | mass_tv | 0.13479 | <= 0.06 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | radial_ks | 0.464625 | <= 0.04 |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | accuracy_pass | false | == true |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | frozen_pass | false | == true |
| rotated100 | Independent live holdout accuracy, 100000 saved draws | passed | false | == true |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | precision | 0.23905 | >= 0.97 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | modes | 12 | == 100 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | min_hq_mode_mass | 0.00017 | >= 0.005 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | mass_tv | 0.13479 | <= 0.1 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | max_mode_mass | 0.02337 | <= 0.02 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | min_cov_eig_ratio | 0.02394 | >= 0.4 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | max_cov_eig_ratio | 2.3305 | <= 1.7 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | max_radial_median_ratio | 2.36499 | <= 1.4 |
| rotated100 | Independent holdout coverage recomputed from the same saved draws | passed | false | == true |
| staggered100 | Sustained gate | coverage decision | false | == true |
| staggered100 | Sustained gate | terminal live accuracy passing checks | 0 | == 5 |
| staggered100 | Live endpoint coverage, 20000 saved draws | precision | 0.34405 | >= 0.97 |
| staggered100 | Live endpoint coverage, 20000 saved draws | modes | 21 | == 100 |
| staggered100 | Live endpoint coverage, 20000 saved draws | min_hq_mode_mass | 0.0006 | >= 0.005 |
| staggered100 | Live endpoint coverage, 20000 saved draws | mass_tv | 0.1303 | <= 0.1 |
| staggered100 | Live endpoint coverage, 20000 saved draws | max_mode_mass | 0.02395 | <= 0.02 |
| staggered100 | Live endpoint coverage, 20000 saved draws | min_cov_eig_ratio | 0.243241 | >= 0.4 |
| staggered100 | Live endpoint coverage, 20000 saved draws | max_cov_eig_ratio | 3.01105 | <= 1.7 |
| staggered100 | Live endpoint coverage, 20000 saved draws | max_radial_median_ratio | 2.32797 | <= 1.4 |
| staggered100 | Live endpoint coverage, 20000 saved draws | passed | false | == true |
| staggered100 | Live endpoint accuracy, 20000 saved draws | precision | 0.344 | >= 0.97 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | abs_cov_trace_bias | 0.448737 | <= 0.1 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | center_rms_sigma | 1.16117 | <= 0.2 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | mass_tv | 0.1303 | <= 0.06 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | radial_ks | 0.371038 | <= 0.04 |
| staggered100 | Live endpoint accuracy, 20000 saved draws | accuracy_pass | false | == true |
| staggered100 | Live endpoint accuracy, 20000 saved draws | frozen_pass | false | == true |
| staggered100 | Live endpoint accuracy, 20000 saved draws | passed | false | == true |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | precision | 0.34425 | >= 0.97 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | abs_cov_trace_bias | 0.53531 | <= 0.1 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | center_rms_sigma | 1.10354 | <= 0.2 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | mass_tv | 0.12414 | <= 0.06 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | radial_ks | 0.366605 | <= 0.04 |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | accuracy_pass | false | == true |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | frozen_pass | false | == true |
| staggered100 | Independent live holdout accuracy, 100000 saved draws | passed | false | == true |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | precision | 0.34425 | >= 0.97 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | modes | 22 | == 100 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | min_hq_mode_mass | 0.00039 | >= 0.005 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | mass_tv | 0.12414 | <= 0.1 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | max_mode_mass | 0.02389 | <= 0.02 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | max_cov_eig_ratio | 2.47099 | <= 1.7 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | max_radial_median_ratio | 2.0228 | <= 1.4 |
| staggered100 | Independent holdout coverage recomputed from the same saved draws | passed | false | == true |

## Charged work and execution

The finite reservation is 43,020 seconds per arm, 172,080 per four-arm mechanism, bounded by 180,000 seconds per mechanism and 900,000 overall. Charged time below includes retained infrastructure predecessors. Physical placement and sharing changed during the execution repair; these charges are accounting, not a speed comparison.

| Mechanism | Control charged seconds | Three-strength search charged seconds |
| --- | ---: | ---: |
| proximal | 2597.349 | 2600.027 |
| coulomb | 2592.576 | 5335.699 |
| sigreg | 2695.542 | 2660.935 |
| covariance | 2702.115 | 3903.762 |
| mmd | 2591.662 | 4253.125 |

The [baseline reproduction audit](baseline_audit.json), reproduced by [baseline_audit.py](baseline_audit.py), compares each fresh control's original gates and all previously published metric values with the archived leader. Additional current component arrays are listed separately; missing or invalid evidence cannot inherit an archived pass. The [aggregate integrity receipt](aggregate-validation.json) checks all 560 task cells, 20,105 numerical measurement rows, relative links and 329 copied task GIFs. [Infrastructure receipt](infrastructure.json), [cross-control word-state audit](control_word_audit.json), and [saved-media validation](saved-media-validation.json) record the interruption/retry identities and numerical checks. Original deadlines, recipes, task gates and update allowances were retained. The initial five interrupted word controls were retried for GPU context contention; there were no scientific retries. New work subsequently used a shared admission lock with one new worker per physical GPU. The native and toy GIF catalogs in each detailed report are rendered from saved actual training evidence with zero added updates, samples or rescored observations.


## Executed sources

Training logs, JSONL streams, checkpoints and state dumps remain outside Git under `/mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/`. Compact reports bind original receipt certificates and hashes, source/runtime/compute profiles, recipe and task contracts; saved native holdout scores are recomputed from the same hashed draws without training or RNG consumption.

Read the operational summary and collector logs locally; neither command launches work:

```sh
python reports/forge/bcap-prior-search/status.py
python reports/forge/bcap-prior-search/status.py --logs | xargs -r tail -F
```

The second command follows the currently active worker logs; rerun it to select newly admitted tasks. Completed worker logs remain in the archive under each mechanism's queue directory. The collector driver logs preserve the original launch and infrastructure repair history.

- proximal: commit `a28c9631d59e2977616c0375e2a07c73171a9930`, source digest `652cc41cc17bd9c92fb79eee4e186f2e1e01036213ae1c7a02f10ce7b86b3a2a`; [compact metrics and provenance](proximal/results.json).
- coulomb: commit `2ce01aec33c644ded0cf7f29b130bdb091e643ff`, source digest `f56cd32fd97769f6863f8709efd3c96a4137349b46882124c713d1917d9d062c`; [compact metrics and provenance](coulomb/results.json).
- sigreg: commit `a2a9183f042586968fe93daad1db5984c4621ddf`, source digest `af27fe8e557defb5d97b82e8b4b3b6f48f70f22eef863da2921734e45e2adf37`; [compact metrics and provenance](sigreg/results.json).
- covariance: commit `18e2aa1018f25de8c447a59287025df6f6aded3e`, source digest `c63305181b75bbfae337143b6836397aa998cadbf172581859031a01c90892c2`; [compact metrics and provenance](covariance/results.json).
- mmd: commit `e85cabc484140cd8afa3c44ce2abe1b93daf0443`, source digest `4a4dd33ea938ea6a48f7c6c5faa83c1237e87f41b28d2848e51dbdf99847d04e`; [compact metrics and provenance](mmd/results.json).
