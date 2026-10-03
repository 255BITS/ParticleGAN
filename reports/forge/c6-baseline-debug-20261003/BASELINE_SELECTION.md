# Exact original Atlas/E22 baseline for causal debugging

Use the **original C6 shared setting `lr=0.0053125`, `prior_lr_mult=1.5`**, preserving each host's resolved Recipe and the original `8021a1c50c4aff90ddea5010d368cffdc857b2f6` sampling contract. Both families have **2 original PASS / 8 required, 1 added-study PASS / 8, 6 UNKNOWN**. Among the 32 retained whole configurations, all others have at most one original pass. This chooses a useful observed baseline for debugging; it does not replace the [primary policy board](../policy-family-inventory.md), rank incompatible cohorts together, qualify a new checkout, or adopt a default.

The [machine-readable proof](BASELINE_SELECTION.json) retains all 32 whole configurations with their own source, denominator and status, plus all six distinct full Recipes. Its inputs were read only. No checkpoints or sample arrays were deserialized, no modules or scorers ran, and no draws, training, GPU operations, Git commands or remote requests occurred.

| Family | Original C6 configuration ID | Intensity2 | Broad Gaussian mixture | Whole configuration |
| --- | --- | --- | --- | --- |
| Atlas | `atlas--abe34fb6f0c5f7802cec80e5a2d88aaffb4aa85e55e4021f646457630abec4f0` | Original PASS; study PASS | Original PASS; study INCOMPLETE | INCOMPLETE; six quality cases UNKNOWN |
| E22 | `e22--5376bd11b0af6ba9ffe3151f3fbc6604e09b8df896869241676f2559e4f730be` | Original PASS; study PASS | Original PASS; study INCOMPLETE | INCOMPLETE; six quality cases UNKNOWN |

The display tie-break can select Atlas `.006375 / P1.5` because it shares one study pass. That setting has only one original pass. Selecting `.0053125 / P1.5` here uses the explicitly requested original-pass criterion, without editing the board or borrowing another configuration's good cases.

## Exact settings to retain

Only the LR and prior-rate multiplier were shared search knobs. Host resources and other task-owned settings remain part of the configuration.

| Host group | Particles / latent width / batch | Budget | D multiplier | Betas | Prior regularizer | Nominal G / D / prior rates |
| --- | --- | --- | --- | --- | --- | --- |
| Both original transpose12 image controls | 32 / 8 / 32 | 600 | 1 | `(0, .999)` | 0 | `.0053125 / .0053125 / .00796875` |
| Broad, unequal-mass and anisotropic vectors | 256 / 4 / 128 | 1,200 | 1.5 | `(0, .99)` | .05 | `.0053125 / .00796875 / .00796875` |
| Three static native 100-mode controls | 20,000 / 2 / 2,048 | 7,000 | 1 | `(0, .999)` | 0 | `.0053125 / .0053125 / .00796875` |

These are nominal optimizer rates. Stationarity, latent damping, row evidence, payoff damping and feature routing can change effective rates and displacement. The report makes no claim about measured movement.

Both families retain the KA2 critic formulation, penalty coefficient 3, `critic_r1_real=True`, `critic_payoff_damping=True`, `amsgrad=True`, `continuous_policy='dv12'`, `lr_control='stationarity'`, learned output noise initially `.029`, `serve_average=4`, and `total_steps=None`. Full external execution limits remain 600 / 1,200 / 7,000. Both use a learnable `ParticlePrior`, with `sigma_rel=0`; `Recipe.standardize=True` is removed by the particle factory rather than enabling standardized MoG reads. E22 retains reference kNN. Atlas adds automatic feature-cell selection, 128 cells and the settled reopen guard. Matching small-host results does not make their complete policy states or algorithms equivalent.

## Required domains and observation contract

All use training seed 24002, isolated evaluation seed 34002, 24 post-update primary scoring observations plus update 0, and nine media frames. Media-only observations do not enter the first-window or terminal scoring cadence. The original API gate requires the full budget and every last-five primary observation to pass. The added study confirms the **first five consecutive primary successes**, then requires at least five later primary observations and **every later observation to pass**. Update 0 is excluded. A later recovery cannot replace the first failed hold.

| Required case | Distinct question | Primary law / original bounds | Default evidence |
| --- | --- | --- | --- |
| `image-develop-img_intensity2-source-transpose12` | Can the transpose12 generator recover both patch brightnesses, .35 and .85, with adequate sharpness and mass? | Public selected-policy draws, output noise off, DV12 latent perturbation retained; both modes, HQ ≥ .9, finite-template TV ≤ .1 and distribution TV ≤ .1; per-image RMSE cutoff .06 | 600 updates; 1,024 draws; primary checks every 25 |
| `api-vector-two-broad` | Can it recover a separated equal-mass two-Gaussian law and within-mode spread, rather than just two centers? | Output noise off with selected-policy latent law; normalized SW1 ≤ .18, mass TV ≤ .15, HQ ≥ .85, component covariance error ≤ .85, min eigen ratio ≥ .15, analytic projected KS ≤ .06, n ≥ 4,096 | 1,200 updates; 4,096 draws; primary checks every 50 |
| `api-grid100` | Recover all 100 equal-mass isotropic modes, their local widths and density fidelity | Noisy selected-policy primary, same-policy output-noise-off diagnostic; all retained native coverage and fidelity bounds | 7,000 updates; 24 × 20k primary checks plus update 0 and media observations; every last-five primary check passes |
| `api-rotated100` | Recover the same 100-mode law with fixed 25-degree geometry | Same primary and complete bounds as grid100; static orientation, not moving jumps | Same native schedule |
| `api-staggered100` | Recover the offset-row 100-mode geometry without axis-aligned support shortcuts | Same primary and complete bounds as grid100 | Same native schedule |
| `api-vector-unequal-mass` | Recover prescribed nonuniform occupancy, including the 2% rare component | Selected-policy output-noise-off law; original component/mass/shape bounds plus projected CDF checks | 1,200 updates; 4,096 draws; every 50 |
| `api-vector-anisotropic` | Recover covariance orientation and narrow axes without rescuing one axis by widening another | Selected-policy output-noise-off law; original covariance/component bounds plus projected CDF checks | 1,200 updates; 4,096 draws; every 50 |
| `image-develop-img_bars4-source-transpose12` | Recover all four horizontal/vertical bar positions, sharp edges and balanced occupancy | Same served image law; all four modes, HQ ≥ .9 and both TV limits .1; per-image RMSE cutoff .06 | 600 updates; 1,024 draws; every 25 |

The JSON carries exact per-case thresholds and definition hashes. Current native fidelity checks are not the historical 34-check protocol with a separate 100k holdout. Current output-noise-off native panels use the same selected policy; they are not separately forced-EMA samples.

## What actually passed, and where it failed

Both intensity runs first confirm at update **475**, from checks 375, 400, 425, 450, 475. Their five later checks 500 through 600 all pass. Both broad runs first confirm at **1,100**, from 900 through 1,100. Checks 1,150 and 1,200 pass, so the original terminal gate passes while the added five-later requirement is INCOMPLETE.

The named H2 continuation restores each exact 1,200-update checkpoint and changes only the external cap to 1,350. Existing retained metadata records complete checkpoint restoration, unchanged Recipe schedule, and bitwise public-sampler parity at 1,200. This review cites that metadata; it did not replay the models or sampler.

| Broad hold check | Atlas | E22 | Failed original bound |
| --- | --- | --- | --- |
| 1,150 | PASS | PASS | None |
| 1,200 | PASS | PASS | None |
| 1,250 | FAIL; projected KS .1058236785 | Same | `projection_ks <= .06` |
| 1,300 | FAIL; projected KS .0776330951 | Same | `projection_ks <= .06` |
| 1,350 | PASS; projected KS .0512585653 | Same | None |

The compound hold is **3 / 5 PASS, overall FAIL** in both families. Endpoint recovery at 1,350 does not turn it into sustained success. Detailed CDF/retained-state mechanism attribution belongs to the separate causal review; no causal diagnosis is inferred from only this chronology. H2 uses wrapper source `82c85cc3…` while executing the original hash-bound 8021 models, policy and observer. Its separate cost and named-question scope do not fill the six unknown quality cases.

## Why Atlas19 is still 19 / 19 without proving this baseline

The original diagnostic replay uses source `a0d6d89f…`, the exact Atlas config `lr=.00425 / P2 / D1`, and its own frozen RA15/RA11 host protocol. It completes 48,800 updates and 19 original questions. The source-bound intensity positive reproduces the historical full checkpoint byte-for-byte and all 24 metric observations after excluding only explicit wall time. This is useful evidence that the original positive can still be reproduced.

| Binding | Historical Atlas19 replay | C6 required public-policy test |
| --- | --- | --- |
| Intensity architecture | **Residual upsample width 16** | **Transpose width 12** |
| Initialization / training RNG | RA11 batch-feature-zero; seed 0; strict original shared CUDA image stream | Public deterministic orthogonal; seed 24002; separate policy/data streams |
| Image observation | Enumerate every one of 32 prior rows; latent perturbation disabled; noisy primary | Sample 1,024 public served draws; additive output noise off; DV12 latent perturbation retained |
| Image gates | Mode count and HQ; TV diagnostic | Mode count, HQ and explicit TV bounds |
| Broad training options | Config D1, P2, prior regularizer 0, betas `(0, .999)` | Host D1.5, P1.5, prior regularizer .05, betas `(0, .99)` |
| Broad observation | Original direct G/prior noisy observation; separate clean and EMA diagnostics | Public selected serving, output noise off; added analytic projected-CDF gate |
| Native protocol | 34 observations, final-five 20k plus independent 100k holdout; noisy primary | 24 post-update primary checks, same-policy output-noise-off diagnostic, no independent 100k or forced EMA panel |
| Convergence rule | Frozen original terminal/phase rules | Original terminal rule plus first-window and uninterrupted later hold |

The historical observer reads `trainer.G` and `trainer.prior`; those references can already contain policy-applied served parameters. They must not be described globally as unselected raw live weights. The verified intensity checkpoint records the fast serving path. Both observer contracts retain their actual source and serving choices.

All **30 `particlegan` files are byte-identical** between C6, the a0d6 replay and the maintained continuous checkout. The critical provider/scorer files also match C6. Maintained `api_run` changes atomic writing and source collection; `run_case` and `render_gif` have equal ASTs. `host_recipe_options`, `resolved_recipe` and `acquisition_hold` also have equal ASTs; the maintained family runner adds coordination and accounting. This static equality does not qualify a new source automatically, but it identifies no package/config regression explaining the cross-protocol comparison.

Recent failed contrasts were different settings: critic-rate source a956 used `.0053125 / P1.5 / D2.25`; generator-half source 488b used `.00265625 / P3 / D4.5`. Both complete first-image FAIL and leave seven cells unknown per family. They cannot replace the original C6 PASS or prove that it fails at acquisition.

Ordinary KA2/K3P **4 / 5 Tier 1** results use separate declared learned-MoG/clean-live hosts. Their 19 quality and two endurance requirements remain UNKNOWN, with qualified tier 0. They neither qualify nor invalidate this finite-particle public-policy baseline. Cold capacity support likewise supplies no learned, terminal-noise, whole-family or speed credit.

## Diagnostic admission checks

A prospective causal diagnostic should bind the original 8021 protected source, exact case hash and full effective Recipe, the host-owned D/prior/betas/resource settings, seed and initializer, selected serving/noise law, and original full observation cadence. A checkpoint-based question must bind all models, optimizers, controller, prior, output-noise/serving state, named and ambient RNG, data stream and cursor. The existing H2 restoration is evidence for that specific continuation, not a generic permission to replace its host or scorer.

Preserve the original intensity positive as a control and target the already observed broad hold failure. A changed source, host, noise law, forced EMA, horizon or cadence is a new explicit cohort. This read-only selection authorizes no unchanged scientific retry. Neither family has a fully qualified eight-case configuration or an eligible default or fair speed winner.

Read-only verification matched **16 planned Recipe hashes, four observed Recipes, 12 raw artifact hashes, both H2 result hashes, all 148 executed C6 source hashes and all 30 unchanged package files**. Complete checkpoint/sample files were hashed as bytes only; their contents were not decoded. Original reports, source, receipts, GIFs, logs and states remain unchanged.
