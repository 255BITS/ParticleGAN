# Fixed critic-balance failure: retained-evidence diagnosis

Both Atlas and E22 completed the original 600-update intensity2 protocol and failed. All 25 saved checks, including initialization and all 24 post-update checks, fail the same four bounds. Neither family acquired the required five-pass window. The other seven cases per family remain **UNKNOWN**. Zero-update capacity passed for all 16 cells; that is no training qualification.

This readout uses only frozen receipts, tensor dictionaries and captured arrays. It constructs no model, draws no sample, updates no parameter and initializes no CUDA context. [Machine evidence](intensity-failure-analysis.json) binds both raw studies, receipts, GIFs, NPZs, checkpoints, recipes, runtime and all 151 Python files in their immutable execution snapshot.

## What fails

The target is a 4×4 center patch in an 8×8 image, at brightness 0.35 or 0.85 with equal mass. The other 48 pixels are zero. The public transpose-width12 host has 32 learned prior rows and an isolated 1,024-draw evaluation. Output noise is omitted from this primary gate; DV12 latent perturbation and selected public weights remain active. Every captured check selected fast weights, despite `serve_average=4` being configured.

At step 600, both runs have HQ **0**, valid modes **0**, nearest-assignment TV **0.5 > 0.1**, finite-template TV **1 > 0.1**, and rejected mass **1**. All 1,024 images are assigned nearest the dim template, but none lies within its RMSE≤0.06 quality neighborhood. This is brightness collapse, not successful recovery of one valid mode. Mean RMSE 0.1750018 is diagnostic; the 0.06 cutoff applies separately to each image.

| Retained step | Mean pixel | Center-patch mean | Patch mean sigmoid derivative | Fraction of pixels ≤1e−4 | Primary status |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0.507069 | 0.506442 | 0.249954 | 0 | FAIL |
| 25 | 0.00243666 | 3.18671e−7 | 3.18663e−7 | 0.652939 | FAIL |
| 50 | 0.000797967 | 4.49789e−9 | 4.49789e−9 | 0.809982 | FAIL |
| 100 | 0.000311183 | 1.04276e−10 | 1.04276e−10 | 0.885849 | FAIL |
| 300 | 0.000106678 | 1.11601e−12 | 1.11601e−12 | 0.933517 | FAIL |
| 600 | 0.0000962449 | 6.48773e−13 | 6.48773e−13 | 0.938400 | FAIL |

These are descriptive calculations on saved draws. The foreground sigmoid is locally almost insensitive to its output logit by the first retained post-update checkpoint. This establishes the numerical failure signature. It does not identify the sequence of critic, prior and generator forces that produced it in updates 1–25.

## What the state establishes

The last retained G-gradient norm is 0.000236256 over 5,173 parameters. Its signed output-bias gradient is −1.03056e−5, which points toward a brighter bias under minimization. G is not wholly gradient-free. The corresponding AMSGrad maximum second moment is 0.00540951, much larger than the squared current gradient. This is an observed recovery-scale signature, not proof that AMSGrad or the penalty caused the initial collapse.

The declared new rates were G 0.0053125, prior 0.00796875 and D 0.011953125 (`d_lr_mult=2.25`). At the final saved boundary, G/prior retain those rates but D is 0.000361887, about **3.03% of nominal**. The saved smoothed payoff error is 5.66708. `continuous.py:314–316` defines reciprocal-square payoff damping, applied to D after role-rate scaling in `policy.py:650–660`; `continuous.py:330–333` updates the smoothed positive G-minus-D loss difference. The saved rate was assigned before the final controller observation, so its fraction need not exactly equal the damping function evaluated at the final error. A higher declared D multiplier does not establish a higher effective D rate throughout training.

The final learned training output sigma is 0.0938399; it is omitted by this primary sampling gate. Birth/death counts show zero realized births and zero moves, no isolation action, no row reset, no controller reopen and no surprise fire. Atlas's settled guard has zero epoch rebases and no excursion. These saved counters exclude those recorded interventions as observed events in this run. They do not explain why ordinary learned dynamics collapsed.

Missing evidence for a causal account includes per-update G/D losses and effective role rates, intermediate optimizer/model states before step 25, and a matched current-source old-D-rate control. This report does not reconstruct those streams or substitute another cohort for them.

## Atlas and E22: exact observed equality and remaining distinction

The 25 retained metric/flag/view records match exactly: 275 scalar metrics. Their NPZ files are byte-identical; all 50 arrays and 1,641,600 values have equal dtype, shape and bytes. Their final G, D, prior, EMA-G and EMA-prior parameter dictionaries, optimizer states, controller state, named streams and caller data-generator state match exactly. Their original nine-frame GIFs are also byte-identical.

Their complete checkpoints are different: family recipes, Atlas backend/guard metadata and global CPU/CUDA RNG differ. Timing and family-labelled logs differ. This is neither an independent positive replication nor a proof of whole-policy equivalence.

Atlas selects reference-kNN/controller-reference serving on this 32-row host because its finite-resolution condition cannot support feature cells (38 minimum BH flags versus a one-row guard ceiling). Its role factors are all 1 here. E22 uses its reference DV12 path. Atlas's 20,000-row native quality hosts can activate the distinct feature-cell backend, and its settled guard can also distinguish the family elsewhere. Those native cases were **not reached**. Both family questions therefore retain a reason, while this completed small-population experiment supplies the same negative output evidence and supplies no native success.

## Relation to the earlier hold failure

[Frozen hold analysis](FROZEN_HOLD_ANALYSIS.md) remains a separate cohort. Its original 1,200-update broad runs passed the original gate and were study-INCOMPLETE. The separately named continuation to 1,350 failed persistence: projection KS 0.105824 at 1,250 and 0.0776331 at 1,300 exceed 0.06; 1,350 recovers to 0.0512586. Other required shape/mass bounds pass, with three of five later holds passing. This proves an attained state did not persist under that sampled gate; it proves neither a capacity defect nor a critic-rate cause.

The new intensity failure is an early local-output saturation signature; the old broad hold failure is a later projection-shape excursion. Different source/profile/law bindings preclude assigning their contrast to one coefficient. Both retain their original verdicts, recipes and raw identities.

## Accepted next declaration, not execution

One new complete shared tuple is `lr=0.00265625`, `prior_lr_mult=3.0`, `d_lr_mult=4.5` for each family. On the supported role-owned hosts, this halves nominal G and learned-output-noise rates while retaining nominal prior/D rates 0.00796875/0.011953125. It targets the observed early sigmoid excursion without changing the objective, architecture, initialization, family controls, prior, sampling law, seed, gates or horizons. Endogenous policy rates can still change; slower acquisition or repeated collapse remains possible. There is no monotonicity or winner prediction.

Root accepted the finite declaration. It remains **not executed**. Sixteen new candidate-bound zero-update capacity records, fresh public optimizer/controller/average ownership and current source/Recipe/sampler bindings are required. Original capacity verdicts and trained successes are not imported. The full eight-case denominator and two original smoke prerequisites remain. No failed unchanged profile or seed-only repeat is proposed.

The two new full attempts paid 54.3587717928458 seconds; the preserved original bootstrap error paid 4.757908704923466, for **59.116680497769266 seconds** already charged. The original 15,360-second campaign ceiling is retained. This report creates no reservation, extends no gate or horizon, and grants no default eligibility.
