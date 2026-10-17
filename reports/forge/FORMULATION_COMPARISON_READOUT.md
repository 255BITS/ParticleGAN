# Formulation comparison readout — 2026-10-01

**All five full 7,000-update runs FAIL the unchanged native gates.** All independent integrity audits PASS. No formulation qualifies or advances to robustness/promotion. Five unique attempts cost **408.765 paid wall seconds** (6.81 minutes), against an 18,000-second reservation ceiling.

The [preregistered plan](studies/FORMULATION_COMPARISON_V1.md) was committed at `66f1c344` before enqueueing. [Machine-readable results](studies/formulation-comparison-v1-results.json), [independent audit](studies/FORMULATION_COMPARISON_INDEPENDENT_AUDIT.md), and [calibration reduction](calibration/formulation-comparison-v1.md) retain all grades, identities, observations and costs.

## Clean/live comparison leaderboard

These are independent 100k live holdouts. Smaller centre RMS, absolute covariance bias, radial KS and mass TV are better; precision must remain high. All five final 20k observations and sustained coverage must also pass. No scalar score selects a winner.

| Matched affine/MoG arm | Full gate | Sustained coverage | Terminal passes | Centre RMS/σ | Covariance bias | Radial KS | Mass TV | Precision | Paid seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K3P | FAIL | PASS | 0/5 | 0.08989 | -0.26559 | 0.12001 | 0.03200 | 0.99517 | 91.240 |
| Fixed BCap | FAIL | PASS | 0/5 | 0.05434 | -0.28512 | 0.13169 | 0.03200 | 0.99317 | 86.198 |
| Fixed R1/R2 | FAIL | FAIL | 0/5 | 0.33750 | -0.18604 | 0.05945 | 0.02802 | 0.97753 | 83.983 |
| v0.7 GAN v3, MoG adaptation | FAIL | FAIL | 0/5 | 0.22990 | -0.10583 | 0.03968 | 0.08079 | 0.97504 | 75.190 |

Accuracy limits: centre RMS ≤ .20σ, absolute covariance bias ≤ .10, radial KS ≤ .04, mass TV ≤ .06. Coverage separately requires all 100 modes, precision ≥ .97, mass TV ≤ .10, maximum mode mass ≤ .02, minimum high-quality mode mass ≥ .005, covariance eigenvalue ratios .4–1.7, radial median ratios .65–1.4, and finite samples across the declared terminal window. Coverage PASS alone is insufficient.

| Separate released-cloud context | Full gate | Coverage | Terminal passes | Centre RMS/σ | Covariance bias | Radial KS | Mass TV | Precision | Paid seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.7 GAN v3, named cloud | FAIL | FAIL | 0/5 | 0.57414 | -0.26608 | 0.10595 | 0.15151 | 0.76844 | 72.155 |

The first three arms isolate penalty choice: coefficient1, cap1, all other K3P settings held fixed. Fixed BCap improves centring (.08989→.05434) but worsens contraction and KS. R1/R2 improves covariance bias and KS but centre RMS reaches .33750σ and sustained coverage fails. None passes a terminal joint check.

The full released formulation on MoG has the smallest shape errors: radial KS .03968 passes the holdout limit, but covariance bias −.10583, centre RMS .22990σ and mass TV .08079 fail. Its five terminal checks and sustained coverage also fail. It changes several training mechanisms together, so this improvement identifies no isolated cause.

The separate cloud result preserves effective released public components/defaults on a newly declared named MLP host. Its z4, batch256, raw Gaussian particle cloud and network architecture differ from the matched z2/batch2048/MoG affine host. It supplies no MoG reference credit and cannot be ranked as a penalty or speed comparison.

## Paired sampling and EMA diagnostics

| Arm | Paired noisy/live full gate | Noise σ at holdout | Noisy precision | Noisy covariance bias | Noisy radial KS | Clean EMA holdout | Noisy EMA holdout |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| K3P | FAIL | 0.02900 | 0.93208 | +0.40062 | 0.15382 | FAIL | FAIL |
| Fixed BCap | FAIL | 0.02900 | 0.93227 | +0.39068 | 0.15017 | FAIL | FAIL |
| Fixed R1/R2 | FAIL | 0.02900 | 0.90450 | +0.42784 | 0.17507 | FAIL | FAIL |
| v0.7 GAN v3, MoG adaptation | FAIL | 0.00000 | 0.97504 | -0.10583 | 0.03968 | FAIL | FAIL |
| v0.7 GAN v3, named cloud | FAIL | 0.00000 | 0.76844 | -0.26608 | 0.10595 | FAIL | FAIL |

Adding the public scheduled output noise to the same clean draws over-widens all three penalty arms and drops precision below .97. It does not rescue their full grades. Both release recipes use zero output noise; their clean/noisy arrays and complete evaluator dictionaries are identical, and the extra noise streams consume no draws. EMA holdout outcomes are diagnostics, not complete native verdicts or alternative qualification. All four 100k holdout metric views and all terminal live/EMA checks remain in the machine-readable results.

## Release and repeatability explanation

The pinned release is **v0.7.0 at `180d18f400335fb295611d624b48a4e072ae3bae`**, whose default was GAN v3: z4, 20k particles, raw cloud, batch256, 7k steps, LR .00425, prior LR×2, Adam(0,.99), relativistic logistic loss, fixed BCap coefficient6/cap1.25/every1, VICReg .05, EMA .995 and a shared full-budget cosine schedule (start .6, floor .05). Input/output noise were absent, effectively zero. Its particle factory discarded `standardize=True`, so the actual cloud was unstandardized with sigma0. Release parity tests cover full model/prior/EMA/optimizer states and schedule boundaries. All K3P-only guard, anchor, damping and direct-gain hooks are disabled in the release ports.

The new K3P control exactly reproduces the previous clean-MoG control `dca8c7aa0eca484c9270d07125f7cb0b`: all **41 primary NPZ archives / 123 arrays**, complete G/D/prior/EMA model and both optimizer states, initial receipts and all **19 existing named final RNG states** match. The four new streams only serve paired evaluation. Process-global checkpoint RNG blobs differ between launches, so whole-checkpoint byte equality is not claimed. This verifies that restoring the penalty selector and adding paired diagnostics did not perturb K3P training.

Historical K3P promotion succeeded under its original cloud/noisy protocol, including the archived 22/22 promotion evidence. Later clean public sampling, learned MoG and named host initialization define different evidence. The present deterministic reproduction argues against spotty repeats for this exact control; it does not establish robustness across seeds or invalidate that historical promotion. The released cloud host here restores neither the historical draw order nor the old runner’s fresh generator-real callback: it uses named initialization and the public default call’s reused real batch.

## Integrity, costs and calibration

All five source/registration bindings, unchanged evaluator replay, **445 manifest files**, checkpoint byte/content certificates, zero-update reconstructed initialization and paired-noise replay passed independent audit. Each run completed all 7,000 generator, discriminator and prior updates; finite-state and intended mechanism checks passed, with zero unintended named-RNG deviations. All target oracles passed. K3P and fixed BCap have 10 passing final coverage observations; the other three have no passing suffix.

Public direct-particle gain has no applicable component on the affine MoG host. All three penalty arms record zero actual direct-gain applications and label its synthetic probe as software evidence. Fixed BCap and R1/R2 also have zero actual critic-guard applications; their synthetic guard probes establish software behavior only. K3P's critic guard applies 11 times. A2 applies 6,999 times on each of these three arms; K3P's anchor applies 5,715 times and fixed penalties have no anchor blend. Release ports have no A2, guard, anchor or direct-gain applications.

| Arm | Training update seconds | Evaluation seconds | Sampling seconds | CUDA allocated peak MiB | CUDA reserved peak MiB | Process RSS peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K3P | 71.296 | 9.736 | 0.075 | 119.6 | 184.0 | 2068.1 |
| Fixed BCap | 65.633 | 10.467 | 0.080 | 119.7 | 180.0 | 2068.0 |
| Fixed R1/R2 | 64.068 | 9.807 | 0.075 | 119.6 | 180.0 | 2067.5 |
| v0.7 GAN v3, MoG adaptation | 52.582 | 12.349 | 0.073 | 119.4 | 180.0 | 2021.2 |
| v0.7 GAN v3, named cloud | 52.859 | 9.056 | 0.101 | 167.7 | 216.0 | 2031.0 |

One serial worker used GPU0 (RTX A6000); GPU1 jobs were left alone. Paid wall time includes setup, sampling, evaluation and artifact I/O. Exclusive synchronized phase timings cover only the adapter phases declared in receipts; they do not sum to complete wall time. CUDA peaks cover the PyTorch allocator, excluding other processes/non-PyTorch allocations; RSS is process-lifetime. FLOPs are unavailable. Different host sizes/batches preclude a speed claim. No reused historical cost is charged to this study.

The profile retains 3 smoke and 16 independent reference tasks per lineage: **4/95 measured, 91 unknown**. Its separate cloud diagnostic has **1/5 measured, 4 unknown**. Four measured reference decisions are FAIL; the released-cloud lineage’s MoG reference remains UNKNOWN. All five smoke decisions are UNKNOWN, zero lineages are paired, false-accept/reject fractions are unknown, and adoption is BLOCKED. An incomplete cost vector is not zero cost. Neither paired noise nor cloud diagnostics add qualification/reference credit. There is no accepted calibration or eligible ordinary/robustness/promotion stage.

Frozen scientific source: `bf9ccf356d95cd0b0023449c741c2a3f99f260f38c779c07cce221e79fde23f3`.
Cohort: `48856fefffd85da9fe55d02f69656e632d042b2cbab09c84329d48eba5bb16a5`.
Profile: `0e6bea41fd24fa06d3349c29e415f5c5d0e9538ddb51841f25d8c47acb8a2493`.
Unchanged criteria: `6eac2531eed68e392c11a57328999ac5dce509f8fe51d23e7b9031b7a4e3f9d1`.
Registration payload: `01a8acf8004a6c1b6e10bd5c4a475e8067c35a74c5d3f7ab496a7daeb1926a8b`.

The final CPU development-environment suite passed **1,871 tests and 18 subtests, 7 skipped** (244.54s). The optional-matplotlib collection issue and eight intermediate compatibility failures remain recorded in [software validation](studies/formulation-comparison-v1-validation.json); fixes preceded registration. Frozen scientific runtime packages were unchanged. Forge validated 47 tasks / 6 views; history inventory covers 7,287/7,287 tracked sources with seven original gaps preserved. Final compiled memory contains 242 records, zero conflicts and no pending readouts. Post-readout registration verification resolves all five requests to their unchanged frozen identities.

## Recommendation and exact receipts

Stop these exact revisions and this selected registration. Before another GPU study, perform a bounded zero-training saved-state audit of critic radial and centering gradients against per-mode moments. Only a supported substantive hypothesis warrants a new frozen registration. Preserve A2 and sigma .025; no coefficient, cap, width, amplitude or seed sweep, best-checkpoint/EMA substitution, 14k continuation or automatic matrix expansion.

All five exact revisions are concluded; the four failed experimental revisions are abandoned and K3P remains a concluded reference. Every failure and unknown is preserved. The campaign has zero reserved seconds and no active selected work; its paid total is 408.765383113 seconds. Public defaults stay unchanged. The suggested saved-state analysis is a next step, not additional training authorized by this registration.

| Arm | Exact revision | Durable attempt |
| --- | --- | --- |
| K3P | `6385ef80463dca6a3c1c1f94dd3269c8e1db3702a8d236c1bd59d69787e06325` | [56177ab1d23243d8a6fe61413d8f870f](attempts/56177ab1d23243d8a6fe61413d8f870f/result.json) |
| Fixed BCap | `5db772f81cafe59b642b5707e600380fb49c3d42623b5c769e03dd677bbbfafd` | [26ebdea492d4408d9ee1b0cccc8bf61b](attempts/26ebdea492d4408d9ee1b0cccc8bf61b/result.json) |
| Fixed R1/R2 | `3bff4e5835a30fd018893f3b3cb1b612c5ba28d39e62247b61b728b38987b85e` | [11d3d8b814674fd0841217c7e4304c38](attempts/11d3d8b814674fd0841217c7e4304c38/result.json) |
| v0.7 GAN v3, MoG adaptation | `a9b2aa0ace96762b024a30c0979a0eedce8d07f9fae5cb5dc03868522132962c` | [0bce04d970264d4c95335030ed8724e7](attempts/0bce04d970264d4c95335030ed8724e7/result.json) |
| v0.7 GAN v3, named cloud | `3f2629ea8a1a6c9176d021ccdb99df48c2f7f67296946110288d59f3015ae14d` | [176ad9907a734b45844d09b2a69ff31d](attempts/176ad9907a734b45844d09b2a69ff31d/result.json) |

Central/per-attempt logs and bulk tensors remain available in the shared operational queue:

```sh
tail -F /home/martyn/dev/ParticleGAN/runs/forge/calibration-formulation-comparison-v1/progress.jsonl
tail -F /home/martyn/dev/ParticleGAN/runs/forge/events.jsonl
```
