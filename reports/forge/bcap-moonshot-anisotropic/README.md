# Anisotropic local real-neighborhood features: final readout

The candidate is rejected. The unchanged incumbent retains **6/6 original Tier 1 gates**; anisotropic local transport retains **4/6**, regressing two-pole and ring acquisition. Both arms pass only word hold among the ten selected Tier 2 questions. There are **zero complete Tier 2 repairs and zero unresolved final cells**. The registered anisotropic covariance prediction is falsified: **2.078374 > 0.85**. Keep the incumbent and stop this frozen candidate revision. This paired research diagnostic grants no ordinary qualification or default adoption.

One global delta enables `kinetic_transport_local_weight=1` and `kinetic_transport_local_geometry="anisotropic_knn_v1"`; the exact incumbent has zero overrides. Global W2, projection and finite critic guards remain disabled, with native full SVD retained. Existing training real batches alone define detached local covariance features. [Frozen derivation, numerical controls and competing explanations](preregistration.md), [spec](spec.json), and [READY registration](registration.json) preserve the pre-result decisions.

## Original numerical gates

Every status below comes from the original certified full-budget gate. The metric column is a diagnostic endpoint, not a replacement for temporal or multi-metric gates. Status links open the corresponding actual-training GIF.

| Original tier | Task | Incumbent | Candidate | Endpoint metric, incumbent → candidate |
| --- | --- | --- | --- | --- |
| 1 | gaussian1d_smoke | [PASS](media/baseline-gaussian1d_smoke.gif) | [PASS](media/candidate-gaussian1d_smoke.gif) | cdf_ks: 0.0718422 → 0.0387693 |
| 1 | two_pole | [PASS](media/baseline-two_pole.gif) | [FAIL](media/candidate-two_pole.gif) | grad_med: 0.952346 → 1.17456 |
| 1 | unused_token_hold | [PASS](media/baseline-unused_token_hold.gif) | [PASS](media/candidate-unused_token_hold.gif) | unused_hold: 0.990721 → 0.990721 |
| 1 | ae_gan_hold | [PASS](media/baseline-ae_gan_hold.gif) | [PASS](media/candidate-ae_gan_hold.gif) | recon_mse: 0.00285519 → 0.00302351; hold: 0.0119253 → 0.0187721 |
| 1 | ring16_acquisition | [PASS](media/baseline-ring16_acquisition.gif) | [FAIL](media/candidate-ring16_acquisition.gif) | mass_tv: 0.0651855 → 0.0385742 |
| 1 | five_word_joint_smoke | [PASS](media/baseline-five_word_joint_smoke.gif) | [PASS](media/candidate-five_word_joint_smoke.gif) | mass_tv: 0.0189453 → 0.0189453 |
| 2 | gaussian1d_stability | [FAIL](media/baseline-gaussian1d_stability.gif) | [FAIL](media/candidate-gaussian1d_stability.gif) | cdf_ks: 0.320623 → 0.0837034 |
| 2 | five_word_joint_hold | [PASS](media/baseline-five_word_joint_hold.gif) | [PASS](media/candidate-five_word_joint_hold.gif) | mass_tv: 0.0208984 → 0.0345703 |
| 2 | trajectory | [FAIL](media/baseline-trajectory.gif) | [FAIL](media/candidate-trajectory.gif) | identity_mse: 0.239862 → 0.244577 |
| 2 | residual_student | [FAIL](media/baseline-residual_student.gif) | [FAIL](media/candidate-residual_student.gif) | identity_mse: 0.061036 → 0.061318 |
| 2 | vector_unequal_mass | [FAIL](media/baseline-vector_unequal_mass.gif) | [FAIL](media/candidate-vector_unequal_mass.gif) | component_covariance_error: 3.69165 → 1.48883 |
| 2 | vector_unequal_width | [FAIL](media/baseline-vector_unequal_width.gif) | [FAIL](media/candidate-vector_unequal_width.gif) | component_covariance_error: 6.28756 → 1.63041 |
| 2 | vector_anisotropic | [FAIL](media/baseline-vector_anisotropic.gif) | [FAIL](media/candidate-vector_anisotropic.gif) | component_covariance_error: 0.449625 → 2.07837 |
| 2 | grid100 | [FAIL](media/baseline-grid100.gif) | [FAIL](media/candidate-grid100.gif) | precision: 0.24072 → 0.27425 |
| 2 | rotated100 | [FAIL](media/baseline-rotated100.gif) | [FAIL](media/candidate-rotated100.gif) | precision: 0.25552 → 0.23499 |
| 2 | staggered100 | [FAIL](media/baseline-staggered100.gif) | [FAIL](media/candidate-staggered100.gif) | precision: 0.30168 → 0.29955 |

Two-pole retains sufficient movement (0.958502 → 0.662042, bound ≥0.3), but critic slope rises from 0.952346 to 1.174558 against ≤1; passing temporal observations fall from 17 to zero. Ring ends inside its endpoint bounds, with mass TV 0.065186 → 0.038574 and covariance error 0.496567 → 0.664401, but its passing suffix falls from **26 to 3**, below the required five. These are measured regressions, not execution timeouts.

Gaussian stability improves substantially without repair: stationary passing checks rise from 2/72 to 41/72 and shifted hold checks from 0/24 to 12/24. Deadline reacquisition still fails; final KS is 0.083703 >0.05. Both arms have their own fully completed passing Gaussian and word producers. Word smoke completes 20,001 updates with the original 20,000 schedule and 900-second allowance; word hold adds 4,000 updates.

## Density and geometry

The three vector density tasks retain the full-component gates: SW1≤0.18, mass TV≤0.15, HQ≥0.85, full covariance error≤0.85, minimum eigen ratio≥0.15, and five sustained passing checks. Core covariance is a separate diagnostic. All six rows below have zero passing suffix.

| Task | Arm | Full covariance | Core covariance | Mass TV | SW1 | HQ | Min eigen ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vector_unequal_mass | baseline | 3.69165 | 0.530393 | 0.0706543 | 0.154638 | 0.95459 | 0.00907291 |
| vector_unequal_mass | candidate | 1.48883 | 0.715745 | 0.15 | 0.186288 | 0.945557 | 0 |
| vector_unequal_width | baseline | 6.28756 | 0.444178 | 0.291504 | 0.295865 | 0.97876 | 0.307472 |
| vector_unequal_width | candidate | 1.63041 | 0.36074 | 0.126221 | 0.11637 | 0.95166 | 0.660346 |
| vector_anisotropic | baseline | 0.449625 | 0.344346 | 0.195964 | 0.197952 | 0.97998 | 0.204013 |
| vector_anisotropic | candidate | 2.07837 | 0.227772 | 0.11141 | 0.12805 | 0.901123 | 0.460744 |

Anisotropic data exposes a shape/mass tradeoff. The control already passes the final covariance signature but fails mass and SW1. The candidate repairs those two endpoints and improves core covariance, yet worsens full covariance from 0.449625 to 2.078374. Its maximum component spill rises from 0.085890 to 0.152486. Improving the central bulk does not establish the complete density shape; the exact registered prediction and the wider primary hypothesis are both falsified.

Unequal width improves covariance, balance, SW1 and eigen spread, but full covariance remains 1.630406 >0.85. Unequal mass improves full covariance while losing the rare component: minimum mass ratio and minimum eigen ratio are zero, with mass TV 0.150000 and SW1 0.186288. Local geometry does not consistently preserve component mass. These outcomes match the preregistered competing explanation; no postnegative allocation term, kernel widening or additional configuration was introduced.

Native grid/rotated/staggered tasks each retain the 7,000-update budget and actual 100,000-sample independent clean holdout. Precision changes 0.24072→0.27425, 0.25552→0.23499 and 0.30168→0.29955, respectively; mass TV improves to 0.07343, 0.03909 and 0.06005. All original gates remain FAIL. On grid, center RMS error improves from 1.520315σ to 1.194195σ while covariance trace bias increases from 0.371385 to 0.558723. Better allocation and centering do not establish density precision.

## Saved mechanism counters

All active counters below are read from exact certified checkpoint bytes; the inactive control has no geometry state. Mirrored applied/consumer records in the receipt are the same state and are not summed. Geometry stores at most 64×8² covariance entries and uses at most 256 reference rows and sixteen other neighbors. DCT compression affects only auxiliary geometry, leaving architecture/data intact.

| Candidate task | Calls | Zero-covariance anchors / anchors | DCT calls | Ridge min–max | Analytic condition bound |
| --- | --- | --- | --- | --- | --- |
| ae_gan_hold | 250 | 0 / 16000 | 0 | 2.10121e-05–0.000304005 | 21 |
| five_word_joint_hold | 5667 | 362688 / 362688 | 5667 | 2.22494e-09–2.66178e-09 | 1 |
| five_word_joint_smoke | 20001 | 1280064 / 1280064 | 20001 | 2.12792e-09–2.7119e-09 | 1 |
| gaussian1d_smoke | 1000 | 0 / 64000 | 0 | 2.54882e-05–0.0155464 | 11 |
| gaussian1d_stability | 6000 | 0 / 384000 | 0 | 1.42892e-05–0.0231453 | 11 |
| grid100 | 7000 | 0 / 448000 | 0 | 0.00711147–0.166143 | 21 |
| residual_student | 400 | 0 / 4800 | 400 | 0.0205598–0.0251503 | 81 |
| ring16_acquisition | 1600 | 0 / 102400 | 0 | 0.000352537–0.155647 | 21 |
| rotated100 | 7000 | 0 / 448000 | 0 | 0.00700931–0.166925 | 21 |
| staggered100 | 7000 | 0 / 448000 | 0 | 0.00780193–0.147524 | 21 |
| trajectory | 400 | 0 / 4800 | 400 | 0.0205598–0.0251503 | 81 |
| two_pole | 80 | 0 / 960 | 0 | 0.0982843–0.100268 | 11 |
| unused_token_hold | 200 | 1600 / 1600 | 0 | 1.42109e-14–1.42109e-14 | 1 |
| vector_anisotropic | 1200 | 0 / 76800 | 0 | 0.000130997–0.00471392 | 21 |
| vector_unequal_mass | 1200 | 0 / 76800 | 0 | 0.000119705–0.224424 | 21 |
| vector_unequal_width | 1200 | 0 / 76800 | 0 | 5.32935e-05–0.0766917 | 21 |

Word smoke has zero covariance at all 1,280,064 anchors, uses DCT on all 20,001 calls, and has ridge 2.13e-9–2.71e-9. This yields tiny isotropic atom kernels, not evidence of learned anisotropic word geometry. Loss sum 1211.520237 and maximum one show the loss was not constantly inactive: near-atom activation can alter later gradients, while off-atom force can vanish. Unused-token loss is exactly one on all 200 calls with all 1,600 anchors degenerate; its unchanged pass offers no useful local-force validation.

Finite nondegenerate geometry on the vector and native hosts rules out an absent-mechanism explanation. The analytic condition bound is 1+trace(C)/ridge, at most 1+10d; it is not a measured eigenvalue condition number and cannot diagnose a tiny bandwidth. Euclidean neighbor selection and the trace ridge are not fully affine invariant. Low-dimensional rotation, translation and uniform-scale controls passed; fixed eight-column DCT can discard higher-dimensional distinctions. A bounded reference set can mix nearby or rare components into one neighborhood; cumulative counters do not identify which neighborhoods caused a failure.

The comparison changes both local-force activation and geometry. Without a third isotropic arm, it cannot isolate anisotropy from adding transport. Shared generator/critic responses, noisy finite-batch neighborhoods and feature force outside support remain competing explanations. [Exact checkpoint hashes and counters](geometry-counters.json) and [scalar analysis](analysis.json) retain these limits.

## Budget and explicit execution repairs

The original reservation was 22,920 seconds per arm, 45,840 paired, with unchanged paid ceilings of 24,000 per arm and 48,000 paired. Final charges are **6,255.570 incumbent + 4,959.769 candidate = 11,215.338 wall seconds**, across 38 attempts. Six specifically user-authorized infrastructure repairs retained the original source, recipe, seed, task allowances and gates. No completed scientific PASS/FAIL result was repeated.

| Arm / task | Original INCOMPLETE charge (s) | Linked repair charge (s) | Certified replacement |
| --- | --- | --- | --- |
| incumbent / vector_anisotropic | 2282.78 | 22.4929 | FAIL |
| candidate / trajectory | 1964.29 | 13.727 | FAIL |
| candidate / ae_gan_hold | 301.124 | 9.34601 | PASS |
| incumbent / five_word_joint_smoke | 900.786 | 739.86 | PASS |
| incumbent / grid100 | 698.026 | 212.257 | FAIL |
| candidate / residual_student | 701.84 | 13.8787 | FAIL |

The six predecessor charges total 6,848.839 seconds and remain paid; linked replacements total 1,011.561 seconds. Across all attempts the sum of original task allowances is 56,040 seconds, including 10,200 seconds of authorized retry allowances; this differs from the initial 45,840 reservation and the actual paid charge. Reboot orphan charges for residual student (701.840) and grid (698.026) include downtime. Host memory/I/O contention and interrupted evaluation also affected earlier timeouts. These charges are wall accounting, not compute or a valid speed comparison. Exact predecessor IDs, canonical result hashes, authorization and recovery receipt hashes remain in [analysis](analysis.json) and the complete [paid attempt history](phase3-results.json). Original incomplete outcomes are never recast as measured quality failures.

## Audit, software and publication

The saved audit verifies the frozen 1,207-file source manifest, identical initial model/prior states and named training bindings on all sixteen tasks, and exact complete non-evaluation stream/batch consumption on fifteen. Word hold follows its own eligible checkpoint: incumbent prefix 834, candidate prefix 1667. Both producers completed and passed, both holds restore exactly without history reset, and both add 4,000 updates; their final consumed streams are explicitly unverified as a matched pair because the offsets differ. This inherited own-checkpoint contract limits causal interpretation of hold endpoints.

Software before freeze passed 128 focused checks and 63 archived-develop compatibility checks, including 42 exact saved-state pairs, within the separate 300-second allowance. [Software receipt](software.json) preserves earlier fixture/metadata failures and the explicitly checked inactive-default/source-ancestry adapter. No extra software tests, training updates or sampling draws were performed for this publication.

Executed commit: `ad29d3b475ec724a7f4b3c668ad54b814a00736f`; scientific digest: `883c2cde79f5565b8449cc3571004e50f73bfc565f2bb86d6c032c234f7917a1`. Both arms were registered READY and executed from this same unchanged source. Reporting uses the read-only dependency annotation adapter at `f90591b85383052e803903088b775f4e9e95d8e1`, SHA256 `70302ffba5fbaf673375d3e4877363edf8ac80026b67cd9f872beb5ebe9c88e5`; it adds no training, grading or sampling. All 32 final cells are measured, so it reclassifies no cell here.

[Certified results and study decisions](phase3-results.json) · [Paired/own-checkpoint audit](phase3-audit.json) · [32 actual-training GIFs and hash receipts](media/index.json) · [Publication file hashes and reproduction provenance](publication.json) · [Read-only counter exporter](analyze_saved.py). GIFs illustrate actual saved training states; all conclusions use metrics. Bulk logs, JSONL, JUnit, raw tensors and checkpoints remain in `/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/anisotropic`; no raw execution logs are tracked.

Tail the retained execution archive with `tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/anisotropic/logs/driver.log`. Counter reproduction reads the saved canonical report and checkpoint archive with `analyze_saved.py --publication reports/forge/bcap-moonshot-anisotropic --archive /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/anisotropic --output /tmp/bcap-anisotropic-saved-analysis` under the repository Python environment and `PYTHONPATH=.`. The frozen registration, exact saved publisher invocation log, adapter identity, checkpoint receipts and file hashes are preserved; no unchanged experiment needs rerunning.

**Recommendation:** retain the incumbent, stop `anisotropic_knn_v1` at this revision, and retain its opt-in implementation as an experimental, inactive-by-default research result. Any new mechanism would require a fresh supported hypothesis and registration; this report authorizes no tuning, seeds, continuation, default adoption or automatic followup.
