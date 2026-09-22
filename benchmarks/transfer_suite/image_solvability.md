# Image GAN solvability

**One shared configuration sustains all four healthy image tasks at the original 600-step budget:** residual nearest-neighbor upsampling, width 16, with the original b_cap coefficient 3, kappa 1.25, LRs 0.0017, Adam (0,0.99), and 32 learned particles. This is an architecture repair; it does not establish that the original transpose network or a new LR controller solved the tasks. The best loss-only card on the original width 12 architecture sustains 2/4.

All gates remain fixed: 24 observations, at least five final passing observations, live HQ≥90%, and sufficient quality-qualified mass in every mode. No seed sweeps: every GAN uses seed 0. EMA is retained separately and never determines selection. All 60 healthy GAN episodes, four supervised controls, and two diagnostic episodes are preserved, including every failure.

Cells below show **sustained verdict · final modes/total · HQ**. A passing final snapshot alone does not qualify. Each row uses one shared card across all four tasks; no per-task oracle union is reported as a default.

## Original architecture and budget

| Shared card | Sustained /4 | Stripes | Bars | Blobs | Intensity |
| --- | ---: | --- | --- | --- | --- |
| r1r2_1 | 2/4 | PASS · 2/2 · 100.0% | FAIL · 4/4 · 84.4% | PASS · 4/4 · 93.8% | FAIL · 0/2 · 9.4% |
| vanilla_r1r2_01 | 1/4 | PASS · 2/2 · 100.0% | FAIL · 3/4 · 100.0% | FAIL · 4/4 · 87.5% | FAIL · 0/2 · 21.9% |
| r1r2_01 | 1/4 | PASS · 2/2 · 100.0% | FAIL · 3/4 · 93.8% | FAIL · 2/4 · 87.5% | FAIL · 0/2 · 3.1% |
| baseline | 1/4 | FAIL · 1/2 · 100.0% | FAIL · 2/4 · 100.0% | PASS · 4/4 · 96.9% | FAIL · 0/2 · 0.0% |
| fixed_prior | 1/4 | FAIL · 1/2 · 100.0% | FAIL · 2/4 · 65.6% | PASS · 4/4 · 96.9% | FAIL · 0/2 · 0.0% |
| prior_lr10 | 0/4 | FAIL · 1/2 · 93.8% | FAIL · 1/4 · 90.6% | FAIL · 3/4 · 100.0% | FAIL · 0/2 · 0.0% |
| cap_coeff03 | 0/4 | FAIL · 1/2 · 100.0% | FAIL · 2/4 · 78.1% | FAIL · 1/4 · 87.5% | FAIL · 0/2 · 0.0% |
| lsgan_vanilla | 0/4 | FAIL · 1/2 · 100.0% | FAIL · 1/4 · 100.0% | FAIL · 0/4 · 0.0% | FAIL · 0/2 · 0.0% |

R1+R2 coefficient 1 fixes stripe coverage and preserves blobs, but bars HQ 84.375% and intensity quality still fail. Frozen particles, 10× prior LR, weaker cap and least-squares variants do not produce a shared solution. All exact cards are in the machine declarations; particle-LR changes use a separate prior parameter group, preserving the generator rate.

## Architecture controls at 600 steps

| Shared card | Sustained /4 | Stripes | Bars | Blobs | Intensity |
| --- | ---: | --- | --- | --- | --- |
| residual16 | 4/4 | PASS · 2/2 · 100.0% | PASS · 4/4 · 93.8% | PASS · 4/4 · 100.0% | PASS · 2/2 · 100.0% |
| residual16_r1r2_01 | 4/4 | PASS · 2/2 · 100.0% | PASS · 4/4 · 93.8% | PASS · 4/4 · 100.0% | PASS · 2/2 · 93.8% |
| residual16_cap10 | 3/4 | PASS · 2/2 · 100.0% | FAIL · 4/4 · 81.2% | PASS · 4/4 · 96.9% | PASS · 2/2 · 100.0% |
| residual12 | 2/4 | PASS · 2/2 · 100.0% | FAIL · 3/4 · 93.8% | FAIL · 3/4 · 71.9% | PASS · 2/2 · 100.0% |
| transpose16 | 0/4 | FAIL · 1/2 · 93.8% | FAIL · 2/4 · 81.2% | FAIL · 3/4 · 90.6% | FAIL · 0/2 · 0.0% |
| transpose24 | 0/4 | FAIL · 1/2 · 100.0% | FAIL · 2/4 · 90.6% | FAIL · 2/4 · 75.0% | FAIL · 0/2 · 0.0% |

Residual16 with cap3 confirms stripes/bars/blobs/intensity at steps 225/550/550/575. Final HQ is 100%/93.75%/100%/100%. R1+R2 is unnecessary: the residual16 R1(.1) card also passes 4/4, but intensity HQ is lower. The cross-domain cap10 card falls to 3/4 because bars HQ falls to 81.25%. It is not a uniformly better setting.

Width controls separate architecture from capacity. Residual12 uses the original D width and fewer G parameters, yet fixes stripes and intensity. Transpose16 has the same D as residual16 and more G parameters, but passes 0/4. Residual architecture plus adequate width is the supported recipe here; greater width alone is not enough.

| Architecture | G parameters | D parameters |
| --- | ---: | ---: |
| transpose12 | 5173 | 2833 |
| residual12 | 3157 | 2833 |
| transpose16 | 8945 | 4929 |
| residual16 | 5361 | 4929 |
| transpose24 | 19561 | 10849 |

## Extra-budget control

| Shared card | Sustained /4 | Stripes | Bars | Blobs | Intensity |
| --- | ---: | --- | --- | --- | --- |
| baseline_1200 | 0/4 | FAIL · 1/2 · 93.8% | FAIL · 3/4 · 100.0% | FAIL · 4/4 · 100.0% | FAIL · 0/2 · 3.1% |

Doubling the original transpose baseline to 1200 steps does not repair the shared failure. Blobs ends with 4/4 and HQ 100%, but lacks the required final passing suffix and correctly remains FAIL.

## Expressivity and diagnostic controls

The supervised witnesses use balanced latent-index/template labels and MSE on the exact original G/prior architecture for 600 steps. They are **not GAN successes**, do not enter selection, and only establish representability when they converge.

| Supervised control | Result | Confirmed step |
| --- | --- | ---: |
| img_stripes2 | PASS · 2/2 · 100.0% | 225 |
| img_bars4 | PASS · 4/4 · 100.0% | 275 |
| img_blobs4 | FAIL · 3/4 · 53.1% | — |
| img_intensity2 | PASS · 2/2 · 100.0% | 250 |

All three originally failing healthy targets are representable with the original architecture and budget under direct supervision. The supervised blobs attempt fails and is retained; ordinary GAN training already demonstrates blobs solvability.

| Diagnostic (zero selection weight) | Shared card | Result |
| --- | --- | --- |
| img_bars8 | residual16 | FAIL · 5/8 · 65.6% |
| img_bars8 | residual16_cap10 | FAIL · 5/8 · 71.9% |

The denser eight-mode diagnostic remains unsolved and does not veto the healthy-task result. The information-poor and spatially uniform architecture diagnostics keep their declared nonblocking importance; their constraints were not removed to manufacture a pass.

## Provenance and limits

Total recorded episode wall time: 511.6s across 66 complete episodes; no numerical errors. These are single CPU observations under shared system load, not a speed estimate. Three analytical and parity tests passed: scoped loss/prior/Adam restoration, prior-only LR scaling and frozen-prior behavior, forbidden gate changes, and exact baseline numerical parity.

The four baseline reruns exactly reproduce every one of 24 live/EMA checkpoints and every loss checkpoint in the parent transfer study. Residual16 bars likewise exactly reproduces the formerly reserved bars setup; it is the same numerical case, **not an independent transfer success**. All former reserved cases are now seen development data. No fresh holdout, natural-image dataset or production default was evaluated or changed.

Run the declared initial search:

```bash
python -u -m benchmarks.transfer_suite.image_solvability --controls \
  --output /tmp/pr36-image-solvability-stage1
```

Additional stages use `--cards <JSON>` with exact declared changes. The handoff bundle `/tmp/pr36-image-solvability-artifacts/` contains all stages, logs, deterministic JSON.gz archives, original-byte SHA256s, exact source bundles and the calibration scripts. The parent PR report retains that bundle. `source.tar.gz` includes the exact scoped shim used to expose loss/prior options without changing the original task runner.
