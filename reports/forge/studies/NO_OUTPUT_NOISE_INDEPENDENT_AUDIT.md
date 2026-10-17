# Training output-noise removal: independent audit

**Integrity audit PASS; scientific result FAIL.** Frozen grading was reproduced exactly for the new run and reused control. No training or queue mutation occurred during this audit.

| Metric | K3P control | Output noise removed | Required |
| --- | ---: | ---: | --- |
| Full verdict | FAIL | FAIL | PASS |
| Coverage verdict | PASS | FAIL | PASS |
| Final modes | 100 | 100 | 100 |
| Final coverage passing suffix | 10 | 1 | >=5 |
| Terminal accuracy passes | 0 | 0 | 5/5 |
| 100k live covariance-trace bias | -0.265594 | -0.163005 | absolute <=.10 |
| 100k live radial KS | 0.120006 | 0.067130 | <=.04 |
| 100k live centre RMS / sigma | 0.089885 | 0.188341 | <=.20 |
| 100k live mass TV | 0.032000 | 0.031700 | <=.06 |
| Original paid seconds | 92.037031446 | 93.746687201 | measured |

Coverage FAIL means a passing suffix of only 1 check against the required five. Checks at 6,000 and 6,750 fail the component covariance upper limit: maximum eigenvalue ratios 1.877673 and 2.274458 exceed 1.7. Checks at 6,250, 6,500 and 7,000 pass, so final 100 modes and a passing final observation do not establish sustained coverage. All five final accuracy checks at 6,000–7,000 fail. Covariance-trace bias and radial KS improve relative to the control but remain outside the limits; centre RMS worsens. The independent 100k live holdout and EMA holdout fail, while all checked oracle references pass.

The only resolved recipe difference is `output_noise_std: .029 → 0`. Source, materialized task, prior, capabilities, seed zero, named stream bindings, optimizer settings and clean public sampling law match. A2 stays enabled. All 11 recorded initial parameter hashes and whole G/D state hashes match; actual step-zero live/EMA/target arrays and holdout target arrays match exactly.

Of 19 final named RNG states, 18 match. Only `[noise,generator,output,cuda:0]` differs, as preregistered when output-noise draws stop; the disabled run’s output stream equals its initial state exactly. Recorded evaluation audits report zero unintended RNG deviations. Both native 45-file manifests, checkpoint byte hashes, checkpoint content digests and complete 7,000-step optimizer states verify.

Registration was published at `0959af91` before execution; its bytes verify. New request `37ceb438a71a1ac65bdea9b5`, candidate revision `0fa6918ca4d0456515de09dbcfaeedffba1d4cecaab00dee5028bcf3b0b34603`. Source `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`; cohort `eca8051cdbaaf1e7b56ef6096a6d29bb403e581403b48d3b4f85796190eab08b`.

The new attempt costs 93.746687201 seconds on NVIDIA RTX A6000 GPU 0; the baseline is reused without another launch. Per-phase timings, allocator/RSS memory and complete receipt hashes are in the JSON. FLOPs remain unavailable.

**Recommendation:** stop this failed one-cell study. Do not launch its unregistered families or extend its failed parent. No full-reference positive, accepted screen or finished promotion candidate was established; subsequent work needs a separately justified bounded question.

[New receipt](../attempts/180af0fcfee447b697d5b5d05dc2877f/result.json) · [Reused control](../attempts/dca8c7aa0eca484c9270d07125f7cb0b/result.json) · [Machine-readable audit](NO_OUTPUT_NOISE_INDEPENDENT_AUDIT.json).

The new profile reduction was also independently reproduced exactly: **8/76 measured, 68 unknown**, two independent reference negatives (one paired), no positives, and **289.272195344 seconds** including the seven reused cells once. It is infeasible: every lineage has a measured smoke or reference failure, leaving no possible true accept while a positive and zero false rejections are both required.
