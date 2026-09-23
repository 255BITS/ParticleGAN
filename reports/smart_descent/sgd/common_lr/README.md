# Plain SGD and gradient-feedback feasibility

The inner optimizer is raw SGD: no momentum, weight decay, clipping, or coordinate scaling. Independent G/prior and D learning rates are tuned on the existing ring4 and grid9 training distributions. Every episode uses seed 0 and 1,200 updates. Live weights determine every score; EMA is retained separately in the episode files.

The objective puts sustained full coverage/HQ first: 20×no sustained pass, plus final/last-five missing-mode and HQ deficits, a small normalized SW1 term, and observed convergence progress. A sustained pass requires every mode, HQ ≥90%, and at least five consecutive passing observations through the final checkpoint. Errors/incomplete runs score 1000.

| Best configuration | G LR | D LR | Objective ↓ | Ring4 modes / HQ | Grid9 modes / HQ | Sustained tasks | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGD constant | 0.25 | 0.005 | 26.2143 | 1/4 / 100.00% | 1/9 / 100.00% | 0/2 | 0 |
| SGD delayed cosine | 0.05 | 0.001 | 27.4959 | 0/4 / 0.00% | 1/9 / 100.00% | 0/2 | 0 |
| Adam delayed cosine | 0.0017 | 0.0017 | 20.9395 | 4/4 / 83.64% | 7/9 / 100.00% | 0/2 | 0 |
| SGD search winner (constant) | 0.25 | 0.005 | 26.2143 | 1/4 / 100.00% | 1/9 / 100.00% | 0/2 | 0 |

The sweep completed 32 schedule/rate settings (64 episodes); 36 episodes failed with recorded exceptions. Observed episode time totals 249.6 CPU seconds. No failed configuration is omitted from the ranking below.

| Configuration | G LR | D LR | Objective ↓ | Ring4 modes / HQ | Grid9 modes / HQ | Error episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sgd_constant_09 | 0.25 | 0.005 | 26.2143 | 1/4 / 100.00% | 1/9 / 100.00% | 0 |
| sgd_constant_04 | 0.05 | 0.001 | 27.4958 | 0/4 / 0.00% | 1/9 / 100.00% | 0 |
| sgd_cosine_04 | 0.05 | 0.001 | 27.4959 | 0/4 / 0.00% | 1/9 / 100.00% | 0 |
| sgd_cosine_08 | 0.25 | 0.001 | 27.4961 | 0/4 / 0.00% | 1/9 / 100.00% | 0 |
| sgd_constant_08 | 0.25 | 0.001 | 27.4961 | 0/4 / 0.00% | 1/9 / 100.00% | 0 |
| sgd_cosine_09 | 0.25 | 0.005 | 27.5007 | 0/4 / 0.00% | 1/9 / 100.00% | 0 |
| sgd_constant_00 | 0.01 | 0.001 | 28.4964 | 0/4 / 0.00% | 1/9 / 100.00% | 0 |
| sgd_cosine_00 | 0.01 | 0.001 | 30.2743 | 0/4 / 0.00% | 0/9 / 0.00% | 0 |
| sgd_constant_01 | 0.01 | 0.005 | 30.2787 | 0/4 / 0.00% | 0/9 / 0.00% | 0 |
| sgd_cosine_05 | 0.05 | 0.005 | 30.2810 | 0/4 / 0.00% | 0/9 / 0.00% | 0 |
| sgd_cosine_01 | 0.01 | 0.005 | 30.2811 | 0/4 / 0.00% | 0/9 / 0.00% | 0 |
| sgd_constant_05 | 0.05 | 0.005 | 30.3483 | 0/4 / 0.00% | 0/9 / 0.00% | 0 |
| sgd_cosine_10 | 0.25 | 0.025 | 514.4979 | 1/4 / 15.50% | 0/9 / 0.00% | 1 |
| sgd_constant_10 | 0.25 | 0.025 | 515.1479 | 0/4 / 0.00% | 0/9 / 0.00% | 1 |
| sgd_cosine_12 | 1.25 | 0.001 | 515.1497 | 0/4 / 0.00% | 0/9 / 0.00% | 1 |
| sgd_constant_12 | 1.25 | 0.001 | 515.1849 | 0/4 / 0.00% | 0/9 / 0.00% | 1 |
| sgd_constant_02 | 0.01 | 0.025 | 1000.0000 | 0/4 / 0.00% | 0/9 / 0.00% | 2 |
| sgd_constant_03 | 0.01 | 0.125 | 1000.0000 | —/— / — | —/— / — | 2 |
| sgd_constant_06 | 0.05 | 0.025 | 1000.0000 | 0/4 / 0.00% | 0/9 / 0.00% | 2 |
| sgd_constant_07 | 0.05 | 0.125 | 1000.0000 | —/— / — | —/— / — | 2 |
| sgd_constant_11 | 0.25 | 0.125 | 1000.0000 | —/— / — | —/— / — | 2 |
| sgd_constant_13 | 1.25 | 0.005 | 1000.0000 | 0/4 / 0.00% | 0/9 / 0.00% | 2 |
| sgd_constant_14 | 1.25 | 0.025 | 1000.0000 | 0/4 / 0.00% | 0/9 / 0.00% | 2 |
| sgd_constant_15 | 1.25 | 0.125 | 1000.0000 | —/— / — | 0/9 / 0.00% | 2 |
| sgd_cosine_02 | 0.01 | 0.025 | 1000.0000 | 0/4 / 0.00% | 0/9 / 0.00% | 2 |
| sgd_cosine_03 | 0.01 | 0.125 | 1000.0000 | —/— / — | —/— / — | 2 |
| sgd_cosine_06 | 0.05 | 0.025 | 1000.0000 | 0/4 / 0.00% | 0/9 / 0.00% | 2 |
| sgd_cosine_07 | 0.05 | 0.125 | 1000.0000 | —/— / — | —/— / — | 2 |
| sgd_cosine_11 | 0.25 | 0.125 | 1000.0000 | —/— / — | —/— / — | 2 |
| sgd_cosine_13 | 1.25 | 0.005 | 1000.0000 | 0/4 / 0.00% | 0/9 / 0.00% | 2 |
| sgd_cosine_14 | 1.25 | 0.025 | 1000.0000 | 0/4 / 0.00% | 0/9 / 0.00% | 2 |
| sgd_cosine_15 | 1.25 | 0.125 | 1000.0000 | —/— / — | 0/9 / 0.00% | 2 |

The scalar-feedback search evaluated 16 policies on both training tasks; 2 fitting episodes failed. Every policy and episode is retained in `fit.json.gz` and `episodes/`.

A read-only layer diagnostic exactly reproduces every metric of the selected constant SGD runs. At initialization, the generator's first weight matrix changes by about 0.0095% of its RMS, while its output bias changes by 25.8% and its output weights by 1.24%. A common scalar LR preserves these relative disparities for a given gradient. This motivates the separate per-tensor study; it is not a proof of what caused collapse. Full measurements are in `layer_diagnostics.json.gz`.

## Decision

The 16-policy scalar-feedback search selected the zero-weight constant controller. No adaptive proposal improved the best fixed SGD pair, which covers only one mode on each training distribution. None of the 64 rate-sweep episodes or 32 feedback-fitting episodes sustains full coverage/HQ.

Test a separate per-tensor relative-step controller to address the measured layer-scale disparity. This evidence limits the tested common-LR SGD setup; it does not establish that all SGD or learned raw-gradient methods fail.

## Artifacts and limits

- `sweep.json.gz` contains all rate pairs, full configuration/runtime/source fingerprints, selection scores, and summaries.
- `episodes/` contains every feature/action/metric trace, separate final EMA results, and complete error tracebacks.
- `fit.json.gz` and `sgd_policy.json.gz`, when present, record all controller proposals and the frozen policy.
- Tests analytically verify `parameter_next = parameter - LR * gradient`, empty optimizer state, and scalar-only controller effects.

This is a limited rate-grid experiment on one architecture and formulation. It does not establish that SGD can never work. Previously observed ring8 and full-suite results are validation data, not fresh held-out tests. No new distribution or architecture is used to tune this study.

Download [exact source](sources.tar.gz) and [archive hashes](archive_manifest.json). All raw JSON downloads use deterministic gzip; the manifest records original-byte SHA256s.
