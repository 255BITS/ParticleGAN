# Saved RowEvidence dimension feasibility

The current window50 caps effective sample size at99; d128 needs384. No saved RA4 or E22 row can mature under that law. Rank33 also cannot mature at finite touches; rank32 can.

| Run/step | Median/max n_eff | Mature for fixed k8 / k16 / k32 | Mean-spectrum 95% rank |
|---|---:|---:|---:|
| RA4/1250 | 11.94/87.74 | 311 / 76 / 0 | 3 |
| RA4/2000 | 13.91/97.19 | 408 / 102 / 1 | 3 |
| E22/1250 | 79.29/93.82 | 915 / 832 / 0 | 3 |
| E22/2000 | 94.48/98.25 | 1006 / 969 / 367 | 3 |

These maturity counts only use saved W/S. They do not estimate projected flags or a corrected quality result.

A fixed projection could reduce the sample requirement. However, scalar Qs stores total128-coordinate energy: projected variance and within-row gradient rank cannot be reconstructed. The spectrum of means across rows is descriptive and does not supply the within-row null dimension. Fitting a projection/rank to the same tested means would also introduce selection.

A later candidate would need an independently fixed generic projection, new projected statistics and explicit checkpoint identity. It would preserve full-dimensional optimizer updates and test force only in that subspace. The current trace-variance null assumes isotropic components and independent touches; projection does not establish those assumptions. No projection or null-law change is justified as a passing correction by these saved states.

Recommendation: keep RA5 evidence unchanged and await its prospective quality result. This receipt has no new gradients, optimizer updates, sampling, seeds or CUDA contexts. Frozen inputs were hashed before and after; global RNG is unchanged.
