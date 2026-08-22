# Provenance of `b_cap`'s spectral stability

`b_cap` penalizes `relu(||grad_x D|| - kappa)^2`, which is identically zero -- and therefore contributes zero curvature -- wherever the critic is flatter than the cap. This file asks whether the reported spectral radii <= 1.002 come from the hinge being partially active at convergence (1), from the endpoint's own loss curvature with the penalty irrelevant there (2), or from b_cap not converging to a flat critic at all (3).

Measurements: 4096 reals + 4096 fakes for the endpoint gradient-norm field (fixed seed 20250821, identical draw for every run); `estimate_update_spectrum` with krylov_dim=24 on a fixed 1024-sample real batch and 1024 particle indices (fixed seed 31337, identical for every run and every condition), `penalty_step=7000`, lazy_k=1, plain-GD LRs {G: 3e-4*m, D: 3e-4*1.5*m, prior: 3e-4*10*m}.

## Verdict

**inherited from the state (explanation 2)**

At the b_cap endpoints the hinge is essentially slack: at most 0.22% of samples clear the cap (reals and fakes pooled over the b_cap groups). Masking the penalty out entirely at the *same* checkpoint state moves the dominant modulus from 1.00102 (actual) to 1.00066 (masked), a change of -0.00037 -- far below the 1.01 that would mark the penalty as load-bearing at this point.

The same masked measurement applied to `f_none_c0p0`'s own checkpoints returns 1.0677, well above 1. The measurement is therefore not what differs between the arms -- the *state* training reached is. Dropping a zero-centered `a_r1r2` penalty at coeff 0.02 onto the b_cap endpoints gives 1.00066, i.e. adding curvature there barely moves the number either: the endpoint is already inside the unit circle without any penalty support.

What the hinge *was* doing shows up in the trajectory rather than at the endpoint: q90 of ||grad_x D|| at the reals sits just above the cap for most of training and stops clearing it only at step 5580 of 7000 (65% of evals engaged). The critic then drops to a near-flat field (median ||grad_x D|| ~0.1) inside the LR anneal, and only *there* does the hinge go slack. So the penalty selected the trajectory and the basin; it is not what pins the final spectral radius.

## 1. Hinge activity at the endpoint

`frac>kappa` is the fraction of samples on which b_cap's penalty is even switched on (for the a_r1r2 / f_none groups the same columns are diagnostic context only -- those arms have no hinge).

| group | n | frac n_r>1 | frac n_f>1 | n_r q10/q50/q90 | n_f q10/q50/q90 |
|---|---|---|---|---|---|
| `b_cap_c1p0_lr2p0` | 5 | 0.00000 ± 0.00000 | 0.00049 ± 0.00027 | 0.035 / 0.095 / 0.195 | 0.034 / 0.089 / 0.176 |
| `b_cap_c1p0` | 5 | 0.00034 ± 0.00020 | 0.00386 ± 0.00161 | 0.040 / 0.102 / 0.223 | 0.034 / 0.088 / 0.176 |
| `a_r1r2_c1p0` | 5 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.006 / 0.014 / 0.029 | 0.008 / 0.022 / 0.051 |
| `a_r1r2_c0p1` | 5 | 0.00000 ± 0.00000 | 0.00210 ± 0.00091 | 0.011 / 0.036 / 0.094 | 0.008 / 0.027 / 0.078 |
| `f_none_c0p0` | 5 | 0.73838 ± 0.16760 | 0.43179 ± 0.20216 | 0.780 / 3.666 / 9.912 | 0.337 / 0.941 / 2.175 |

## 2. Activity trajectory

From the original runs' `metrics.jsonl` (mean over seeds). `q90 > 1` is the bracket for "more than 10% of samples are above the cap", i.e. the hinge is meaningfully engaged; `q90 < 1` means it is switched off for at least 90% of the batch.

| group | last step with q90 n_r > 1 | last step with q90 n_f > 1 | evals engaged (reals) |
|---|---|---|---|
| `b_cap_c1p0_lr2p0` | 5940 ± 102 | 6080 ± 40 | 0.746 ± 0.025 |
| `b_cap_c1p0` | 5220 ± 194 | 5520 ± 75 | 0.558 ± 0.103 |
| `a_r1r2_c1p0` | never | never | 0.000 ± 0.000 |
| `a_r1r2_c0p1` | never | 2740 ± 1271 | 0.000 ± 0.000 |
| `f_none_c0p0` | 7000 ± 0 | 7000 ± 0 | 0.986 ± 0.000 |

**`b_cap_c1p0_lr2p0`**

| step | med n_r | q90 n_r | med n_f | q90 n_f | q90_nr>1? |
|---|---|---|---|---|---|
| 1000 | 0.799 | 1.011 | 0.970 | 1.143 | yes |
| 2000 | 0.866 | 1.037 | 0.988 | 1.136 | yes |
| 3000 | 0.926 | 1.073 | 1.035 | 1.150 | yes |
| 4000 | 0.915 | 1.026 | 1.000 | 1.093 | yes |
| 5000 | 0.946 | 1.034 | 1.008 | 1.077 | yes |
| 6000 | 0.875 | 0.998 | 0.944 | 1.013 | no |
| 7000 | 0.094 | 0.192 | 0.089 | 0.176 | no |

**`b_cap_c1p0`**

| step | med n_r | q90 n_r | med n_f | q90 n_f | q90_nr>1? |
|---|---|---|---|---|---|
| 1000 | 0.724 | 1.020 | 0.827 | 1.083 | yes |
| 2000 | 0.733 | 1.026 | 0.901 | 1.152 | yes |
| 3000 | 0.842 | 1.025 | 0.952 | 1.089 | yes |
| 4000 | 0.873 | 1.025 | 0.954 | 1.066 | yes |
| 5000 | 0.885 | 1.021 | 0.950 | 1.053 | yes |
| 6000 | 0.128 | 0.278 | 0.113 | 0.239 | no |
| 7000 | 0.103 | 0.229 | 0.090 | 0.184 | no |

**`a_r1r2_c1p0`**

| step | med n_r | q90 n_r | med n_f | q90 n_f | q90_nr>1? |
|---|---|---|---|---|---|
| 1000 | 0.045 | 0.085 | 0.076 | 0.140 | no |
| 2000 | 0.042 | 0.086 | 0.081 | 0.178 | no |
| 3000 | 0.030 | 0.058 | 0.061 | 0.121 | no |
| 4000 | 0.024 | 0.044 | 0.051 | 0.096 | no |
| 5000 | 0.020 | 0.037 | 0.040 | 0.079 | no |
| 6000 | 0.016 | 0.030 | 0.028 | 0.057 | no |
| 7000 | 0.014 | 0.028 | 0.022 | 0.049 | no |

**`a_r1r2_c0p1`**

| step | med n_r | q90 n_r | med n_f | q90 n_f | q90_nr>1? |
|---|---|---|---|---|---|
| 1000 | 0.217 | 0.471 | 0.272 | 0.701 | no |
| 2000 | 0.169 | 0.385 | 0.279 | 0.686 | no |
| 3000 | 0.136 | 0.290 | 0.295 | 0.619 | no |
| 4000 | 0.116 | 0.243 | 0.248 | 0.495 | no |
| 5000 | 0.069 | 0.149 | 0.082 | 0.206 | no |
| 6000 | 0.046 | 0.110 | 0.043 | 0.109 | no |
| 7000 | 0.036 | 0.093 | 0.028 | 0.076 | no |

**`f_none_c0p0`**

| step | med n_r | q90 n_r | med n_f | q90 n_f | q90_nr>1? |
|---|---|---|---|---|---|
| 1000 | 22.068 | 41.706 | 51.599 | 90.977 | yes |
| 2000 | 19.169 | 39.694 | 45.096 | 85.525 | yes |
| 3000 | 18.343 | 39.703 | 45.031 | 92.056 | yes |
| 4000 | 16.874 | 35.218 | 42.659 | 93.895 | yes |
| 5000 | 7.943 | 22.006 | 13.599 | 37.457 | yes |
| 6000 | 3.745 | 9.546 | 1.180 | 2.621 | yes |
| 7000 | 3.574 | 9.637 | 0.944 | 2.178 | yes |

## 3. Masked vs. actual spectrum (same state, same batch)

(a) `actual` = the run's own regularizer; (b) `masked` = `f_none` at coeff 0, i.e. the penalty removed from the update map; (c) `r1r2_probe` = a zero-centered `a_r1r2` penalty at coeff 0.02 dropped onto the same state (b_cap groups only).

| group | condition | dominant modulus | power-iter modulus | max abs Im |
|---|---|---|---|---|
| `b_cap_c1p0_lr2p0` | actual | 1.00124 ± 0.00036 | 0.99996 ± 0.00003 | 0.00142 ± 0.00052 |
| `b_cap_c1p0_lr2p0` | masked | 1.00088 ± 0.00014 | 0.99998 ± 0.00007 | 0.00053 ± 0.00013 |
| `b_cap_c1p0_lr2p0` | r1r2_probe | 1.00089 ± 0.00010 | 0.99996 ± 0.00003 | 0.00055 ± 0.00011 |
| `b_cap_c1p0` | actual | 1.00081 ± 0.00036 | 0.99996 ± 0.00005 | 0.00053 ± 0.00009 |
| `b_cap_c1p0` | masked | 1.00043 ± 0.00002 | 0.99998 ± 0.00005 | 0.00037 ± 0.00005 |
| `b_cap_c1p0` | r1r2_probe | 1.00043 ± 0.00004 | 0.99999 ± 0.00006 | 0.00033 ± 0.00002 |
| `a_r1r2_c1p0` | actual | 1.00006 ± 0.00002 | 0.99998 ± 0.00001 | 0.00005 ± 0.00001 |
| `a_r1r2_c1p0` | masked | 1.00011 ± 0.00001 | 1.00001 ± 0.00001 | 0.00005 ± 0.00002 |
| `a_r1r2_c0p1` | actual | 1.00029 ± 0.00014 | 0.99991 ± 0.00012 | 0.00024 ± 0.00004 |
| `a_r1r2_c0p1` | masked | 1.00038 ± 0.00010 | 0.99997 ± 0.00008 | 0.00034 ± 0.00008 |
| `f_none_c0p0` | actual | 1.06769 ± 0.05229 | 0.97627 ± 0.02554 | 0.03121 ± 0.02780 |
| `f_none_c0p0` | masked | 1.06769 ± 0.05229 | 0.97627 ± 0.02554 | 0.03121 ± 0.02780 |

Per-seed dominant moduli:

| run | actual | masked | masked - actual | r1r2_probe |
|---|---|---|---|---|
| `b_cap_c1p0_lr2p0_s1` | 1.00081 | 1.00066 | -0.00015 | 1.00088 |
| `b_cap_c1p0_lr2p0_s2` | 1.00120 | 1.00106 | -0.00014 | 1.00100 |
| `b_cap_c1p0_lr2p0_s3` | 1.00120 | 1.00083 | -0.00037 | 1.00074 |
| `b_cap_c1p0_lr2p0_s4` | 1.00191 | 1.00100 | -0.00090 | 1.00101 |
| `b_cap_c1p0_lr2p0_s5` | 1.00107 | 1.00085 | -0.00022 | 1.00084 |
| `b_cap_c1p0_s1` | 1.00069 | 1.00044 | -0.00025 | 1.00044 |
| `b_cap_c1p0_s2` | 1.00050 | 1.00045 | -0.00005 | 1.00037 |
| `b_cap_c1p0_s3` | 1.00150 | 1.00042 | -0.00108 | 1.00048 |
| `b_cap_c1p0_s4` | 1.00080 | 1.00039 | -0.00040 | 1.00042 |
| `b_cap_c1p0_s5` | 1.00056 | 1.00046 | -0.00010 | 1.00043 |
| `a_r1r2_c1p0_s1` | 1.00009 | 1.00013 | +0.00003 | - |
| `a_r1r2_c1p0_s2` | 1.00004 | 1.00010 | +0.00006 | - |
| `a_r1r2_c1p0_s3` | 1.00005 | 1.00010 | +0.00006 | - |
| `a_r1r2_c1p0_s4` | 1.00005 | 1.00009 | +0.00004 | - |
| `a_r1r2_c1p0_s5` | 1.00007 | 1.00010 | +0.00003 | - |
| `a_r1r2_c0p1_s1` | 1.00034 | 1.00039 | +0.00006 | - |
| `a_r1r2_c0p1_s2` | 1.00022 | 1.00053 | +0.00031 | - |
| `a_r1r2_c0p1_s3` | 1.00015 | 1.00021 | +0.00006 | - |
| `a_r1r2_c0p1_s4` | 1.00021 | 1.00037 | +0.00016 | - |
| `a_r1r2_c0p1_s5` | 1.00055 | 1.00039 | -0.00016 | - |
| `f_none_c0p0_s1` | 1.05008 | 1.05008 | +0.00000 | - |
| `f_none_c0p0_s2` | 1.02564 | 1.02564 | +0.00000 | - |
| `f_none_c0p0_s3` | 1.15914 | 1.15914 | +0.00000 | - |
| `f_none_c0p0_s4` | 1.08872 | 1.08872 | +0.00000 | - |
| `f_none_c0p0_s5` | 1.01490 | 1.01490 | +0.00000 | - |

## Sanity guard

Mode coverage recomputed on the rebuilt EMA models (20k samples, fixed global seed) against the run's own `summary.json` final row. `mode_coverage` draws its 20k particle indices from the global RNG, so hq carries a sampling wobble of order 1e-3; anything larger would mean the rebuild is not the run's model.

| group | max |modes dev| | max |hq dev| |
|---|---|---|
| `b_cap_c1p0_lr2p0` | 0 | 0.00115 |
| `b_cap_c1p0` | 0 | 0.00470 |
| `a_r1r2_c1p0` | 0 | 0.00280 |
| `a_r1r2_c0p1` | 0 | 0.00400 |
| `f_none_c0p0` | 2 | 0.00435 |

