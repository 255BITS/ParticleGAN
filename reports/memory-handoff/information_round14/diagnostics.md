# Completed-model diagnostics, round14

All models evaluated on CPU; original and counterfactual histories recomputed on the same device.
Response windows contain32 points starting after the indicated number of generated writes.
Ideal radius/speed response is1. Nonzero sensitivity alone does not establish correct retention.

## Median normalized radius response

| Model | 0 | 1 | 8 | 32 | 128 | 512 | Last256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.24621 | 0.19072 | -0.05319 | 0.00785 | 0.00178 | -0.01885 | 0.02537 |
| match_shuffle25_5k | 0.29256 | 0.28229 | 0.19576 | 0.13146 | -0.03879 | 0.06709 | -0.01336 |
| future_clean10 | -0.16198 | -0.20828 | -0.36838 | 0.04665 | 0.00895 | 0.00210 | 0.00001 |
| future_mixed10 | 0.13966 | 0.11600 | -0.01786 | 0.00199 | -0.00426 | -0.00001 | -0.00000 |
| future_full10 | 0.30894 | 0.26762 | 0.02182 | 0.01274 | 0.00858 | -0.00041 | 0.00000 |
| future_mixed25 | 0.03836 | -0.02176 | -0.13405 | 0.09488 | 0.00715 | -0.00013 | -0.00190 |
| future_mixed10_detachwrite | 0.14798 | 0.10850 | -0.16859 | -0.11871 | 0.00190 | -0.00000 | -0.00000 |

## Median normalized signed-speed response

| Model | 0 | 1 | 8 | 32 | 128 | 512 | Last256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.57958 | 0.55483 | 0.28013 | 0.08597 | 0.00226 | 0.01956 | 0.00129 |
| match_shuffle25_5k | 0.85914 | 0.85626 | 0.73545 | 0.19188 | 0.00066 | 0.00385 | -0.00207 |
| future_clean10 | 0.23921 | 0.21645 | 0.13784 | 0.01034 | 0.00010 | -0.00000 | 0.00000 |
| future_mixed10 | 0.25427 | 0.22663 | 0.18415 | 0.02438 | 0.00017 | -0.00000 | 0.00000 |
| future_full10 | 0.74861 | 0.72049 | 0.39666 | 0.12163 | -0.00090 | 0.00000 | 0.00000 |
| future_mixed25 | 0.86262 | 0.84038 | 0.54588 | 0.17130 | -0.01167 | 0.00033 | 0.00007 |
| future_mixed10_detachwrite | 0.03217 | 0.00655 | 0.00667 | 0.00437 | 0.00058 | 0.00000 | 0.00000 |

## Correct mean direction in both original and flipped history

| Model | 0 | 1 | 8 | 32 | 128 | 512 | Last256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.53125 | 0.50000 | 0.36719 | 0.18750 | 0.14844 | 0.17188 | 0.09375 |
| match_shuffle25_5k | 0.77344 | 0.75781 | 0.69531 | 0.37500 | 0.13281 | 0.14844 | 0.14062 |
| future_clean10 | 0.41406 | 0.40625 | 0.23438 | 0.10156 | 0.10938 | 0.12500 | 0.10156 |
| future_mixed10 | 0.48438 | 0.44531 | 0.29688 | 0.32812 | 0.11719 | 0.13281 | 0.01562 |
| future_full10 | 0.54688 | 0.49219 | 0.28906 | 0.26562 | 0.16406 | 0.09375 | 0.02344 |
| future_mixed25 | 0.55469 | 0.53906 | 0.40625 | 0.32812 | 0.14844 | 0.07031 | 0.02344 |
| future_mixed10_detachwrite | 0.42188 | 0.39844 | 0.25781 | 0.18750 | 0.06250 | 0.04688 | 0.00781 |

## Prefix32 point-head nearest-history ranking

| Model | Clean | After .25 write | After full write | Clean local MSE | Full-write local MSE |
|---|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.88281 | 0.90625 | 0.85938 | 0.00469 | 0.01378 |
| match_shuffle25_5k | 0.89062 | 0.90625 | 0.89844 | 0.00406 | 0.00998 |
| future_clean10 | 0.88281 | 0.87500 | 0.84375 | 0.00462 | 0.01684 |
| future_mixed10 | 0.89062 | 0.89062 | 0.85938 | 0.00456 | 0.01551 |
| future_full10 | 0.88281 | 0.89844 | 0.83594 | 0.00430 | 0.01418 |
| future_mixed25 | 0.82812 | 0.82031 | 0.81250 | 0.00408 | 0.01279 |
| future_mixed10_detachwrite | 0.87500 | 0.88281 | 0.83594 | 0.00470 | 0.01784 |

## Prefix32 nearest-history future ranking

| Model | Horizon conditioned | h0 clean | h4 clean | h12 clean | h4 full write | h12 full write |
|---|---|---:|---:|---:|---:|---:|
| match_shuffle25 | False | 0.88281 | 0.57812 | 0.48438 | 0.58594 | 0.49219 |
| match_shuffle25_5k | False | 0.89062 | 0.60938 | 0.50781 | 0.60938 | 0.50000 |
| future_clean10 | True | 0.88281 | 0.72656 | 0.53906 | 0.71875 | 0.54688 |
| future_mixed10 | True | 0.89062 | 0.72656 | 0.53906 | 0.71094 | 0.50781 |
| future_full10 | True | 0.88281 | 0.72656 | 0.54688 | 0.69531 | 0.52344 |
| future_mixed25 | True | 0.82812 | 0.80469 | 0.59375 | 0.81250 | 0.58594 |
| future_mixed10_detachwrite | True | 0.87500 | 0.71875 | 0.54688 | 0.71094 | 0.53906 |

Old references have no horizon conditioning; their h4/h12 scores use the immediate head.
All models supply their own full-write corruption. Improved future classification is not
a guarantee of useful immediate gradients or repeated memory retention.

MSE is evaluation only. Ranking uses noisy next points and fixed donor identities;
mismatched continuations are not guaranteed impossible under observation noise.
Neither classification accuracy nor timed-target gradient alignment establishes stable autonomous dynamics.
