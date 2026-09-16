# Completed-model diagnostics, round13

All models evaluated on CPU; original and counterfactual histories recomputed on the same device.
Response windows contain32 points starting after the indicated number of generated writes.
Ideal radius/speed response is1. Nonzero sensitivity alone does not establish correct retention.

## Median normalized radius response

| Model | 0 | 1 | 8 | 32 | 128 | 512 | Last256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.24621 | 0.19072 | -0.05319 | 0.00785 | 0.00178 | -0.01885 | 0.02537 |
| match_shuffle25_5k | 0.29256 | 0.28229 | 0.19576 | 0.13146 | -0.03879 | 0.06709 | -0.01336 |
| mixed_mild | 0.46969 | 0.43296 | 0.17769 | 0.05861 | -0.00143 | -0.00001 | 0.00006 |
| explored_mild | 0.08261 | 0.03711 | -0.10367 | -0.01152 | 0.01208 | -0.00183 | 0.00000 |
| mixed_full | 0.35445 | 0.33966 | 0.17748 | -0.02396 | 0.01003 | 0.00000 | -0.00000 |
| explored_full | 0.27209 | 0.23363 | -0.00638 | -0.10665 | 0.00981 | 0.00018 | -0.00002 |
| mixed_mild_headonly | -0.02864 | -0.07205 | -0.34362 | -0.14924 | -0.02107 | 0.00000 | -0.00000 |

## Median normalized signed-speed response

| Model | 0 | 1 | 8 | 32 | 128 | 512 | Last256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.57958 | 0.55483 | 0.28013 | 0.08597 | 0.00226 | 0.01956 | 0.00129 |
| match_shuffle25_5k | 0.85914 | 0.85626 | 0.73545 | 0.19188 | 0.00066 | 0.00385 | -0.00207 |
| mixed_mild | 0.66026 | 0.64258 | 0.37249 | 0.08116 | 0.00150 | 0.00400 | 0.00000 |
| explored_mild | 0.83700 | 0.80366 | 0.53157 | 0.13558 | 0.01671 | -0.00153 | 0.00000 |
| mixed_full | 0.82810 | 0.76733 | 0.51420 | 0.09738 | 0.00119 | -0.00000 | -0.00000 |
| explored_full | 0.35143 | 0.32975 | 0.18556 | 0.00587 | 0.00445 | -0.00000 | -0.00000 |
| mixed_mild_headonly | 0.06744 | 0.04454 | 0.07714 | -0.00763 | 0.00038 | 0.00000 | -0.00000 |

## Correct mean direction in both original and flipped history

| Model | 0 | 1 | 8 | 32 | 128 | 512 | Last256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.53125 | 0.50000 | 0.36719 | 0.18750 | 0.14844 | 0.17188 | 0.09375 |
| match_shuffle25_5k | 0.77344 | 0.75781 | 0.69531 | 0.37500 | 0.13281 | 0.14844 | 0.14062 |
| mixed_mild | 0.60156 | 0.57031 | 0.40625 | 0.14844 | 0.15625 | 0.14062 | 0.12500 |
| explored_mild | 0.60156 | 0.58594 | 0.42188 | 0.21094 | 0.08594 | 0.09375 | 0.08594 |
| mixed_full | 0.62500 | 0.59375 | 0.40625 | 0.25000 | 0.10938 | 0.10938 | 0.10156 |
| explored_full | 0.50781 | 0.48438 | 0.30469 | 0.19531 | 0.14844 | 0.10156 | 0.06250 |
| mixed_mild_headonly | 0.53125 | 0.51562 | 0.30469 | 0.23438 | 0.02344 | 0.00781 | 0.01562 |

## Prefix32 point-head nearest-history ranking

| Model | Clean | After .25 write | After full write | Clean local MSE | Full-write local MSE |
|---|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.88281 | 0.90625 | 0.85938 | 0.00469 | 0.01378 |
| match_shuffle25_5k | 0.89062 | 0.90625 | 0.89844 | 0.00406 | 0.00998 |
| mixed_mild | 0.89062 | 0.90625 | 0.83594 | 0.00459 | 0.01423 |
| explored_mild | 0.88281 | 0.90625 | 0.83594 | 0.00492 | 0.01567 |
| mixed_full | 0.87500 | 0.91406 | 0.82812 | 0.00456 | 0.01438 |
| explored_full | 0.88281 | 0.88281 | 0.85156 | 0.00457 | 0.01616 |
| mixed_mild_headonly | 0.91406 | 0.91406 | 0.83594 | 0.00573 | 0.02004 |

MSE is evaluation only. Ranking uses noisy next points and fixed donor identities;
mismatched continuations are not guaranteed impossible under observation noise.
Neither classification accuracy nor timed-target gradient alignment establishes stable autonomous dynamics.
