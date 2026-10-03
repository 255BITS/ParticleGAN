# Latent-geometry isolation preflight: rejected

The [synthetic design](SPEC.md) was frozen before the [script](simulate.py)
ran. It reads only synthetic latent tables. [Full numeric results](results.json)
and the [tail-friendly log](results.log) are archived.

| Neighbors `k` | Calibration tables passing | Largest legitimate mean gain | 8D heavy-tail legitimate mean | 8D rare-group mean |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 4/8 | .1171 | .1171 | .0591 |
| 10 | 5/8 | .1198 | .1198 | .0168 |
| 20 | 4/8 | .1221 | .1221 | .5243 |
| 40 | 2/8 | .1207 | .1207 | .7494 |

The preset legitimate-row mean limit was `.07`; the 8D heavy-tail table
missed it for every `k`. Larger neighborhoods also treated the 16-row
legitimate rare group as isolated (`k=20` and `40`). All planted isolates
cleared the required `.40` mean gain and `.75` fraction above `.30`; the
problem is false mobility on legitimate rows. Scale and rotation checks all
passed, with maximum gain difference `1.8e−14`.

**Decision:** no `k` qualified, so the predeclared rule rejects this LOF
mechanism. Validation dimensions `{3,16}` and native training were not run.
This rules out the specified local-isolation gain under its fixed synthetic
criteria; it does not rule out every possible table-geometry controller.
E4's A2 violation remains unresolved.
