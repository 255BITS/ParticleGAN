# Narrow shared critic-cap search

The current shared candidate passes 18/19 older toys. A predeclared narrower
κ bracket retained all its other fields, including β₂=.999, network floor
.01, prior floor .05, and the historical global output-noise path. Source
was frozen at `1c1a086`; the new isolated selector was absent. Each host kept
seed 0, one CPU thread, its original budget, data, architecture, and gates.

| Shared κ | Residual-student gate | Further validation |
| ---: | --- | --- |
| .950 | FAIL | Not promoted |
| .975 | FAIL | Not promoted |
| 1.025 | FAIL | Not promoted |
| 1.050 | FAIL | Not promoted |
| 1.075 | PASS | 6/9 remaining bottlenecks pass |

The κ=1.075 variant regresses trajectory, mode hold, and vector overlap.
It passes unequal mass, unequal width, and all four image cases. It does
not qualify for the fresh full-19 replay required by the predeclared rule.
No native or combined-22 result is inferred from these subsets.

The exact manifest, configurations, executable orchestration, source
archives, compressed episodes, and logs are retained locally at
`artifacts/toy100-accuracy/compatibility/kappa-near-v2/`. All 71 copied files
matched the RAM originals by SHA-256, and independent strict regrades after
relocation reproduced every verdict without an integrity error.
