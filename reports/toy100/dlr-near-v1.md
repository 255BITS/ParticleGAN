# Narrow shared discriminator-rate search

Four predeclared D-rate multipliers, .95, .975, 1.025, and 1.05, were applied
to the earlier 18/19 recipe. All other fields stayed fixed: κ=1, β₂=.999,
network floor .01, prior floor .05, and historical global output noise.
The source was frozen at `1c1a086`; seeds, budgets, architectures, data,
and frozen gates were unchanged.

| D/G learning-rate multiplier | Residual-student gate | Additional nine-host screen |
| ---: | --- | --- |
| .950 | FAIL | Not promoted |
| .975 | PASS | 6/9; trajectory, unequal mass, overlap fail |
| 1.025 | PASS | 6/9; trajectory, mode hold, overlap fail |
| 1.050 | FAIL | Not promoted |

The two residual repairs both regress other required behavior. Neither
qualifies for a fresh full-19 replay. No native or common-22 result is
inferred from these subsets.

The local directory `artifacts/toy100-accuracy/compatibility/dlr-near-v1/`
retains the frozen manifest, exact configs, script, source archives,
compressed episodes, logs, SHA-256 copy manifest, and independent strict
regrades after relocation. Every replay is valid failing numerical evidence.
