# Bounded 16-row κ/prior Halton screen

No row passed the five mandatory frozen transfer hosts. The best row, h05,
passed `mode_hold`, `residual_student`, `trajectory`, and
`vector_unequal_mass`, then failed `vector_overlap`. The remaining four hosts
were skipped under the predeclared stage rule. No row qualified for a fresh
all-19 replay or production policy support.

The table gives the exact κ hold and linear-transition duration in absolute
updates, the one global prior LR multiplier, and the first strict failure.
Each row stopped at its first failure among the five mandatory hosts.

| Halton index | Hold | Duration | Prior LR × | Passed / tried | First failure |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 01 | 100 | 83 | 1.9960 | 0/1 | `mode_hold` |
| 02 | 75 | 117 | 2.0120 | 0/1 | `mode_hold` |
| 03 | 125 | 61 | 2.0280 | 3/4 | `vector_unequal_mass` |
| 04 | 62 | 94 | 2.0440 | 0/1 | `mode_hold` |
| 05 | 112 | 128 | 1.9832 | 4/5 | `vector_overlap` |
| 06 | 88 | 72 | 1.9992 | 0/1 | `mode_hold` |
| 07 | 138 | 106 | 2.0152 | 2/3 | `trajectory` |
| 08 | 56 | 139 | 2.0312 | 3/4 | `vector_unequal_mass` |
| 09 | 106 | 54 | 2.0472 | 0/1 | `mode_hold` |
| 10 | 81 | 87 | 1.9864 | 0/1 | `mode_hold` |
| 11 | 131 | 120 | 2.0024 | 0/1 | `mode_hold` |
| 12 | 69 | 65 | 2.0184 | 0/1 | `mode_hold` |
| 13 | 119 | 98 | 2.0344 | 0/1 | `mode_hold` |
| 14 | 94 | 131 | 2.0504 | 0/1 | `mode_hold` |
| 15 | 144 | 76 | 1.9896 | 3/4 | `vector_unequal_mass` |
| 16 | 53 | 109 | 2.0056 | 0/1 | `mode_hold` |

h05 used κ1.25 through step112 and reached κ1.0 at step240. Its rare-mass
host passed with final mass TV 0.0416 and 56 of 4,096 samples in the 2%
component. Its overlap host failed with normalized sliced Wasserstein 0.1229.
Eleven rows stopped at `mode_hold`, three at unequal mass, one at trajectory,
and one at overlap. These failures show why a late mode-hold pass alone did
not qualify a common recipe.

The manifest (archived `kappa-halton-manifest.json`) commits indices 1–16, exact
base-2/3/5 radical inverses, Python half-even integer rounding, every hold,
duration, multiplier, and the SHA-256 of all 16
row configs (archived `kappa-halton-configs`). The
scratch adapter (archived `benchmarks/transfer_suite/kappa_halton_research.py`)
was frozen at source commit `71e466de2841d3fdf91636e311ba8ce9922097ad`;
manifest SHA-256 is
`6f2014a05edf34ae256c50a826f0fe9e4eca20d49a723c7c1e2714ce67495f55`.
All other best-18 recipe fields, host resources, seeds, budgets, data, and
thresholds were fixed. Every executed episode retains complete applied κ
centers and actual optimizer-rate actions, verified against the declaration.

Raw evidence is stored locally at
`artifacts/toy100-accuracy/kappa-halton-v1-71e466d`. All 54 files matched
their RAM originals by SHA-256; relocated independent regrading reproduced
all 31 executed episodes and every skipped stage. This is **scratch,
common-22-ineligible** evidence because the κ curriculum is not a production
recipe field.
