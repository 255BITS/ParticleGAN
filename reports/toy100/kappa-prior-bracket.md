# κ curriculum × prior-rate bracket

The fixed global κ schedule (1.25 through update 100, then linear to 1.0 at
update 200) was combined with three predeclared global prior LR multipliers.
The 2.05 row repaired the previous `mode_hold` five-check streak but passed
only **7/9** frozen screening hosts. The other two multipliers failed
`mode_hold` and did not advance. No row qualified for a fresh all-19 replay or
production policy support.

| Prior LR multiplier | `mode_hold` terminal suffix | Hosts attempted | Strict passes | Blocking result |
| ---: | ---: | ---: | ---: | :--- |
| 1.95 | 0/5 | 1/9 | 0/1 | `mode_hold`: 7/8 modes at the final check |
| 2.05 | 5/5 | 9/9 | 7/9 | `vector_unequal_mass`, `vector_overlap` |
| 2.10 | 4/5 | 1/9 | 0/1 | `mode_hold`: one terminal check short |

At 2.05, `residual_student`, `trajectory`, all three image hosts, and
`vector_unequal_width` passed. `vector_unequal_mass` missed the 2% rare
component entirely at the final observation (mass TV 0.1669), while
`vector_overlap` ended with normalized sliced Wasserstein 0.1095 and failed
its frozen verdict. These are observed regressions under the shared setting;
there was no seed, budget, threshold, or host-specific adjustment.

The archived predeclared manifest fixes the exact
three multipliers and the nine original hosts. The archived
scratch adapter `benchmarks/transfer_suite/kappa_prior_bracket_research.py`
used the frozen best-18 recipe for every other field and source commit
`9c4359f224808c9f19d39c33003b8fd3a303fa07`. Each executed episode
contains the complete applied κ-center trace and actual optimizer-rate
actions. Independent regrading checks every action against the declared
schedule and original host gate.

Raw evidence is stored locally at
`artifacts/toy100-accuracy/kappa-prior-bracket-v1-9c4359f`. All 18 files
matched the RAM originals by SHA-256, and relocated regrading reproduced
0/1, 7/9, and 0/1 with the declared screening skips. These are **scratch,
common-22-ineligible** results because the κ curriculum is not a production
recipe field.
