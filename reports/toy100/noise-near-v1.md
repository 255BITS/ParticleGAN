# Four predeclared follow-ups to the closest transfer subset

The [frozen manifest](noise-near-v1-manifest.json) declared four single-field
changes to the best 7/8 transfer row before training (SHA-256
`f0e8fe4d95d9bf421a8f21bcec09ffa70536294d13dae1489938292ad0fdea7a`).
The original row was regraded from its exact saved episode. Each new row used
the same eight frozen hosts, seed 0, one CPU thread, host budgets, thresholds,
and source bytes. Every episode passed independent integrity regrading and
recorded the actual G-output and D-input noise mechanism.

The shared base is β₂=0.99, prior LR multiplier 2, output σ=0.0275 with 20%
warmup, input σ=0.5 ending at 12.5% of each host budget, LR anneal start 60%,
and LR floor 5%. Only the field named in the table changed.

| Row | Single change | Frozen live passes | Failed hosts |
| --- | --- | ---: | --- |
| Reused base | None | **7/8** | overlap (four of five required terminal checks) |
| `near_e012` | Input noise ends at 12% | 6/8 | mode hold, unequal width |
| `near_e013` | Input noise ends at 13% | 6/8 | mode hold, unequal width |
| `near_i045` | Input noise peak 0.45 | 6/8 | unequal width, overlap |
| `near_a055` | LR anneal starts at 55% | 5/8 | unequal width, overlap, four-bar image |

None reaches the declared 8/8 promotion threshold, so none was run across all
19 canonical transfer hosts. These are subset diagnostics, not a 19/19 or 22/22
result. The native 100-mode grid trial of the reused base also fails strict
accuracy and coverage despite sampling all 100 centers; this transfer screen
does not establish a common recipe.

The ranked numeric results and local master log are retained at
`artifacts/toy100-accuracy/compatibility/noise-near-v1/results.json` and
`artifacts/toy100-accuracy/compatibility/noise-near-v1/master.log`. Each new
row has a copied config, source archive, protocol, compressed episodes, and
task log under its declared artifact path. Raw `artifacts/` paths are local
workspace evidence, not GitHub links. The [shared search ledger](shared-recipe-search.md)
also includes all four episodes and their strict outcomes.
