K3P stays the selected base. No candidate passed hold, extension, and the full shift verdict together, so nothing was promoted.

**p3_floor_reopen** is the best partial result. It keeps K3P's particle rule and direct response, restores K3P noise, and dwells at 10% of the initial rate only after the generator gradient collapses. The critic anchor then stays on even if the rate later moves.

| Candidate | Hold | Extension | Pre-shift | Deadline |
|---|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 | 300/300 | published | 28/81 |
| **p3** | **1200/1200, min HQ 0.992** | **300/300, min HQ 0.995** | **120/120** | **77/81 FAIL** |
| p2 | same hold and extension | same | 120/120 | 0/81 |
| p1 | incomplete FAIL (stuck at 7 modes) | — | 0/120 | 27/81 |

p3's four deadline misses are steps 2970, 3150, 3160, and 3170. The passing suffix starts at step 3180, so the delay is 780 updates. Both optimizers still took all 3600 steps. The 0.25 reopen never actually held: the return test compares the gradient with the acquisition peak, so the rate falls back to the dwell within one sample. Sampled post-shift multipliers peak at 0.163.

Toy gates, the frozen recovery control, the horizon-prefix audit, and the repeated-change stress were not run. Noise on p2 and p3 is still K3P's schedule on the driver's 1200-step noise horizon.

Details, hashes, and replay commands are in `result.md`. The ledger is `tests.jsonl`.
