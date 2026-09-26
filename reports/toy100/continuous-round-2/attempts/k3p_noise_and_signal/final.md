K3P stays the selected base. No candidate passed hold, the 300-update extension, and the shift deadline together, so nothing was promoted.

| Candidate | Hold | Extension | Recovery deadline |
|---|---|---|---|
| K3P parent (published, not rerun) | 1200/1200 | 300/300 | FAIL 28/81 |
| **ns3 fixed schedule + shock** | **1200/1200 PASS**, converged step 1881, min HQ 0.964 | **FAIL 274/300**, min modes 7 | **FAIL 0/81**, pre-shift 92/120 |
| ns2 paced floor | FAIL, streak 15, 8 modes only at step 6300 | NOT_RUN | FAIL 13/81, delay 1110 |
| ns1 cosine heat | FAIL, streak 0, max 7 modes | NOT_RUN | FAIL 0/81 |

ns1 never left full rate. The critic-gradient cosine kept flipping, so the heat stayed at 0.97, the mix stayed at 1, and the EMA anchor never started. ns2 did reach a low rate, but only after the ring was still at 5 modes, then a gradient spike reopened the noise and collapsed it.

ns3 is the useful negative. Copying the parent network cosine and the ring noise lengths as fixed constants (1600-step horizon, anneal 0.6, floor 0.01, input span 120, output span 240), without reading the training budget, restored the 1200-update hold and turned the anchor fully on. The 1.5× gradient-RMS shock did not see the target change: the reopen weight ended at 0.004, the critic rate sat on the floor (minimum 4.26e-5), and the deadline was 0/81. A small reopen around step 1900, which lifted the critic rate to 1.3e-4, is the likely source of the 26 extension misses.

Toy gates, the frozen recovery control, the two-horizon prefix, and the repeated-change stress were not run. Details, hashes, and replay commands are in `result.md`.
