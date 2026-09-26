K3P stays the selected base. Three mobility proposals all kept the ring hold and the 300-update extension, and all three missed timely recovery. Nothing was promoted.

| Candidate | Hold | Extension | Deadline | Delay | Worst deadline HQ |
|---|---|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 | 300/300 | 28/81 | 1130 | published with that failure |
| **pm1 relative mobility** | **1200/1200, min HQ 0.938** | **300/300, min HQ 0.990** | **79/81** | **610** | **0.880 at step 2990 (7 modes)** |
| pm3 matched prior dwell | 1200/1200, min HQ 0.925 | 300/300, min HQ 0.947 | 79/81 | 810 | 0.900 at step 3190 |
| pm2 full dwell anchor | 1200/1200, min HQ 0.936 | 300/300, min HQ 0.974 | 78/81 | 1040 | 0.881 at step 3420 |

pm1 is the partial lead. Both rates stay at full strength until the joint generator-plus-prior gradient has fallen to 5% of its own peak, which on this ring is after 8 modes already exist (step 500). Each role then dwells at 10% of its base rate and reopens to 25% only while its own gradient is above five times its quiet floor. Critic mixing follows that network multiplier, not the learning-rate clock. At the shift the gradient jumped about 22 times the floor, both roles briefly reopened, and the gradient cooled within about 50 updates. The only deadline misses are steps 2990 and 3000, while both rates were already back at the dwell floor. Both optimizers ran all 3600 updates, and the ring went from 8 modes to 0 at the shift and later returned.

pm2 forced the EMA anchor fully on during that settled dwell. Recovery got worse: delay 1040 and 78/81. pm3 set the settled prior step equal to the network step. The roles did diverge, and recovery was still 79/81 with a longer delay and a weaker hold. Sensitive toys, the frozen control, horizon-prefix equality, repeated shifts, and the native seed matrix were not run. Inherited K3P noise still follows the driver's 1200-update horizon. That is a labeled schedule, not a horizon-free result.

The shift spike is already detected. The remaining miss is a quiet one-mode dropout that gradient size does not mark, so a longer reopen or a stronger dwell anchor is the wrong next change. A stationary reopen of this size has also already hurt native centers. Details, hashes, and replay commands are in `result.md`.
