None of the three candidates kept K3P's 8-mode ring while making particle, generator, and critic rates reversible. K3P remains ahead: 8 modes by update 500, hold 1200/1200, extension 300/300, recovery 28/81. These runs did not repeat that parent.

| Candidate | Shift (shift at 2400) | Official hold | Where it ended |
|---|---|---|---|
| **pnb2** | **FAIL** 0/120 pre-shift, 0/120 recovery | **FAIL** `NOT_CONVERGED`, 0-length 8-mode streak, hold 0/1200, extension not started | Hold finished at step 6300 with **7 modes / HQ 1.0**. Shift ended at 5 modes / HQ .660 |
| pnb3 | **FAIL** 0/120, min HQ 0 | not run | 6 modes / HQ .994 after a collapse to 0 modes at update 2000 |
| pnb1 | **FAIL** 0/120, min HQ 0 | not run | 7 modes / HQ .762 |

Critic gradient cosine is already negative while K3P is still acquiring the last modes. pnb1 used that cosine to leave the early penalty by update 130 and to swing the three rates apart. pnb2 kept the early penalty and one shared gain, but the same cosine had cut the gain to 0.59 by update 400, and the ring locked at 6 modes with HQ 1. pnb3 left the shared multiplier at 1 so the prior stayed at twice the network rate. Full rate plus a constant 0.25 EMA anchor still missed the 8th mode and collapsed once.

Every shift kept optimizing: 3600 updates on each role, and 3598 extra EMA-critic evaluations. Mode counts moved after the shift, so this is not a frozen generator pretending to adapt. The 22 toys, the repeated-change stress test, and a matched frozen control were not run. Input and output noise still follow the driver's 1200-step horizon, so none of these is a horizon-independent formulation. Details, replay commands, and paths are in `result.md`.
