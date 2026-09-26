**rcm3 keeps K3P's ring hold and 300-update extension, reopens the critic mix when the target shifts, and still fails recovery.** Post-shift checks are 0/120, including 0/81 from the deadline step onward. The ring is moving, not frozen: 0 modes at the shift, 7 modes and HQ 0.821 at step 3600, with both optimizers still taking updates at the floor rates.

The learning-rate-ratio handover was the only change. The anchor decay, particle prior, and sparse-latent rule are the copied K3P files. The parent bundle was not edited and was not rerun.

| Candidate | Hold | Extension | Recovery |
|---|---:|---:|---|
| **rcm3 applied-step ratio** | **1200/1200**, min HQ 0.990 | **300/300**, min HQ 0.999 | **FAIL**, 0/81 deadline-onward |
| rcm1 gradient agreement | FAIL, stuck at 7 modes | not run | not run |
| rcm2 anchor-gradient gap | FAIL, max 6 modes | not run | not run |
| K3P parent (published) | 1200/1200, min HQ 0.907 | 300/300, min HQ 0.988 | FAIL 28/81 |

rcm3 sets the mixing weight from the fast/slow ratio of the critic's applied step size. It does not read the learning rate, the step index, or the horizon. It closed at critic step 497 and reopened at step 2403, on the shift, when the step size jumped about 3×. Returning to the early penalty at the floor rate (4.25e-5) did not restore 8 modes by step 3600.

rcm1 closed by step 300 because a gradient spike dominated its peak, and it never reached 8 modes. rcm2's anchor-gradient gap stayed high, so the mix never left the early penalty and it topped out at 6 modes.

A 40-update prefix matched with network horizon caps 1600 and 3200 (same randomness digest, same step sizes, mixing weight still 1). That prefix ends before either learning-rate anneal. Longer runs still depend on the inherited network cosine (cap 1600), the prior cosine (`total_steps`), and the noise horizon of 1200 hardcoded in the hold and shift drivers.

Toys, the frozen recovery control, and the repeated-change stress test were not run. Recovery has to pass before those are meaningful. Details, hashes, and replay scripts are in `result.md`.
