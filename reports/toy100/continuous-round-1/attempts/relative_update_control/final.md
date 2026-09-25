K3P still leads. A reversible step-size rule can keep its hold and extension, and the same rule made target-shift recovery worse.

The parent was not rerun. Its published scores stay 1200/1200 hold, 300/300 extension, and 28/81 recovery with a 1130-update delay. Three candidates changed only `mechanism.py`. The particle rule, direct response, and config match the pinned hashes. Host learning-rate multipliers were the identity (floors 1 and 1), so the cosine horizon does not move rates. Critic mixing reads a training signal, and the exponential-moving-average anchor stays that signal's stationary term.

| Candidate | Hold | Extension | Deadline recovery |
|---|---|---|---|
| K3P parent | 1200/1200, min HQ 0.907 | 300/300, min HQ 0.988 | 28/81, delay 1130 |
| ruc1 energy peak | not run; never reached 8 modes | — | 0/81 |
| ruc2 leak from step 0 | not run; stuck at 6 modes | — | 0/81 |
| ruc3 quiet-then-leak | **1200/1200 PASS**, min HQ 0.989, converged at step 1603 | **300/300 PASS**, min HQ 0.996 | **0/81** |

ruc1 collapsed the step off the initial gradient and peaked at 4 modes. ruc2 cut the rate immediately and locked a 6-mode state at HQ 0.999. The parent, still at full rate, has 8 modes by step 600.

ruc3 holds the full rate for 800 quiet steps, then leaks toward the 1% / 5% floors. That acquires the ring: 8 modes by step 500. Relative displacement `rms(Δθ)/rms(θ)` falls from 0.085 at step 100 to 0.00021 at step 2400, with the critic learning rate at 4.39e-5. A moment-ratio crossing after the shift reopens the critic rate to 0.00423 and the relative step to 0.029, and later crossings keep it near full rate through step 3600. The anchor stays on during that reopen. Recovery passes 0 of 120 post-shift checks and ends at 6 modes, HQ 0.75. The parent's small floor step eventually reacquired. This sticky reopen did not.

The shift window 1200–2400 is 109/120. The 11 failures are steps 1300–1400, minimum HQ 0.699, during the leak. The convergence gate starts its 1200-update hold only after that dip, at step 1603. No toy screen and no frozen control: live recovery did not pass.

Two 40-update prefixes with horizon caps 1600 and 4800 matched on every mobility row and on the RNG digest. The remaining scheduled piece is the drivers' 1200-update noise horizon. ruc3 also uses a fixed 800-step quiet count, which is not a fraction of the training budget.

Code is under `repo/reports/toy100/relative-update-control/ruc1|ruc2|ruc3/`. Runs, replay commands, and the gate log are in `result.md` and `tests.jsonl` in the attempt directory.
