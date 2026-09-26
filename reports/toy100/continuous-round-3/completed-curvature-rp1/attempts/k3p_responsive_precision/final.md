**rp1_signal_close** keeps the ring stable and recovers the shifted target on time, then fails grid100 accuracy. K3P stays the selected base. Nothing is promoted.

The ring results, all on this candidate’s own runs:

| Gate | Result |
|---|---|
| Hold | 1200/1200, min HQ 0.96851, converged at step 1400 |
| Extension | 300/300, steps 2601–2900, min HQ 0.96802 |
| Pre-shift | stationary 5/5, continued hold 120/120 |
| Deadline recovery | **81/81**, delay 310 (stable from step 2710), min HQ 0.91382 |
| Frozen control | **0/81**, Adam updates frozen at 2400 |

The live driver status was UNCONFIRMED, which is what it returns when the quality windows pass. A second active run reproduced that 81/81 diagnostic exactly. `match_frozen_control` on the saved probe fields returned PASS. Pre-shift hashes match between the active and frozen captures for the generator, critic, prior, both optimizers, EMAs, controller scalars, response history, CUDA RNG, and CPU RNG. `means_pre_shift` is reconstructed by subtracting the host shift `(1, 0)`, because the hook runs after the mean update. Per-row latent stats, `response.prior_ids`, and mechanism `pending`/`depth` were not separate witnesses.

The controller is the AP3 cold path without the 800-step dwell and without the peak reset. After the anchor latches, a critic-gradient RMS reopen sets the rate gain to 0.2 and leaves the mixing weight at 0. Two hundred fifty quiet steps, then the existing 0.99 decay, close it again. On this shift the level stayed under 0.25, the rate decayed from 0.000884 back to the floor, and it did not snap open a second time. Input noise ends at update 120 and output noise reaches 0.029 at update 240. Those are fixed step counts, not a fraction of the training budget. The driver still requires a `noise_horizon` argument of 1200, and the host still passes `network_lr_horizon_cap`; the rate rule ignores both.

**grid100, seed 1234, 7000 updates: FAIL.** Coverage passes (100 modes, final HQ 0.98905). Accuracy fails all five terminal checks. The only missed limit is live center RMS 0.261–0.289 against 0.20. Further natives and extra seeds were stopped. rotated100’s training log reached step 7000 (last live 97 modes, HQ 0.8514) and was killed before a verdict file; that is not a score. The 19 transfer toys belong to the other lane.

Mechanism hash `ed49869c6e06e1ac638cf04dba36b123aa7cb3b883650d7e16aa8843aaa22d14`. Report, ledger, replay commands, and artifacts: `result.md` and `tests.jsonl` in this attempt, and `reports/toy100/k3p-responsive-precision/` in the repo.
