K3P stays the selected base. A shared bound on realized Adam displacement did not pass hold, the 300-update extension, and recovery together. Nothing was promoted.

| Candidate | Hold + extension | Shift deadline | What failed |
|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 and 300/300 | 28/81 | Published parent |
| jt2 growth cap, ceiling 0.10 | Confirmed at step 3351, then failed at 3436 | 0/81, max 5 modes | One hold sample at HQ 0.871 while 8 modes were still present. Extension not started |
| jt3 growth cap, ceiling 0.50 | Not confirmed, streak 0 | 0/81, ended 7 modes | Matched the unclipped run through update 100, then diverged before update 200 |
| jt1 fixed ball of radius 0.05 | Not confirmed, streak 0 | 0/81, ended 5 modes | Never reached 8 modes. The ball reopened to 0.05 during a 7-to-3 drop |

Host learning rates stayed at 0.00425 and 0.0085 because the floors passed to the drivers were 1. jt1 sets the critic mixing weight from the radius. jt2 and jt3 keep that weight at 1 and use the displacement cap as the critic constraint, so the EMA anchor stays off. Both optimizers kept stepping after the target change.

Toy gates, the frozen recovery control, the horizon-prefix check, and the repeated-shift stress test were not run. The full write-up, replay commands, and ledger are in `result.md` and `tests.jsonl` for this attempt.
