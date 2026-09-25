K3P stays the selected base. Three critic-confidence proposals each ran the canonical ring hold and the target-shift test. All six gates failed. Nothing is promoted.

Half-batch gradient agreement and paired-margin uncertainty do not supply a usable close. Agreement stayed high for the whole run, so the rate never left its initial value and the anchor never turned on. Margin spread is high on a multi-modal batch, so that rule hit the floor at step 202, before eight modes existed. A third rule, critic-gradient size against its own slow average with a frozen baseline, also never closed: the average tracked the gradient, and the "below half for 100 steps" test did not fire.

| Candidate | Hold | Extension | Shift deadline | Stationary / continued hold |
|---|---|---|---|---|
| K3P parent (published, not rerun) | 1200/1200 | 300/300 | FAIL 28/81 | published pass |
| cc1 half-split coherence | FAIL after 153 hold updates (converged at 3933, lost at 4087) | NOT_RUN | FAIL 0/81 | 0/5 and 0/120 |
| cc3 frozen-baseline quiet | FAIL, longest 8-mode streak 120 | NOT_RUN | FAIL 0/81 | 5/5 and 33/120 |
| cc2 margin uncertainty | FAIL, streak 0, ended at 6 modes | NOT_RUN | FAIL 0/81 | 0/5 and 0/120 |

cc1 kept critic learning rate at 0.00425 for every update, including all 3600 shift steps, so the generator was still moving and the controller simply did not react. cc2 reached the floor and stayed there; the shift then fell from about HQ 0.83 to 0.45. cc3 did pass the short stationary window (5/5) and then missed the continued pre-shift hold (33/120). Its quiet counter never reached 100, so the frozen-baseline reopen guard was never exercised.

Host noise is still the original schedule on all three. The ring drivers still pass a 1200-step noise horizon. Image screening, the 22-toy matrix, native 7000-update runs, and the frozen control were not run. The pinned parent mechanism is unchanged.

Full scores, hashes, and replay commands are in `result.md`. The next close needs a reference taken once and then frozen: a tracking average never separates, and the RP1 grid trace showed a stationary reopen to multiplier 0.208 undoing a center RMS that had improved to 0.206 at the floor.
