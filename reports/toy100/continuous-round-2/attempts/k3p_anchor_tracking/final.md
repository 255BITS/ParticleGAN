K3P stays the selected base. None of the three anchor-tracking candidates passed hold, the 300-update extension, and the shift verdict together, so nothing was promoted.

| Candidate | Hold | Extension | Shift | Toys |
|---|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 | 300/300 | FAIL 28/81, delay 1130 | 22/22 prior evidence |
| at3, constant peak rate, slow reference after 200 steps | FAIL, 356 good checks, broke at step 5778 (HQ 0.611) | NOT_RUN | FAIL: stationary 0/5, continued hold 0/120, deadline **38/81**, delay 1070 | NOT_RUN |
| at2, rate from generator-gradient energy | FAIL, stopped at step 4350, max 7 modes, HQ ~1, streak 0 | NOT_RUN | NOT_RUN | NOT_RUN |
| at1, cosine persistence and Adam innovation | FAIL, stopped at step 5650, max 7 modes, streak 0 | NOT_RUN | NOT_RUN | NOT_RUN |

at3 is the only run that reached 8 modes. On the shift it kept both optimizers stepping at the peak rates (discriminator and generator 0.00425, prior 0.0085, 3600 updates, no reset) and finished at 8 modes with HQ 1.0. Sustained recovery was at step 3470, past the deadline, and the pre-shift hold was 0/120. The separate hold also failed, so the higher deadline count does not outrank K3P.

What the failures narrow:

- Critic-gradient cosine stays near zero during acquisition, and Adam innovation is already quiet by about step 100. Neither one waits for 8 modes.
- Generator-gradient energy is already small at 7 modes and HQ 1. Flooring the rate there locks out the eighth mode (at2).
- Turning the slow anchor on at step 200, even with the rate held at the K3P peak, misses the parent's step-600 acquisition (at3: 6 modes, HQ 0.45 at step 600).
- Reopening the full rate from a one-tensor Adam spike while the anchor is still on knocks modes off (at1).

A later rule has to keep pure `a_r1r2` and the peak rate until a signal that is still absent in that 7-mode, HQ-1 state. This round did not find that signal. The matched frozen control, the sensitive four, the 22 toys, the two-horizon prefix, and the delayed/repeated change were not opened.

Sources, hashes, replay commands, and the gate ledger are in `result.md` and `tests.jsonl` next to this attempt. The pinned K3P mechanism hash is unchanged.
