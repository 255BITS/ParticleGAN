K3P stays the selected base. The best candidate in this lane, **ap3_partial_reopen**, passes its own ring hold and extension and improves recovery, but the shift deadline still fails, so nothing is promoted.

| Candidate | Hold | Extension | Shift pre-hold | Deadline recovery |
|---|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 | 300/300 | parent evidence | FAIL 28/81, delay 1130 |
| **ap3** | **PASS 1200/1200**, min HQ 0.96851, converged at step 1400 | **PASS 300/300**, min HQ 0.96802 | **120/120**, min HQ 0.97559 | **FAIL 72/81**, delay 780 |
| ap1 | NOT_RUN | NOT_RUN | 120/120 | FAIL, stuck at 4 modes, final HQ 0.91675 |
| ap2 | NOT_RUN | NOT_RUN | 120/120 | FAIL, oscillated, final 7 modes, HQ 0.73608 |

The phase signal is critic gradient size against a decaying peak, not critic cosine and not the learning-rate clock. All three candidates acquire 8 modes by step 500 at full rate with the early penalty, then damp to the K3P floors. On a later jump in gradient size:

- ap1 also turned the early penalty back on and settled on 4 precise modes.
- ap2 kept the EMA anchor and restored the full base rate. Modes moved and never locked at 8.
- ap3 kept the anchor and restored a partial rate (generator/critic 0.000884, prior 0.00204). Eight modes were back by step 2600. Nine deadline checks from step 3070 through 3170 then dropped as low as 5 modes and HQ 0.34839. Sustained recovery starts at step 3180. Final live state is 8 modes at HQ 0.99194. Both networks and the prior took all 3600 Adam steps, with nonzero post-shift movement.

Toys, the matched frozen control, the horizon-prefix audit, and the repeated-change stress test are **NOT_RUN**. Input and output noise still follow the driver's 1200-step horizon. That is a labeled leftover schedule, not a horizon-free formulation.

The deadline misses happen while ap3 is still forcing that partial rate for a fixed 800 steps after a recovery that had already reached 8 modes. A next attempt can let the post-shift gradient peak decay that partial rate, while keeping this cold path and keeping the anchor on. Full details, hashes, and replay commands are in `result.md`.
