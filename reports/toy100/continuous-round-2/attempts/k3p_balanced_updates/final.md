K3P stays the selected base. Three coupled-mobility candidates each ran their own shift and their own hold. None passed hold, the 300-update extension, and the deadline recovery together, so nothing was promoted.

| Candidate | Shift | Own hold + extension |
|---|---|---|
| K3P parent (not rerun) | deadline **28/81** | **1200/1200** and **300/300** |
| kbu3 late quiet band | stationary **0/5**, continued **101/120**, deadline **0/81** | **FAIL** after 884 good hold updates; extension **NOT_RUN** |
| kbu1 joint band | stationary **5/5**, continued **33/120**, deadline **0/81** | **FAIL** `NOT_CONVERGED`, streak 120; extension **NOT_RUN** |
| kbu2 quiet leak | stationary **0/5**, continued **0/120**, deadline **0/81** | **FAIL** `NOT_CONVERGED`, streak 0; extension **NOT_RUN** |

The shared multiplier is one factor on the generator, the critic, and the prior. Mixing follows that factor, not the learning-rate clock. Host floors were 1 and 1, so the cosine schedule stayed at the recipe rates (0.00425 and 0.0085). Applied rates tracked the multiplier. The driver noise horizon of 1200 updates is still there and was not removed.

What the failures narrow:

- Full rate acquires 8 modes by update 500 and then collapses at update 1600. kbu1 never left a multiplier of 1, because a 100-step quiet streak never formed.
- Leaking as soon as the critic step drops below relative displacement 0.02 locks 7 modes. kbu2 never reached 8, including across a 4800-check hold.
- Waiting for displacement below 0.012 avoids that collapse and holds 8 modes from update 1510 through the shift, but the 8th mode arrives after the fixed 1000–1200 window. At the shift the critic moment ratio was 0.53, so the capped band did not reopen. Recovery stayed **0/81** and finished at 6 modes. The separate hold then missed HQ 0.90 once, at update 2590, with 8 modes still present.

Toy gates, the matched frozen control, and a two-horizon prefix were not run. The next useful signal is one that actually jumps when the data move. The moment ratio did not. The leak still has to stay at full rate through update 500 and be down before update 1600. Details, hashes, and replay commands are in `result.md`.
