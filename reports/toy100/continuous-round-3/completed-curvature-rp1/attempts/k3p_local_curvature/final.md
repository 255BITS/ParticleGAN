K3P stays the selected base. Three local-secant candidates each ran their own hold and target-shift tests. None passed hold, so none advanced.

| Candidate | Hold | Shift deadline | What the rate did |
|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 + 300 | 28/81 | Scheduled floors |
| lc1 absolute secant | 314/1200, then HQ 0.878 | 56/81, pre-hold already failed | Stuck at the initial Adam rate, mixing weight 1, anchor off |
| lc2 early-mean secant | 0/4800, ended at 7 modes | 0/81 | Ratcheted to the floor by step ~400 and froze a short cover |
| lc3 step-independent secant | 0/4800, ended at 4 modes | 0/81 | Stayed near 20% of the base rate, mixing near 0.5, and reopened without rebuilding eight modes |

The absolute Malitsky ratio sits above this Adam cap, so it never leaves the initial rate. Rescaling it by an early mean ratchets to the floor because the ratio shrinks with the step. Canceling that factor stops the ratchet and still leaves a partial rate that locks four or five precise modes. Negative directional curvature is too common on this ring to use as a growth signal.

Extension, frozen control, the 22 toys, horizon-prefix checks, and repeated shifts were not run. Input and output noise are still K3P's horizon schedule on every candidate. Details, hashes, and replay commands are in `result.md`, with one ledger row per gate in `tests.jsonl`.
