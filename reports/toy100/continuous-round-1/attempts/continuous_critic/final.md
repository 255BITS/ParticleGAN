Three proposals completed; **none qualified**. K3P remains unchanged.

| Candidate | Hold | Recovery deadline |
|---|---|---|
| K3P parent | PASS 1200 + 300 extension | FAIL 28/81 |
| C1 static anchor | NOT_CONVERGED | FAIL 0/81 |
| C2 innovation damping | NOT_CONVERGED | FAIL 0/81 |
| C3 stationary innovation | Failed check 27; worst HQ .768 | FAIL 18/81 |

C3 passed the **1,024-update horizon-invariance audit**, with its EMA anchor active under constant positive rates. Full-22 qualification and repeated-change stress remain **NOT_RUN**. Three audit serialization ERROR records were preserved after fixing the serializer.

Recommendation: retain K3P; next isolate C3’s precision drops with reversible critic damping while holding rates and noise fixed.

[Report, leaderboard, and replay commands](../result.md) · [Test ledger](../tests.jsonl)