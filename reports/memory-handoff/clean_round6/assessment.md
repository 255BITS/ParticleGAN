# Completed: noiseless controls

Both2k jobs finished successfully. Both had zero full cold256/1024 and zero
original-orbit continuation passes at prefixes8/32. clean_gru had24.2% late-window
circle rate and49.2% late stopping at1024; clean_recent4_delta had0% late circles
and58.6% late stopping. No longer runs are justified by these results.

Removing observation noise did not solve the tested models at this budget.
This was an easier diagnostic setting, not a noisy-prefix victory.
See the [session assessment](../feedback-session.md),
[leaderboard](leaderboard.md), and [metrics](results.json).
