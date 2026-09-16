# Completed: recent-memory architecture scouts

All six2k scouts finished successfully. All full cold256/1024 and1024-step original-orbit
pass rates were zero. The bounded GRU control had one prefix32 warm256 pass
(1/128), which did not persist to1024; all other short warm rates were zero. Explicit recent points and residual prediction did not
solve the task at this budget. Detached feedback substantially worsened drift
in the residual models (prefix32 radial errors517 and1466 reference radii).

One diagnostic lead is recent4_delta without feedback: late-window circle rate
53.9% at256 and60.2% at1024. This is not full-path success or faithful handoff.
Its prefix32 radial error remains1.085 and reference direction agreement48.6%.
A bounded exact extension to5k is selected to test whether startup improves;
it must retain the original full-path metrics, without discarding burn-in.

Four matched single-write gradient scouts will test whether allowing G to learn
through its earlier output helps. They retain the D-owned writer and at most two
G evaluations per phase. See the corrected round5 plan; the first round5 attempt
was invalidated for a real-score context-gradient omission.

[Leaderboard](leaderboard.md), [results](results.json), [architecture](plan.md).

A completed-only direction audit found that all77 passing final128-point windows
were CCW (also77/128 on the final256 points;76/128 on the final512). Thus even the
late-circle lead lacks direction coverage. See `recent4_delta_late_windows.json`.
