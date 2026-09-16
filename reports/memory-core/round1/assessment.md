# Completed core formulation round 1

All nine jobs completed successfully: seven new 2k scouts and two existing-control
reevaluations. No pending/running/failed jobs remain. Queue wall time was 53.2 minutes.

Handoff plus cold feedback improves cold circle success over the matched 2k
autonomous control (15.6% at 1024 points). Heavy handoff reaches 33.6%; light handoff
reaches 26.6%. Heavy has skewed passing directions (4 CW / 33 CCW at 256), while
light is balanced (18 / 16) and has better measured marginal coverage. Both have
zero late stopping. These are single-initialization scout comparisons, not a claim
of statistically established superiority.

All formulations, including the old 10k control, get zero reference-orbit passes
at both prefix lengths and both evaluation horizons. The toy remains unsolved.
Handoff-only has the best first continuation point (0.118–0.121 radii mean error)
but loses the orbit and has 60.2% late stopping from cold starts. This supports
training feedback explicitly; good one-step handoff is insufficient in this run.
Handoff+warm improves reference-direction agreement to about 75% but still has
large geometry and speed errors. Prefix training as implemented is not sufficient.

Recommend extending handoff_cold_light first for balanced coverage, with
handoff_cold_heavy as the stronger raw-circle-rate comparison. Keep warm fidelity
as an independent decision axis; neither candidate currently solves it. Diagnose
how reference geometry/direction information decays after the cutoff before
spending on another wide architecture sweep. No extensions were launched during
the completion/status review.

Full numeric leaderboard: [leaderboard.md](leaderboard.md).
