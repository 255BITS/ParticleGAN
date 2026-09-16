# Completed: one-write feedback scouts

All eight2k scouts finished successfully in345.2 seconds on two GPUs. Every new
scout had zero full256/1024 cold-circle passes and zero original-orbit passes
at both prefix lengths. No longer runs were selected. Training cost was31.3s for
the shorter-prefix scout and78.6–81.3s for the others.

The strongest replacement cases reduced stopping but produced rapid alternating
motion. For example, feedback_p50 had zero late-stopped paths but mean absolute
angular speed1.62rad/step and direction consistency.032 at256; the real speed
range is .12–.40. Replacing every eligible context also worsened clean-prefix
startup error to2.866 reference radii at prefix32 (dense baseline .098).
Mature-prefix exclusion, partial replacement, and a strength ramp did not yield
passing paths. These observations are consistent with poor recovery dynamics,
but do not establish a unique mechanism or show all local feedback training fails.

The local target assumption matters: after inserting a generated observation,
the original real next point is a recovery target, not necessarily the natural
continuation of the inserted point. These scouts do not justify escalating to
long training rollouts; the user's constraint remains in force.

Next, six architecture scouts examine explicit recent observations inside D's M
and residual G output, with ordinary local GAN training plus two feedback cases.
No new auxiliary loss. See [round4 plan](../recent_round4/plan.md).

[Full numeric leaderboard](leaderboard.md), [results](results.json), [formulation](plan.md).
