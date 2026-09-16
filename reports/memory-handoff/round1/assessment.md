# Completed 2k handoff-only scouts

All ten jobs completed successfully, with no pending/running/failed jobs left in
the initial queue. Wall time was 6.59 minutes on two GPUs. Training used only
single-point GAN losses; all long generated rollouts were final evaluations.

No scout passed the cold 256-point test. `handoff_dense4_start25` passed 1/128
cold 1024-point trajectories (0.78%); every other scout passed zero. None passed
original-orbit fidelity after either real prefix length at either horizon. This
is not a successful toy solution or an established winning formulation.

| Variant | Training time, 2k | Late stopping, cold1024 | First continuation error, prefix32 (radii) |
|---|---:|---:|---:|
| Dense default | 76.8 s | 68.8% | .098 |
| Prefix8 | 20.6 s | 21.9% | .141 |
| Prefix16 | 29.4 s | 35.2% | .103 |
| Input noise .03 | 78.2 s | 56.2% | .114 |
| Input noise .10 | 76.1 s | 4.7% | .191 |
| Memory jitter .05 | 77.7 s | 74.2% | .105 |
| Memory8 | 76.2 s | 65.6% | .193 |
| Memory64 | 78.5 s | 46.9% | .107 |
| Interaction head | 78.7 s | 43.8% | .101 |
| Zero-prefix fraction .25 | 78.2 s | 46.9% | .126 |

Dense next-point supervision improves first-point error over the old single-point
handoff baseline (.118 radii), without making the feedback loop stable. Smaller
memory provides no measured throughput improvement at this scale; shortening real
history does. Input noise .10 reduces stopping and improves original-orbit radial
error relative to the dense default, but still fails geometry/direction fidelity.
These results do not establish that pointwise training cannot work; they establish
failure of these configurations at the tested budget.

Two bounded follow-ups were selected by completed metrics: input noise .10 for
reduced stopping, and zero-prefix fraction .25 for the only nonzero cold1024 pass.
Each resumes its exact 2k model/optimizers/RNGs to 5k on the unchanged 10k schedule,
adding 3k updates. These are diagnostic extensions, not promotion of a solved model.
No trajectory loss or generated training rollout was added.

[Initial leaderboard](leaderboard.md).
[5k follow-up leaderboard](../long/leaderboard.md).
