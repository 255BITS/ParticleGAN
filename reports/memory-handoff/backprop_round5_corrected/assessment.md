# Completed: corrected bounded-gradient comparison

All five valid jobs finished successfully in 306.5 seconds on two GPUs. Four
fresh 2k scouts retained G gradients through one generated write, including both
real/fake score branches of the paired API loss. All had zero full cold-circle
and original-orbit passes. This corrects the implementation, but does not solve
local recovery training. The residual-memory feedback models still drift badly.

The exact recent4_delta continuation to 5k also failed: full cold/warm passes
remain zero; late-circle rate at 1024 fell from 60.2% to zero and late stopping
rose to 84.4%. Prefix32 radial error grew from 1.085 to 21.085 reference radii.
Do not promote it further based on its earlier late-window result.

An evaluation-only latent swap probe on the 2k model yielded just one full cold
pass when swapping after point1, and none when swapping after point4. Those
variants violate the fixed-particle policy and are excluded from leaderboards.
This is not convincing evidence that changing particles repairs the failure.
See `../recent_round4/latent_intervention.json` and the reproducible diagnostic
`experiments/diagnose_memory_latent.py`. The unmodified rollout was checked
against saved trajectories before interpreting the interventions.

The earlier `backprop_round5` queue is invalidated and excluded, including its
two completed jobs. This corrected queue has no training failures. Reporter
guards reject invalidated queues and baselines; checkpoint guards reject legacy
backprop checkpoints with the missing real-context gradient. 60 focused tests
plus one reporter-exclusion test passed.

The next controlled check removes observation noise in two otherwise matched
models (ordinary GRU and recent4 residual). It is an easier data setting, not a
claim that the original noisy task has been solved. No further noisy-data
extensions are justified by the present results.

[Leaderboard](leaderboard.md), [metrics](results.json), [formulation and fix](plan.md).
