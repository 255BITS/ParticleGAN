# Gaussian acquisition smoke and Tier 2 stability

The new Tier 1 question is **can training produce a correct Gaussian?** Any one
of 24 scheduled states must pass the unchanged finite, location, width and KS
bounds, with a second independent draw passing at the same state. All 1,000
updates and all 24 paired observations still finish. No learning shutdown or
best-checkpoint serving is introduced.

The new Tier 2 question requires every one of 72 stationary observations between
updates 1,000 and 4,000 to pass. Training then changes the target mean from 2 to 3,
continues for 2,000 updates without resetting optimizer/history or the original
schedule, reacquires by update 5,000 at five terminal checks, and passes every
remaining 24 shifted hold checks. A frozen copy of the learner's own update-4,000
state receives matched evaluation draws.

These are new task identities with 256 learned uniform MoG particles, sigma `.1`,
latent dimension 2, width 32, depth 2 and two critic Fourier frequencies. The
original sigma-`.025` `gaussian1d_acquisition` task and historical five-terminal
verdicts remain unchanged. The ordinary view is revision 7 with required counts
6 / 20 / 2. Screening remains provisional until calibration passes.

A zero-update confirmation of the fixed historical BCAP update-1,000 checkpoint
is being published separately. It supplies no ordinary qualification and retains
the original failed five-terminal acquisition verdict. The parent inventory run
will measure the new ordinary tasks under their exact bindings.

The public shared host is `experiments.forge.gaussian_tasks`; architecture
variants supply explicit task cards to `benchmarks.toy_audit.gaussian_smoke_study`.
All model/trainer/sample/restore execution requires CUDA. CPU reference scoring
and rendering are explicit exceptions. Constructor, target, trainer sampling,
primary evaluation and confirmation streams remain isolated and checkpointed.
