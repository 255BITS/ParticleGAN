# Adversarial memory exploration: round9

The user rejected MSE training objectives. This plan supersedes the proposed
prediction-space MSE repair in dynamics_round8/next-experiments.md. Metrics may
still use squared error. Existing legacy configs remain supported.

## Hypothesis

Generated samples explore reachable memory states. G may learn useful recovery
if D judges the resulting continuation against the undisturbed real history.
Previous feedback experiments instead gave D and G the same disturbed memory.

For a target at t+1, with one fixed particle per episode:

    M = D.write_history(real[:t])
    proposed = G(z, M, clock(t))
    replacement = (1-alpha)*real[t] + alpha*proposed
    M_explored = D.write(M, replacement)
    M_reference = D.write(M, real[t])
    fake = G(z, M_explored, clock(t+1))
    real_score = D.score(real[t+1], M_reference)
    fake_score = D.score(fake, M_reference)
    losses = ordinary_particle_GAN_losses(real_score, fake_score)

Both scores and exact default B-cap use identical reference memory. No target or
future point enters either history. In shared controls, both scores instead use
M_explored. Half of eligible training positions receive replacements; all other
positions use ordinary real history. Positions below4, including zero memory,
retain ordinary training. Actual selected replacement fraction is logged.

D phase: fake point/proposal detached; D trains its writer through the judging
history and its candidate head. In clean judging, the explored writer branch
does not receive D gradients. G phase: rebuild contexts after D updates; freeze
D parameters but preserve derivatives through the replacement write to G's
proposal. A detached-proposal control disables this route. D alone owns and
trains the writer. The reference is used only for judging during training.

Runtime remains M=0, x=G(z,M,clock), M=D.write(M,x), repeated without an expert.
Training uses at most one generated write followed by one generated read per
sampled target. It is bounded local feedback, not a full generated rollout.

## Proposal adapter

Optional stateless G refinement:

    proposal = H(z, M, clock)
    readable = M + R(M, proposal)
    final = H(z, readable, clock)

R is an identity-initialized residual adapter (64 hidden,16 bottleneck). It uses
the proposal as a nonlinear summary of particle, clock and memory; it has no
extra direct clock input. Both H passes share weights. Only final is written to
D memory, never readable. The memory-only residual control has the same hidden
width/bottleneck and128 fewer first-layer parameters. This comparison therefore
changes conditioning and computation, not exactly parameter-matched capacity.
No denoising/regression/stability/temporal/prediction auxiliary is used.

## Predeclared scouts

All fresh2k updates, unchanged10k schedule, M32/G64/D128, batch128x4 target
positions, six-band clock, default data/noise and public API exact B-cap. No
clipping, EMA, seed sweep, private G memory or trajectory objective. Set
adversarial_only=true to reject auxiliary-loss configs. Prior regularization
remains the standard particle GAN recipe and is applied once per update.

Replacement alpha ramps linearly over500 updates to .25 (mild) or1 (full).
This is a fixed blend, not a distance gate. Full replacement intentionally tests
larger deviations; recovery to the original process is a modeling assumption.

| Scout | Judging memory | Alpha | G adapter | Gradient through proposal |
|---|---|---:|---|---|
| clock_control | real, no exploration | 0 | none | n/a |
| shared_s25 | explored | .25 | none | yes |
| clean_s25 | real | .25 | none | yes |
| clean_s25_detach | real | .25 | none | no |
| shared_full | explored | 1 | none | yes |
| clean_full | real | 1 | none | yes |
| proposal_control | real, no exploration | 0 | proposal | n/a |
| proposal_clean_s25 | real | .25 | proposal | yes |
| proposal_clean_full | real | 1 | proposal | yes |
| residual_clean_s25 | real | .25 | memory-only residual | yes |

Comparisons: clean/shared isolates judging context at each strength; clean
connected/detached isolates feedback gradients; residual/proposal clean tests
adapter conditioning; proposal/no-adapter controls measure refinement without
exploration. A proposal+shared factorial arm is deferred pending evidence.

Complete G calls per D/G phase:1/1 for controls,2/2 with exploration. Internal
reader calls are doubled for proposal adapters, hence up to4/4 with exploration.
Proposal runtime is two reader calls per point. Record measured cost too.

## Pipeline and selection

Queue: runs/memory_path/exploration_round9, both cuda:0 and cuda:1. Drain emits
completed/failed notifications only; leaderboard consumes completed runs only.
Stable central log:

    tail -F runs/memory_path/core_round1/train.log

Primary: full cold-circle and warm original-orbit pass fractions at256/1024,
prefixes8/32, startup included. Report signed-speed error, direction agreement,
long radial error, first32 position error, stopping and particle coverage.
No images for ranking. Evaluate one-write error and memory/clock/adapter use on
completed models. MSE in these diagnostics is not part of training.

Extend at most two exact checkpoints to5k if primary metrics improve with
supporting continuous errors and no return to widespread stopping. If all full
pass rates remain zero, allow at most one diagnostic extension only if BOTH
prefixes improve long radial error by at least25% against a matched control,
improve early position error and one-write next-prediction error, and do not
worsen direction agreement or stopping. Compare clean to shared at equal alpha,
proposal to its no-exploration control and the corresponding plain clean run,
and residual to plain clean. Do not promote solely on denoising or loss values.

Cold bootstrap and warm process preservation remain distinct unsolved tasks.
Benefits under the learned128-particle evaluation panel do not establish unseen
radii/speeds or held-out-particle generalization. Clock recurrence diagnostics
can follow promising models; a repeating wrong path alone is not proof of
corrupted memory. No automatic change to clock encoding is included this round.
