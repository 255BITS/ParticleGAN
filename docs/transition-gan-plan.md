# Shared-latent transition GAN: planning handoff

Status: Implementation authorized and resumed. See [the runnable example](transition-gan.md).
The original planning choices below are preserved as rationale. After an initial
point-particle GAN comparison, the user clarified that the intended recipe is MoG
with about 1k particles and requested marginal critics. The current example uses
the MoG preset with 1,024 components and 28,000 updates; joint-only and
joint-plus-marginal discriminator configurations are available.

## Intent and agreed architecture

Start from the repository's trajectory-generation work, but generate one transition
instead of an entire trajectory. The motivating hypothesis is that jointly learning
states, actions, and outcomes can help their individual generators.

```text
             G_s(z, context) -> s
shared z ->  G_a(z, context) -> a       -> D(s, a, s_next, context)
             G_next(z, context) -> s_next
```

Each component receives the SAME sampled particle latent. Do not silently replace
this with G_a(s,z) or G_next(s,a,z): the user specifically questioned that additional
conditioning and wants to explore coordination through shared latent inputs and
joint feedback. External observed context is compatible with the existing toy;
it is not another branch's generated output.

Initially use one joint discriminator. With reference triples it can constrain both
marginals and their relationships. Additional marginal discriminators are later
ablations, not a prerequisite. Every extra discriminator needs reference evidence.

## What exists in this checkout

- Branch: feature/world-model-no-trajectory (created from
  feature/masked-observation-gan for this separate line of work).
- lib/trajectory.py: Routes analytic data family; temporal U-Net generator;
  MLP/temporal/hybrid discriminators; DDGAN and one-shot GAN sampling.
- experiments/train_trajectory.py: config-driven trainer using public recipes,
  learned particle prior, Rp logistic, lazy gradient cap penalty, prior spread
  regularization, constant learning rate, and EMA.
- reports/trajectory/README.md and reports/trajectory/hybrid/READOUT.md describe
  completed experiments; configs/trajectory holds existing configurations.
- The existing task predicts 64 future 2D positions from observed approach,
  obstacle geometry, and a route-preference class. It draws analytic paths; it
  does NOT have action labels or an action-driven physics simulator.
- reports/trajectory/hybrid/MOTION_HANDOFF.md concerns a later recorded-human-motion
  task. There are untracked results/motion artifacts, but no motion trainer in the
  current tracked source. Do not assume those are the intended simulator example.
- Existing unrelated untracked user files must be preserved. The earlier
  docs/masked-observation-gan.md is a separate proposed missing-data experiment.

## Recommended minimal task (proposal, not yet user-confirmed)

Preserve the existing route family and geometries. Sample two adjacent positions
p(t), p(t+dt) and form a six-dimensional record:

```text
s      = p(t)                 # 2D position
a      = p(t+dt) - p(t)       # 2D displacement
s_next = p(t+dt)              # 2D position
```

The known one-step relation is F(s,a)=s+a. Here "action" means displacement, not
force or a recovered controller command. This is a deliberately simple kinematic
consistency test. If the user means physical controls, select a small controlled
simulator first instead of presenting derived displacements as those controls.

Use observed geometry, route preference, and normalized physical time as context.
Do not expose the sampled route ID or the analytic path coefficients. Sample the
same hidden route/coefficient draw for both positions. Evaluate the analytic route
formula only at the requested times: neither the networks nor their losses need
whole trajectories. Verify equivalence to adjacent positions from Routes.sample.

G_next predicts its own next position. F is an evaluation oracle initially; do not
overwrite generated next positions with F(s,a) or impose a consistency loss in the
main arm. That would remove the relationship we want the joint GAN to learn.

Scale state, action, and next-state blocks using training-only statistics, frozen
across arms. Displacements are much smaller than absolute positions; otherwise a
joint discriminator or distance can effectively ignore actions. Report physical
consistency errors after converting back to physical coordinates.

## First comparison

1. One MLP generating the concatenated six-dimensional transition.
2. Three MLP branches with independent parameters receiving the same latent,
   generating state/action/next-state separately, with one joint discriminator.

Use independent branch MLPs if testing a distinct architecture: a shared trunk with
three linear output heads is algebraically equivalent to one concatenated linear
output layer. Match total generator parameter budgets approximately and report
exact counts; keep prior, D, losses, context, update budgets, and real draws matched.

Use a one-shot particle GAN first. The repository already has a one-shot trajectory
arm; adding diffusion is a separate question. Use one seed, never seed-only repeats.
Run correctness checks and short viability scouts, then a bounded matched-budget
comparison (proposed 1k scouts and 10k final steps, following the existing study).
Treat old trajectory results as context, not a matched transition leaderboard.

## Measurements and checks

- Held-out geometry conditional joint sliced Wasserstein-1 on normalized triples;
  separate interpolation/extrapolation, and per-block marginal distances.
- Physical consistency: norm(s_next - s - a), including its distribution/quantiles.
- Coverage and spread of local transitions, plus real-vs-real reference floors.
- Diagnostic shuffled-branch control: independently permute generated s/a/s_next
  WITHIN each fixed context, preserving marginals while breaking correspondence.
  Joint/consistency diagnostics must detect this failure on real data too.
- Test shared-latent sampling, gradients into every branch and the learned prior,
  scaling round trips, and sampler agreement with the existing analytic routes.
- Visualize generated and reference transition arrows at selectable scene, class,
  and time, plus predicted next points versus s+a and residual histograms.

No claim about a rollout-capable world model: the initial model jointly samples
triples, and does not directly answer arbitrary supplied state/action queries.
No claim that joint learning helps an individual marginal until measured against
an appropriate marginal-only baseline, which can follow this first viability test.

## Follow-ups only after the base experiment

- Joint D versus joint D plus marginal critics, with explicit reference datasets.
- Independent branch latents as a structural negative control, not a seed repeat.
- Joint versus marginal-only training to test whether learning the relationship
  actually improves individual state/action/outcome distributions.
- Sparse paired triples plus abundant partial records to reconnect with MisGAN.
- Physical action/state dynamics, conditional inference, or rollouts as separate
  extensions once the generated joint transitions are sound.

## Execution and reporting expectations

Use existing runner provenance/config/summary contracts. New paths should be scoped
under transition experiments (e.g. results/transition/ and configs/transition/).
Flush progress to per-run log.txt and results/transition/live.log; provide a single
tail -F command. Recheck GPU availability at execution time.

After completion produce a leaderboard, explain joint versus marginal performance
and failure modes, and recommend the next discriminating experiment. Retain configs,
samples, checkpoints, source provenance, and a compact visual report. The user asked
for token efficiency and easy-to-tail logs. Do not spawn agents without authorization.
