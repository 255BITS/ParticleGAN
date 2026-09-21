# Transition GAN with an observation encoder

The no-argument example and `configs/transition/default.yaml` now select
`encoder_shared_state`, the winning recipe, with a fresh default output path
`results/transition/default`. Named historical configs preserve their original
settings; completed results and their pinned metadata remain unchanged.

This experiment asks whether learning to encode real transitions also helps the
original joint sampler. It adds an inference path to the existing MoG/bcap
example, using the repository's deterministic `particle_ae` routing.

```text
Original sampling (same noisy z and observed context):
G1 -> st
G2 -> at
G3 -> st+1

Real input:
E(st, at) -> z -> G1 -> reconstructed st
              -> G2 -> reconstructed at
              -> G3 -> predicted st+1

Synthetic composition:
G1 -> st ─┐
G2 -> at ─┴-> E(st, at) -> z_hat -> G3 -> st+1
```

Each G remains an independent MLP. E also receives the observed geometry, time
and preference class; it never receives the target next state. E chooses one of
1,024 standardized MoG centers and adds a bounded offset:
`z_hat = selected_center + sigma * 3 * tanh(offset / 3)`.
Routing uses temperature .25, summed squared distances and the repository's
straight-through gradient. E's query is layer-normalized, its offset head starts
at zero, and there is no KL penalty or latent-ID reconstruction target.

## The two arms

Both start from the settings of `concat_class8_marginals`: three Gs of width128,
class-input gain8, geometry/time gain1, scalar class-concatenated critics.

- `encoder_separate`: `D_joint(st, at, st+1)`, `D_state(st)`, `D_action(at)`,
  `D_next(st+1)`.
- `encoder_shared_state`: the same encoder/objective, with one `D_state` serving
  both state roles. `D_joint` and `D_action` remain separate.

The shared state critic receives actual time: t for st and t+dt for st+1. It
uses a common state scaler derived from an equally weighted mixture of the two
training-state populations. The conversion first undoes the original per-block
normalization; the common variance includes between-population mean differences.
The joint/action critics and benchmark scaler retain their original coordinates.
This arm therefore tests weight sharing together with the normalization and time
alignment needed to make that sharing meaningful.

## Exact objectives and gradient paths

Let `A(x)` denote the generator's usual adversarial loss:
`joint(x) + mean(state(x), action(x), next_state(x))`.
Let x be the original generated triple and y the composed triple that retains
x's state/action but gets its next state by encoding those inputs and running G3.

```text
L_G,E,prior = (A(x) + A(y)) / 2
            + MSE(G(E(real_st, real_at)), real_triple)
            + MSE(G1/G2(E(fake_st, fake_at)), stopgrad(fake_st, fake_at))
            + raw_particle_spread
```

Both MSE weights are1. Real MSE averages all six normalized coordinates, so it
weights the three two-coordinate roles equally. Synthetic MSE averages the four
state/action coordinates. Real prediction supervision is new relative to the
adversarial-only baseline. The G update reuses its existing real batch; it makes
no additional real-data draws.

The synthetic input remains differentiable: feedback can flow through G3, E,
and the original G1/G2. Only the synthetic reconstruction target is detached.
The selected particle centers also receive decoder gradients; routing uses the
existing straight-through surrogate. The prior spread term is applied once to
the full raw center table. No objective uses the analytic `st + at` identity.

For D, the first half of a batch uses original triples and the second half uses
composed triples, generated without gradients. The real batch, D batch size,
number of updates and per-role bcap penalties stay fixed. Each role contributes
its full Rp logistic plus bcap loss. Shared state weights receive both state-role
losses; generator marginal weighting remains an average over three roles.
The logged `g_terms` describe original-path adversarial terms; `g_loss` is the
mean over the original and composed paths.

## Recipe and comparison limits

We adapt encoding to `get_recipe("mog")`; we do not switch to the different
`ae_gan` optimizer/training preset. Both arms use seed24002, 1,024 particles,
z_dim32, fixed sigma_rel.025 (calibrated sigma approximately.13112), batch256,
28k updates, Rp logistic, bcap cap1/coefficient1/every update, the existing
optimizer groups, raw-center spread, cosine schedule and EMA. E joins G's Adam
parameter group and gets its own EMA; the prior retains its faster optimizer
group. E initialization has a separate seed, identical between these two arms.
No seed-only repeats.

The real draw budget remains14,336,000 training observations plus32,768 for
normalization. These are fresh analytic samples, not a finite scarce dataset.
The encoder adds capacity and computation. Its exact parameter count is reported
separately from G and D; leaderboard encoder entries form an explicitly marked
paired-supervision cohort. Historical generator-capacity and protocol checks
remain in force. GPU wall times are descriptive, not controlled throughput tests.

Action is displacement in this toy, so `st+1 = st + at` is already a perfect
analytic predictor. Learning this through E/G3 is a representation/coordination
experiment, not evidence of learning unknown physical dynamics. This deterministic
encoder also does not model multiple possible outcomes for an identical input.

## Evaluation and running

The original prior-generated benchmark is unchanged and still determines the
leaderboard rank. The same saved references additionally evaluate:

- Real-input next-state Euclidean error, its p95, and state/action reconstruction.
- Synthetic-composition joint SW1, coverage and transition residual.
- Real and synthetic encoder component usage and entropy-effective component
  count, including train/test, interpolation/extrapolation and per-class paired
  errors. Usage counts summarize routing, not latent semantic correctness.

`test_inference.npz` and `train_inference.npz` save the outputs and routing IDs;
`inference.json` and `summary.json` hold the metrics. `final.pt` contains EMA G,
E and prior, the scaler, configuration, and final (non-EMA) D. Restore constructor
settings from the checkpoint configuration. The encoder is used at inference for
the composed/conditional paths only; original prior sampling needs no E.

```bash
.venv/bin/python -u examples/transition_gan.py --config configs/transition/encoder_separate.yaml
.venv/bin/python -u examples/transition_gan.py --config configs/transition/encoder_shared_state.yaml
tail -F results/transition/live.log
```

Completed output folders cannot be reused. For a short smoke check, supply
`--steps 2 --device cpu --out-dir <fresh-directory>`. Full results and interpretation
are recorded in [the leaderboard](../reports/transition/leaderboard/README.md)
and [readout](../reports/transition/leaderboard/READOUT.md).


## Completed round

The shared-state arm leads the original generation benchmark at joint SW1
0.08565, versus 0.09538 with separate critics and 0.10027 for the previous
adversarial-only leader. Coverage favors separate critics (27.7% versus 24.8%).
Real-input next-state errors are 0.01665 and 0.01794 for shared/separate critics;
the synthetic paths have residuals 0.01471 and 0.01322. The full
[readout](../reports/transition/leaderboard/READOUT.md) explains the tradeoffs,
narrow encoder routing, action-response diagnostic and recommended next benchmark.

Reusable verification and frozen action-response diagnostic:

```bash
.venv/bin/python experiments/verify_transition.py results/transition/encoder/encoder_shared_state
.venv/bin/python experiments/audit_transition_actions.py results/transition/encoder/encoder_shared_state --out /tmp/action-audit.json
.venv/bin/python -m unittest tests.test_transition tests.test_trajectory tests.test_transition_encoder
```

The action probe holds state/context fixed and perturbs physical actions. These
inputs leave the training route manifold; the probe is separate from the pinned
prior-generation benchmark. Metrics and leaderboards drove this round's analysis.


## Visual demo and sharing

[Open the standalone toy viewer](../reports/transition/demo/index.html) to compare
real transitions, original sampling, synthetic composition and real-input next-state
prediction. Select either encoder, both classes, all train/test geometries and
five saved times. Detail views use matched axes and show consistency errors as
red connectors. Time slices contain independent transitions, not rollouts.

The page includes the architecture diagram and a copy-ready description with
measured results and limitations. The [ZIP bundle](../reports/transition/demo/transition_demo.zip)
contains that offline HTML plus PNG/PDF toy plots and PNG/SVG/PDF architecture.
The static toy plot discloses its interpolation scene/class; the interactive view
also exposes the extrapolation failures. No new training was required.
