# Next experiment: abundant transitions, scarce expert actions

Status: completed after the user resumed on 2026-09-20. See
`reports/gym/lunar_lander_sparse_action/READOUT.md`: sparse probes landed 44/50,
auxiliary 34/50, full-label reference 49/50. Both sparse models selected update
2,500. The original frozen design below is retained for provenance. No additional
training, DAgger, RL, subset search, or weight sweep was launched.

## Question and agreed comparison

Can learning from abundant state/successor pairs help recover a policy when
expert action labels are scarce? The previous experiment provided enough direct
action supervision that the policy could succeed without accurate state decoding.

Keep all state/successor pairs from the same 47 expert training episodes, but
reveal action labels from only **five whole episodes**. Compare two runs from
scratch with identical initialization and data access:

```text
                    +-- G1 -> reconstructed st
st -> E -> z -------+-- G2 -> at
                    +-- G3 -> predicted st+1

actual simulator.step(at) -> observed st+1 -> E -> ...
```

Terrain enters E and each generator. No action, previous action, successor,
episode ID, or label-availability indicator enters E.

- **Probes:** labeled action loss trains E/G2/prior. G1/G3 train on all available
  state/successor pairs using detached z; their gradients cannot reach E/prior.
- **Auxiliary:** same data and losses, but G1/G3 can also train E/prior.

Both have one encoder, MoG1,024, z32, independent width128 generators, bounded
offsets, and a trainable prior with the same regularizer. No discriminator in
this initial comparison. The user explicitly asked whether the previous winner
had a discriminator; it did not. Be clear about this in explanations.

## Practical defaults to freeze before training

These implement the agreed comparison; they are proposed details rather than
additional user requirements. Record the final protocol before full runs.

1. Read actual heuristic behavior episodes from
   `results/gym/lunar_lander/data/episodes.json`, split `train` only. There are
   47 episodes / 9,297 transitions. Do not use alternative branch actions.
2. Select five episode IDs by a fixed hash ordering, independent of rewards,
   lengths, outcomes, validation, and test results. Suggested ordering:
   SHA256 of `lunar-sparse-actions-v1:<episode_id>`, take the first five.
   Save the IDs, source hashes, selected action-record count, and label mask.
   Do not try multiple subsets or pick an easier subset after results.
3. Materialize all current states, successors, and terrain, but make hidden
   expert actions unavailable to the training loss. Store action targets only
   for labeled records. Do not silently retain full-action supervision through
   a reused helper or preprocessing step.
4. Fit shared state normalization from the available training states/successors;
   fit action normalization from the five labeled episodes only. Share these
   statistics exactly between arms. This avoids using the hidden actions even
   to estimate their marginal mean/scale. No pretrained weights are loaded.
5. Use two matched RNG streams: batch256 sampled from labeled action records,
   and batch256 sampled from all state/successor records. Each optimizer update
   uses one batch of each. Both arms use identical corresponding draws.
   Normalize action MSE over labeled examples, not all examples with missing
   labels set to zero; otherwise label sparsity also changes the loss weight.
6. Retain 2,500 updates, checkpoints250/1,000/2,500, seed24003, default MoG
   optimizer/prior regularizer/EMA conventions, and auxiliary weights1 each.
   Keeping weights fixed isolates action-label scarcity. This is a changed
   supervision experiment, not a training run differing only in seed.
7. Losses: labeled standardized action MSE; current/successor continuous MSE
   plus contact BCE on all records; common prior regularizer once per update.
   In probes, only auxiliary-head inputs are detached. G2 is supervised only on
   labeled records. Do not mask away state/successor training on unlabeled rows.
8. Count unique labeled episodes/transitions, labeled draws, auxiliary draws,
   trainable/inference parameters, GPU time, and simulator calls separately.
   Training remains individual transitions, with no simulator calls or unrolling.

State/successor observations may themselves reveal information about the action.
That indirect supervision is the point of this experiment. Claim a reduction in
**explicit action labels**, not absence of action information or a free source
of transitions. The larger transition dataset still comes from expert behavior.

## Evaluation and interpretation

Use fresh paired validation20/test50 worlds, disjoint from original collection
and both prior control rounds. Suggested ranges791000–791019 and891000–891049;
verify provenance before freezing. These are shared evaluation worlds, not
training-seed repetitions.

Select each new arm by validation landing rate, then mean return, then earlier
checkpoint in an exact tie. Test selected/final only; reuse identical results.
Retain explicit terminal outcomes, Wilson intervals, paired per-episode returns,
action traces, inference costs, engine usage, and source/checkpoint hashes.

Re-evaluate the existing **full-label state-only probe** controller and heuristic
on the same fresh worlds as references. Historical50/50 results are not new
paired scores. The full-label reference differs in available labels and scaler;
the two new arms are the controlled comparison. It is not necessary to retrain
the full-label reference or every old controller in this initial round.

Score G1/G3 on held-out expert transitions and actual learner rollouts, with
persistence on the same records. Cross-score both new models on both fixed
rollout datasets if useful. G3 has no alternative-action input and is trained
on expert successors, so it is not a general counterfactual dynamics model.
Action agreement on withheld training labels may be reported only as a clearly
posthoc diagnostic; those labels must not enter training or checkpoint selection.

Primary question: does auxiliary learning improve landing under five labeled
episodes? Report negative results just as clearly. Do not expand the label
fractions, change weights, or repeat subsets automatically to obtain a win.

## Implementation and verification

Existing sources and reports are hashed. Preserve them; use new modules/scripts
and new experiment directories rather than mutating frozen trainers/evaluators.
Suggested paths:

- `experiments/train_gym_sparse_action.py`
- `experiments/evaluate_gym_sparse_action.py`
- `configs/gym/lunar_lander_sparse_action/`
- `results/gym/lunar_lander_sparse_action/`
- `reports/gym/lunar_lander_sparse_action/`
- Stable flushed log: `results/gym/lunar_lander_sparse_action/live.log`

Reuse the state-only model, actual simulator evaluation helpers, and viewer
interfaces where possible. Meaningful tests should prove exact gradient scopes,
matched initialization/draws, whole-episode label selection, and no hidden-action
leakage. Perturbing hidden action values must not change preprocessing, batches,
losses, or model updates. Check that action loss normalization is independent of
the number of unlabeled records. Run focused CPU tests and GPU1 correctness
smokes, then the two full runs sequentially. Broaden tests once as appropriate.

Add both new controllers to the playable viewer, retain existing options, and
select its default using comparable fresh validation results. Preserve the actual
simulator loop and Play/Pause/Step/Reset. Finish with a leaderboard, explanation,
costs, limitations, and a metrics-grounded next recommendation.

## Workspace and completed work

- Repo `/home/martyn/dev/ParticleGAN`; branch `feature/gym-world-model`.
- Use `.venv/bin/python`. **GPU1 only; GPU0 belongs to the user.**
- User previously explicitly requested subagents for implementation. Delegate
  bounded independent work when implementing this next round.
- AGENTS.md: no seed-only repeats, token efficient, tail-able logs, metrics and
  leaderboards, explanations/recommendations after completion.
- Lunar work is uncommitted. Preserve existing and unrelated changes; no commit,
  PR, merge, or deployment requested. No active goal has been created.
- Read `docs/gym-state-control.md` and
  `reports/gym/lunar_lander_state_control/READOUT.md` for the completed round.

Completed full-label state-only results on test691000–691049:

| Controller | Landings | Mean return |
| --- | ---: | ---: |
| Prior pretrained imitation | 50/50 | 283.16 |
| Heuristic | 50/50 | 282.45 |
| Scratch state-only probes | 50/50 | 282.24 |
| Scratch joint auxiliary | 27/50 | 130.92 |

Both new runs selected update2,500; training22.61s probes /20.73s auxiliary.
Auxiliary improved expert G3 continuous MSE .28094→.02341 (~12x), but expert
action MSE worsened .03825→.05735. Cross-scoring showed auxiliary decoded states
better on both fixed rollout datasets while controlling worse. Persistence beat
both G3 models on expert next-state MSE (.017384). No universal reliability or
pretraining-benefit claim follows from these results.

Relevant code:
`lib/gym_state_control.py`, `experiments/train_gym_state_control.py`,
`experiments/evaluate_gym_state_control.py`,
`experiments/diagnose_gym_state_cross.py`, `experiments/plot_gym_state_control.py`.
Full-label checkpoints:
`results/gym/lunar_lander_state_control/{probes,auxiliary}/best.pt`.
Prior imitation checkpoint:
`results/gym/lunar_lander_control/imitation/best.pt`.

Full suite332passed +27subtests,4opt-in skips. Both3-stepGPU1 smokes passed.
Browser verified all six controller modes, reset identity, Play/Pause/Step,
landing/terminal stop, and no JavaScript exceptions. Old/new evaluation protocols
still verify. New round has58 report artifacts plus its artifact manifest.

Current live server: http://localhost:8787, CPU, exec session34449 launched with
`examples/gym_lander_live.py --seed 591000`. Default `state_probes`; six options:
expert, original, imitation, joint, state_probes, state_auxiliary. The user is
actively interacting, so do not assume the saved state/seed is still current.
Verify ownership before restarting and leave services8765/8766 untouched.
