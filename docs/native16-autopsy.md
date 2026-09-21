# Why the native RpGAN toy passed and Lunar landings did not

PR #16 merged as `3328fa5` ("Pin the particle controller with a live RpGAN
pair"). Its CPU gate passed. The same recipe, trained 2500 steps on cuda:1
with the same adversarial world model and expert episodes as the earlier
particle run, did not land.

These Lunar numbers are the post-merge measurement. This checkout did not
rerun them.

| Controller | What was scored | Result |
| --- | --- | --- |
| PR #16 2D gate, fixed live `(record, z)` arm | EMA action MSE | 0.1396, pass (≤ 0.18) |
| PR #16 2D gate, observation-critic arm | EMA action MSE | 2.1065, collapse (≥ 1) |
| Post-merge particle finetune | `diag_action_mse` during training | stayed about 0.07–0.18 |
| Post-merge particle finetune | shared-protocol landings | val 1/20, mean return about −84.6, selected step 2500; test 4/50, mean return about −66.8 |
| Earlier collapsed particle run, same protocol | test landings | 2/50 |
| L2 imitation, same protocol | test landings | 50/50, mean return about 286.8 |

The proxy moved from the collapse regime into the gate's pass band. Landings
stayed next to the collapsed run. L2, which optimizes the action, still lands.

## Answers

1. **The 2D toy measures the wrong proxy.** Expert action is
   `tanh(-2.2 * previous)`. State is in the record and does not set the
   action. There is no dynamics and no landing. The pass rule is EMA action
   MSE ≤ 0.18. On that plant a zero action has teacher-forced MSE 0.5513 and
   closed-loop MSE 0.0689, because the target is a function of the previous
   command and a zero command makes the next target zero. Closing the loop on
   the same target does not create a landing test.

2. **The gym trainer ran the fixed arm, not the collapsed one.** Trainable
   scope is `E_control`, G2, and a fresh `D(record, z)`. G1, G3, `E_pair`, the
   MoG table, and the observation critics stay frozen. The generator step is
   Rp logistic with adversarial weight 1 and L2 weight 0. The prior term keeps
   the control code on the live real pair; the control term detaches the real
   score. Sample-point `b_cap` (coeff 1) is on `(record, z)`. That is the
   pairing `experiments/toy_particle_native_2d.py` calls `fixed`. The
   observation-critic arm is what the toy collapses, and it is not the loss
   that stepped the post-merge run. The log moving from above 1 into 0.07–0.18
   is what that fixed proxy does. Terrain context, the 18-D record, and frozen
   G1/G3 are real differences. They are not a silent return to the observation
   game. Checkpoint choice was validation landings, and the best checkpoint
   was still 1/20, so EMA-versus-live selection did not hide a landing policy.

3. **`diag_action_mse` is expert-previous teacher forcing.** Each logged row
   decodes `E_control(state, expert previous)` on a shuffled training minibatch
   and compares G2 to the expert action, under `no_grad`, in standardized
   coordinates, before the optimizer step. Playback starts from previous
   command `[-1, 0]` and then feeds the learner's own action. The trainer
   records that split in provenance and never calls the simulator
   (`simulator_calls=0`). The L2 controller that lands 50/50 was scored the
   same way on expert commands (held-out standardized action MSE 0.0186,
   physical 0.0068, validation records in
   `reports/gym/lunar_lander_control/selections/imitation.json`, whose note
   says rollout uses learner commands). A small expert-previous error can
   belong to a landing policy. It can also belong to a policy that only
   matches the expert's own previous commands. The diagnostic cannot tell
   those apart.

4. **The honest gate is a state pad with the learner's previous command.**
   On that plant the #16 fixed recipe finishes at teacher-forced MSE 0.0151
   (old gate PASS) and landing rate 0.326 (honest gate FAIL). No Lunar claim
   follows from that rate.

## What the gym code actually scores

`build_expert_records` sets previous command to `[-1, 0]` and then to the
expert action shifted by one step. Training shuffles those rows.
`control_decode` reads state and that previous command, never the current
expert action. `diagnostic_l2` compares the decoded action slice with the
expert action on that same row. `evaluate_controller` replaces previous with
the action just sent to the simulator.

The imitation readout already names this split, and that controller lands.
The particle update never leaves the expert-previous rows. Relativistic
`D(record, z)` can look consistent on those rows, which is the 2D pass and
the healthy `diag_action_mse`, without a policy that stabilizes a state the
learner drove off the expert trajectory.

`examples/five_modes.py` inverts the generator because the encoder reads `x`
and sits on the live real side of one RpGAN term. `E_control` does not read
the expert action or the successor. Copying `E_pair` into `E_control` puts
the previous command in the slot `E_pair` used for the current action. On a
long expert episode that slot is almost the current action once the craft has
settled, so the copied encoder already looks accurate on a shuffled pool.

## CPU probes

`python -u experiments/toy_native16_autopsy.py` writes
`results/gym/native16_autopsy/live.log` (line buffered). Tail it with
`tail -F results/gym/native16_autopsy/live.log`. One CPU run, about 6 seconds,
torch 2.14. Numbers below are that run
(`reports/gym/native16_autopsy/summary.json`). `tests/test_native16_autopsy.py`
locks the analytic false-pass, not the trained floats.

### Previous-only plant (the PR #16 target)

| Policy | Teacher-forced MSE | Closed-loop MSE | Old gate (≤ 0.18) |
| --- | ---: | ---: | --- |
| Zero action | 0.5513 | 0.0689 | FAIL on the logged metric, but the closed loop looks healthier |
| Exact `tanh(-2.2 * previous)` | 0 | 0 | PASS |

### State pad

State updates as `s <- clip(s + 0.5 a, -1.5, 1.5)`. The expert action is
`tanh(-2.2 * s)`. Previous command starts at 0, then becomes the policy's own
action. Landing means `|s| < 0.08` after 24 steps, from 257 starts on a grid
in `[-1, 1]`. Teacher-forced MSE uses expert episodes of 32 steps (previous
command is the expert's). `early_tf` is the first 4 steps of those episodes.
The old gate is full-pool teacher-forced MSE ≤ 0.18. The honest gate is
landing rate ≥ 0.80.

Analytic policies, no training:

| Policy | Full-pool TF | Early TF | On-policy MSE | Landings | Old | Honest |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Expert | 0 | 0 | 0 | 1.000 | PASS | PASS |
| Expert action + 0.4 | 0.1600 | 0.1600 | 0.1600 | 0.000 | PASS | FAIL |
| Copy previous command | 0.0273 | 0.2185 | 0.5580 | 0.082 | PASS | FAIL |
| Expert action + previous | 0.0225 | 0.1800 | 0.5717 | 0.214 | PASS | FAIL |

The +0.4 bias sits in the same 0.07–0.18 band as the post-merge log and never
reaches the pad (final `|s|` stays near 0.19). Copying the previous command
is a pass on the shuffled pool and a fail on the transient and on the pad.
That is the expert-previous diagnostic: settled steps, where previous and
current expert actions match, dominate the average.

Trained arms use the same pretrain as the 2D gate (paired reconstruction,
then `E_control` copied from that encoder) and the same 250/400 step budget.
The fixed arm is the PR #16 generator step: Rp logistic, `b_cap` coeff 1,
adversarial weight 1, L2 weight 0, live `(record, z)` pair, only the control
encoder and the action head. `action_pair` is that loss with the state
channel removed from the critic. `l2_probe` minimizes action MSE on the same
pairs. It is a solvability check for this plant. It is not a Lunar config,
and the Lunar trainer still rejects a nonzero L2 weight.

| Arm | Step 400 TF | Early TF | On-policy MSE | Landings | Old | Honest |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| fixed, adv weight 1 | 0.0151 | 0.1218 | 0.3099 | 0.326 | PASS | FAIL |
| action pair, adv weight 1 | 0.0185 | 0.1259 | 0.2452 | 0.574 | PASS | FAIL |
| L2 probe, this plant only | 0.0082 | 0.0663 | 0.0740 | 0.868 | PASS | PASS |

At step 1 the fixed arm is already an old-gate pass: teacher-forced MSE
0.0123, early 0.1066, on-policy 0.5167, landings 0.132. Four hundred RpGAN
steps leave the shuffled error in that band and do not cross the pad. The
low training diagnostic is mostly the initialization on expert previous
commands, which is what `diag_action_mse` measures.

Restricting the critic to `(action, z)` raises landings from 0.326 to 0.574
and still fails 0.80. The full-record critic is not the whole miss. The L2
probe on the same batches does pass, so the pad is solvable when the action
itself is the loss. That is why L2 lands the craft and this adversary does
not, and it is not a reason to set the Lunar adversarial weight to 0.

## Recommendation

Keep adversarial weight 1. Do not treat EMA action MSE ≤ 0.18, or a training
`diag_action_mse` in 0.07–0.18, as evidence the controller will land. The
collapsed run and the post-merge run already showed that proxy moving while
test landings stayed at 2/50 and 4/50.

Before another 2500-step Lunar finetune, require the state-pad honest gate:
learner previous command, landing rate ≥ 0.80, adversarial weight 1. The
merged fixed recipe scores 0.326 there. Scoring only the action slice inside
the same teacher-forced RpGAN scores 0.574 and is not enough to take to the
simulator.

The training change to try on that gate, and not on Lunar first, is a batch
whose previous command is the controller's own action along a short rollout
of this plant, with the loss still the live RpGAN pair. This autopsy did not
run that change. If it cannot pass the pad, it will not explain a Lunar
landing either.
