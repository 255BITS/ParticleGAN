# Why the original slow policy missed a training landing

The original policy's failure was not simply a new terrain. The saved
`slow_expert.npz` contains 92 successful expert flights from training resets
`24000:24096`. Replaying the [archived original slow checkpoint](../reports/lunar_fast/audit/original_slow.pt)
on those **exact resets** landed 89/92. It flew out of bounds on seeds 24032
and 24053. Seed 24064 received the old classifier's generic `crash` label;
the saved record does not establish whether that contact was a physical crash
or an incomplete landing. The expert had landed on all three. The
[training replay](../reports/lunar_fast/audit/training_replay.json) also shows
the imitation-only checkpoint landed 91/92, including 24053.

At seed 24053 the learned flight starts from the **identical stored initial
state**. The expert's first main command is zero; the published policy sends
`+0.1709`, igniting the upward engine. The first next state already differs
from the expert trajectory by more than 0.01 in observation L2 distance. The
expert landed in 220 steps; the published policy left bounds after 541. The
imitation-only checkpoint also sends a positive first main command (`+0.1356`)
but recovers and lands in 240 steps. Thus the initial ignition error is
observed, while its sole responsibility for the eventual flyaway is **not**
established. After the trajectories diverge, the failed policy visits states
that the successful expert episode did not demonstrate.

The training data include 42 high/rising slow-expert states in which the main
engine is off. On those exact inputs the published policy incorrectly fires
upward on 22/42. Imitation-only does so on 8/42. In a
[fixed-data loss ablation](../reports/lunar_fast/audit/loss_ablation.json),
400 updates of the world successor term alone gave 19/20 validation landings
and 5/42 wrong upward commands; RpGAN alone gave 14/20 and a large action
error. These results implicate the interaction of action calibration, the
learned world gradient, and adversarial drift. They do not isolate a single
loss as the only cause of the 24053 failure.

The physical command has a discontinuity: main `0` is off, while **any**
positive command fires the stock upward engine at at least half power. The old
world model was fit mostly to successful expert actions and badly flattened
the off-to-up transition. On real one-step Box2D branches its predicted mean
vertical-velocity jump was about `0.00468` against `0.02731` measured. A
centered successor-target ablation reduced wrong upward firings to 1/42 at
400 and 1,200 updates, but they returned at 2,400 (30/42), despite 20/20
validation. This showed that changing only the loss target was not a durable
calibration fix.

The [counterfactual collector](../lib/lunar_flight.py) now accepts supplied
training expert episodes generically. It restores each selected state by
replaying its exact seed and full expert-action prefix, checks the state match,
then takes one real Box2D step for each of six main commands: down, off, first
upward ignition, low up, medium up, and full up. It keeps the expert's side
command. The initial audit made 5,472 branches from 152 slow/fast training
episodes; these are **world-model data only**, never policy imitation labels.
The collector never draws from validation or test flights. Together with an
engine-power action representation, this moved predicted ignition jump to
`0.02496` versus `0.02731` actual and reduced jump MAE from `0.01963` for a
raw-action counterfactual world to `0.00644` for the power-feature world. The
[model audit](../reports/lunar_fast/audit/calibrated_world.json) preserves the
fixed-data comparison.

Using the calibrated world and the original actual-successor loss, slow
training at 1,200 RpGAN updates landed 92/92 successful expert training
resets and 20/20 fixed validation resets under the then-current flag rule,
with only 1/42 wrong upward commands in the high/rising slice. A 400-update
fast continuation from that slow policy scored 20/20 validation and won all
20 same-reset paired flights at 1.143× speed. At 2,400 slow updates,
high/rising false upward commands rose to 39/42; one training reset received
a failing old flag score despite 20/20 validation. Its physical contact state
was not audited. This supports the bounded 1,200/400 schedule as an action
calibration choice, but does not establish 2,400-step stability.

The first calibrated full run labeled slow 28/30 and fast 29/30 on the
`94000:94030` cohort. The three apparent misses were **false negatives of
cached leg flags**, not demonstrated physical failures. Slow seeds 94006 and
94029 and fast seed 94020 ended with both legs in active terrain contact.
On fast 94020, `EndContact` cleared leg 0's cached flag at step 169 while
another enabled terrain contact remained. Replaying the frozen slow and fast
checkpoints on those same 30 resets each gave 30/30 by actual active-contact
scoring. The [old report](../reports/lunar_fast/audit/contact_correction/report.json)
is retained; its 28/30 and 29/30 values
remain historical cached-flag scores. The frozen-checkpoint replay is
separate from the fresh full pipeline run with corrected scoring: the frozen
weights scored 30/30 each at 1.222× paired speed. The new run
retrained from the newly classified expert flights using the same
hyperparameters, selected fast round 3, and scored slow 30/30 and fast 30/30
on the fixed 94000 regression cohort. Fast won all 30 same-reset flights at
1.177× paired speed; both controllers had 20/20 validation landings. The
[corrected report](../reports/lunar_fast/report.json) contains those new
weights and measurements. The 94000 cohort was already examined during
diagnosis, so this is a regression result rather than an untouched test.

The original outcome classifier also conflated some sleeping contacts with a
generic crash. The later classifier added `incomplete_landing` and
`off_pad_landing`, but still trusted cached leg flags that can disagree with
active Box2D contacts. The original model-calibration and long-budget
findings remain measured action/physics errors; they do **not** explain the
three apparent 94000 test misses, which arose in scoring. Recipe files once
displayed `ema_decay=0.995`; EMA was never updated or used for any of these
flights. Current metadata explicitly sets the unused value to zero.
