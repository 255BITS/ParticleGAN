# Next discussion: persistent process state without full-rollout training

User requested commit/push and compaction, and asked whether to add G memory.
These are brainstorming candidates, not selected/queued experiments. Continue
discussion after compaction. Keep the new2k proposal_mixed_pair25 as baseline.

## Small G-owned recurrent state

Let G maintain a small state S alongside D-owned M. G reads both; S can retain
its own trajectory context while M supplies D's interpretation of recent samples.
This could reduce reliance on a representation that D continually changes, but
S can also drift or settle into a wrong attractor. More memory alone is no guarantee.

The training-state construction must be explicit: build S over a real prefix,
then train the existing bounded local generated continuation. Its update should
consume the actual observation during prefix construction and the generated
sample at runtime. There are no full generated training rollouts. G's state
update is trained by GAN gradients, D still owns its writer, and z stays fixed
per trajectory. A cold start initializes both states without expert observations.
This is extra real-prefix recurrent computation; it is not free or already coded.

Compare S sizes8/16 and a matched model without G access to M. Measure both
dependence on M (interventions) and improvement over that separately trained
control. Also test the new radius/speed/direction history responses, primary
cold/warm circle passes, continuous Q, phase errors and coverage. Do not reward
a model for merely copying a different arbitrary orbit for each history.

Historical evidence: reports/autonomous-memory/scout/README.md. The earlier
private-GRU model had55.5% cold full-256 passes at2k versus52.3% for a separately
trained no-M-reading control. The with-M model was sensitive to removing M;
it did NOT simply ignore M. Its5k result reached85.2% but all passing directions
were CCW, and there was no matched5k no-reading control. Those experiments used
the older generated-rollout training and do not establish local-only success.

## Persistent anchor versus evolving cursor

A less flexible alternative is a small persistent code for the process plus a
dynamic cursor. For a warm start, initialize the anchor from observed history;
for a cold start, from the fixed particle. Let current D memory and the clock
inform progression/correction without overwriting the anchor every step.
Learn the code with adversarial objectives, without radius/center/velocity labels
or MSE training targets. This offers a generic bias for retaining stable context,
but can preserve a mistaken initialization or be ignored by G. Cold and warm
initialization/training must both be specified before implementing it.

## Structured recurrent dynamics

A learned rotational or norm-preserving subspace could provide an evolving cursor
whose motion does not decay to a fixed point. Condition dynamics on persistent
context rather than imposing one universal clock period. This is an explicit
inductive bias for periodic processes, not evidence of a general sequential
solution. Avoid analytically specifying the circle's center/radius in training.
Previous slow/fast D-memory scouts already failed; repeating rate/size sweeps
alone is not a new hypothesis.

## Locate the current failure first

The process interventions show near-zero median effect of expert radius/speed on
late outputs. They do not say whether M has lost the information or G cannot use
it. Evaluation-only restoration of a real-history state at late times, together
with measures of process information recoverable from M, could separate reader,
writer, and clock-distribution failures. Teacher information would be diagnostic
only, never part of the deployed loop or a full generated training objective.

Preferred discussion order: identify where process information stops affecting
the loop, then compare small G recurrence against the persistent-anchor variant.
Keep default exact B-cap, no clipping/EMA, no MSE training objectives, no seed
sweeps, configuration-based variants, both-GPU queue and completed-only selection.
