# Previous-action GAN from scratch

The original 50/50 imitation fine-tune retained previous actions:
`E_control(st, at-1, terrain) -> z -> G2 -> at`. It updated E_control/G2 from
pretrained weights with expert action MSE. The later state-only probe model
also achieved 50/50 in a different round. These are different models.

This experiment follows the explicit request to train a **previous-action GAN
from scratch**, not to continue the imitation checkpoint. One encoder serves
both the control and adversarial paths:

```text
prior -> z -> G1 -> st
           -> G2 -> at
           -> G3 -> st+1

E(st, at-1, terrain) -> z -> G1 / G2 / G3
playback: E(st, at-1, terrain) -> z -> G2 -> at -> actual simulator
```

E and every G receive terrain11. G1/G2/G3 remain independent width128 networks.
E uses the previous-command input dimensions of the original control encoder,
hard particle routing with a soft straight-through gradient, and bounded offsets.
No current action or successor is an encoder input. All neural weights and the
MoG prior start fresh; no pretrained parameters or normalization are loaded.

Training retains all 9,297 actual expert transitions from 47 training episodes,
including their action labels. Previous actions are computed within each episode
before shuffling; the first is engines-off `[-1, 0]`. Training uses expert previous
actions; rollout uses the learner's own previous actions. This is full action
supervision, with no sparse observation masks. Statistics are fitted on the
training expert records only; current/successor states share a scaler.

Each update trains the joint D, action D, and a shared state D with a
current/successor role flag. Each role averages comparisons against prior-generated
and control-encoded triples. D sums role losses and applies bcap to each comparison.
G/E/prior minimize the averaged joint adversarial loss plus the mean of the three
marginal roles (weight 1), standardized action MSE (weight 1), G1/G3 reconstruction
(each continuous MSE + contact BCE, weight 1), and one full-table prior regularizer.
All paths remain live throughout; there are no detached probes, paired second
encoder, synthetic cycle, trajectory unroll, reward loss, or simulator training calls.

The default MoG recipe supplies 1,024 particles, z32, relativistic GAN loss, bcap,
neural LR 0.0006, prior LR 100x, cosine decay after 60%, and EMA 0.995. Train one
run for 2,500 updates, batch256, seed24003, GPU1. G and D draw independent batches:
640,000 records each, 1.28 million total training draws. No seed-only repeats.

Checkpoints at 250/1,000/2,500 are selected on 20 new validation worlds
1191000–1191019 by landing rate, mean return, then earlier update. Test the selected
and final checkpoint on the same 50 new worlds 1291000–1291049; identical checkpoints
reuse rollouts. All reset worlds are disjoint from collection and earlier evaluations.

Reevaluate the current joint GAN, current marginal GAN, and legacy joint GAN on
these worlds. Keep the imitation fine-tune as a separate non-GAN reference. It
cannot rank on the GAN-only board or choose the GAN default. Comparisons with these
references change multiple factors: labels, encoder, initialization, objectives,
and record draws. This is a new recipe benchmark, not an isolated previous-action
ablation or GAN-versus-L2 ablation.

Report landing rate with Wilson intervals, mean return, paired landing/return
differences, failures, training cost, and expert prediction metrics against
persistence. G3 has no independently chosen current action input; its predictions
do not establish counterfactual simulator dynamics.

Freeze sources/config/data/reference hashes in the new report's protocol before
the full run. Preserve older protocols and reports. Progress:

```bash
tail -F results/gym/lunar_lander_previous_gan/live.log
```
