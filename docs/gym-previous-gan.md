# Previous-action GAN with expert action L2

This experiment brings the original successful controller's previous-action
input into a single-encoder GAN trained from scratch.

The completed run landed **7/50** fresh test worlds. The original imitation
fine-tune again landed **50/50** on those same worlds; the existing joint GAN
landed 27/50 and remains the validation-selected GAN baseline. This recipe did
not recover imitation performance. See the
[readout](../reports/gym/lunar_lander_previous_gan/READOUT.md) for the comparison
and fixed-input action diagnostics.

```text
prior -> z -> G1 -> st
           -> G2 -> at
           -> G3 -> st+1

E(st, at-1, terrain) -> z -> G1 / G2 / G3
playback: E(st, at-1, terrain) -> z -> G2 -> at -> actual simulator
```

The original [50/50 imitation fine-tune](gym-control.md) used a separate
E_control initialized from the paired world-model encoder. Only E_control and
G2 were fine-tuned with action MSE; G1/G3, E_pair, the prior, and D were frozen.
The [later 50/50 state-only probe model](gym-state-control.md) was a different
scratch model without previous-action inputs. Those results came from different
test worlds and are not a matched architecture comparison.

Here all Gs, E, D, and the prior start fresh. One `E(st, at-1, terrain)` receives
both action MSE and joint/marginal adversarial feedback through all three Gs.
G1/G3 reconstruction losses also shape E/prior. Prior-generated and encoded
triples both receive adversarial losses. There is no second paired encoder,
synthetic cycle, imitation pretraining, or detached probe phase.

Training uses all 9,297 transitions and action labels from 47 heuristic-expert
training episodes. Previous commands are computed within episodes before
shuffling. Training uses expert previous commands; live control uses its own
previous commands, beginning with `[-1, 0]`. E never receives the current action
target or successor. All inputs retain observed terrain context. This experiment
uses complete triples, with no sparse masks.

The default MoG recipe has 1,024 particles, z32, relativistic loss, bcap, bounded
encoder offsets, and EMA. All adversarial, action MSE, and state reconstruction
terms remain active for 2,500 updates on GPU 1. See the
[frozen plan](gym-previous-gan-plan.md) for losses and evaluation rules.

The [GAN-only leaderboard](../reports/gym/lunar_lander_previous_gan/README.md)
compares the new recipe with previous GAN checkpoints on fresh paired worlds.
The imitation fine-tune is reevaluated as a separate non-GAN reference. Training
history, supervision, and losses differ across these references, so this does
not isolate the effect of previous actions or GAN versus MSE.

Reproduction uses a fresh output directory and a new frozen evaluation protocol
if source/configuration changes:

```bash
.venv/bin/python -u experiments/evaluate_gym_previous_gan.py --freeze
.venv/bin/python -u experiments/train_gym_previous_gan.py \
  --config configs/gym/lunar_lander_previous_gan/previous_marginals.yaml
.venv/bin/python -u experiments/evaluate_gym_previous_gan.py \
  --run-dir results/gym/lunar_lander_previous_gan/previous_marginals
.venv/bin/python -u experiments/diagnose_gym_previous_gan.py
tail -F results/gym/lunar_lander_previous_gan/live.log
```

Checkpoints and raw training arrays stay under ignored `results/gym/`. Reports
include evaluation traces, frozen protocol, training source archive, optimizer
configuration, normalization, and metrics. Control runs E/G2/prior only; the
actual simulator supplies the next observed state. G3 cannot evaluate an
independently chosen current action and is not a general counterfactual model.
