# Slider-error fine-tune: recipe only, no new landings

Arm B replaces imitation action MSE with the Anima paired-error critic on the
two action coordinates. `E_control` and `G2` still fine-tune from the
adversarial world-model checkpoint. `G1`, `G3`, `E_pair`, the prior, and the
transition discriminators stay frozen. No joint or marginal loss is optimized.
The error critic's lazy gradient cap is applied to its noise coordinates, not
to transition samples.

No full training run and no rollout evaluation were done here. The table below
repeats published readout numbers so this arm has a baseline. Do not read any
row as a score for this fine-tune.

## Published baselines

Worlds differ across reports. Each row names the readout it came from.

| Controller | Landings | Mean return | Source |
| --- | ---: | ---: | --- |
| Imitation L2 fine-tune | 50/50 | 286.76 | [control readout](../lunar_lander_control/READOUT.md) |
| Joint L2 fine-tune | 12/50 | 74.87 | same control worlds |
| Original prototype | 0/50 | -374.53 | same control worlds |
| Previous-action L2, reselected | 11/50 | 13.95 | [slider readout](../lunar_lander_slider_gan/READOUT.md) |
| Scratch sliders, all heads | 6/50 | -26.40 | same slider worlds |
| State-only joint GAN | 37/50 | 186.88 | same slider worlds |
| Imitation reference on slider worlds | 50/50 | 278.56 | [slider leaderboard](../lunar_lander_slider_gan/README.md) |
| Scratch previous-action L2 | 7/50 | -4.12 | [previous-action readout](../lunar_lander_previous_gan/READOUT.md); its own worlds |

This fine-tune has no landing count.

## What the CPU check showed

`tests/test_gym_slider_finetune.py` trains four updates on a synthetic
world-model checkpoint. The generator objective matches the paired-error loss
alone: its gradient on the decoded transition is confined to the two action
columns, and the logged action MSE does not require grad. After training, only
`E_control` and `G2` move among the world-model modules. Transition `D` is not
stepped (`discriminator_optimizer_record_draws` is 0). The noise-coordinate cap
is zero on updates that are not multiples of four. Reloading `final.pt` repeats
the control action.

## Recommendation

Run one 2,500-update training on GPU 1, then score checkpoints 250, 1,000, and
2,500 on the existing control validation worlds before touching test worlds.
The question is whether the paired-error critic can replace action MSE when the
controller is already initialized from the world model that imitation tunes to
50/50. The scratch all-heads slider never had that initialization, and it lost
to L2 (6/50 versus 11/50), so it does not answer this question.

Stop if validation landings stay far below the imitation curve (3/20, 9/20,
20/20 at those same checkpoints). Do not add joint or marginal critics, do not
swap in sample-point `b_cap` on transitions, and do not repeat the run with
another seed. `R` starts random while `G2` starts useful, so early updates can
drag the action head before the critic is informative. That is part of this
recipe, not a reason to put MSE back into the graph.
