# Slider-error fine-tune for Lunar Lander control

Arm B replaces the imitation controller's action MSE with the
[paired-error critic](gym-slider-gan.md) and leaves every other imitation choice
in place. This is not the completed from-scratch all-heads slider run, and it
does not train joint or marginal transition critics.

The scratch all-heads slider landed **6/50** against **11/50** for matched
previous-action L2. Imitation with action MSE landed **50/50**. Those numbers
are cited from the existing readouts below. This recipe has a CPU smoke test
and no landing evaluation yet.

## Loss graph

```text
E_control(st, at-1, terrain) -> z -> G2 -> at
playback: that command -> actual simulator

error = (scaler.action(at_hat) - y_action) / S
real_R = sigma * epsilon
fake_R = same sigma * epsilon + error

L_R = mean softplus(R(fake_R) - R(real_R)) + recipe critic penalty on those noise coordinates
L_GE = mean softplus(R(real_R) - R(fake_R))
```

`y_action` is the frozen world-model scaler's normalization of the expert
command. `S` is the slider edit scale fit only on the 9,297 training actions:
per-coordinate sample standard deviation of action minus training mean, floored
at 1e-4, then the median-row-RMS gain. `R` is the pinned global-mix critic with
eight width-48 tokens, one four-head attention layer, and score bound 8. Tokens
mix the two action-error coordinates. They are not trajectory steps. Noise
follows the slider geometric schedule with absolute hold 1.0 on this run's
update budget. Real and fake inside a pair share noise. The critic update and
the control update draw independent noise and independent minibatches.

`R` trains with the recipe's critic optimizer and critic penalty
(`recipe.make_critic_optimizer`, `recipe.make_critic_penalty`) on its 2-d noise
coordinates; `E_control` and `G2` use `recipe.make_generator_optimizer`. The
penalty is not applied to lander states, actions, or transition samples.

Action MSE is computed under `no_grad` and written to the log as `action_mse`.
It is not added to `L_GE`. State coordinates of the decoded view receive no
gradient from `L_GE`.

## What was removed, what stays

Removed from the [imitation fine-tune](gym-control.md) optimization graph:

- standardized expert action MSE

Still the imitation fine-tune:

- initialization is `results/gym/lunar_lander/adversarial/best.pt`
- `E_control` starts as a copy of the paired encoder
- trainable modules are `E_control` and `G2`, plus the new critic `R`
- `G1`, `G3`, `E_pair`, the MoG prior, and transition `D` stay frozen
- the original scaler stays frozen
- expert previous commands in the shuffled 9,297-record set; the learner's own
  previous command at rollout, starting from `[-1, 0]`
- MoG Adam learning rate, betas, EMA 0.995, and the cosine schedule
- generator minibatches use seed 24002 with offset 11, the imitation stream
- 2,500 updates, batch 256, checkpoints at 250, 1,000, and 2,500
- no simulator calls, reward loss, or trajectory unroll during training

`R` uses its own Adam at the recipe's discriminator learning rate. Playback
does not run `R`.

## How this differs from the other Lunar Lander arms

| Recipe | Optimized action term | Who trains | Transition GAN |
| --- | --- | --- | --- |
| Imitation L2 | action MSE | `E_control`, `G2` | frozen, unused |
| Joint L2 fine-tune | action MSE, plus reconstruction | all G/E/prior/D and `E_control` | Rp + recipe critic penalty |
| Previous-action L2 | action MSE, plus state MSE/BCE | scratch G/E/prior/D | Rp + recipe critic penalty |
| Scratch sliders, all heads | 18-d paired-error game | scratch G/E/prior/D/R | Rp + recipe critic penalty, still on |
| **This fine-tune** | **2-d action paired-error game** | **`E_control`, `G2`, `R`** | **frozen, unused** |

The scratch `slider_scope: action` path is a different experiment. It still
trains every generator from scratch, keeps joint and marginal GAN losses, and
keeps G1/G3 reconstruction. It was implemented and tested, and it was not the
completed 6/50 run. This fine-tune does not reopen it.

A classic ParticleGAN replacement would drop action MSE and train the joint and
marginal critics with relativistic loss plus the recipe critic penalty. That graph
is not this one. The only adversary here is `R`, and the only penalty is on `R`'s
noise coordinates.

## Run

GPU 1, fresh output directory:

```bash
python -u experiments/train_gym_slider_finetune.py \
  --config configs/gym/lunar_lander_slider_finetune/action_error.yaml
tail -F results/gym/lunar_lander_slider_finetune/live.log
```

Each line is prefixed with `[action_error]`. `metrics.jsonl` records `loss`,
`error_G` (same value as `loss`), `error_D`, `cap`, diagnostic `action_mse`,
and `sigma`. `cap` is the recipe critic penalty on R, applied every update.

CPU smoke overrides the device and writes a new directory:

```bash
python -u experiments/train_gym_slider_finetune.py \
  --config configs/gym/lunar_lander_slider_finetune/action_error.yaml \
  --steps 4 --device cpu \
  --out-dir results/gym/lunar_lander_slider_finetune/smoke_cpu \
  --live-log results/gym/lunar_lander_slider_finetune/smoke_cpu.log
```

The repository checkout does not include the world-model checkpoint or expert
episodes, so the committed check is the unit test on a synthetic checkpoint.
Do not reuse a nonempty `out_dir`.

Landing selection, if a full run is scored later, should use the frozen control
protocol worlds. Test scores must not choose the checkpoint. No seed repeat.
