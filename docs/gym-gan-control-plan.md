# GAN training throughout the three-generator controller

Status: completed on 2026-09-20 after the user requested adversarial training
throughout and a GAN-only leaderboard. Joint GAN landed 34/50, legacy GAN 23/50,
and joint plus marginals 16/50. Both new arms selected update 2,500; the playable
default is gan_joint. See the readout in reports/gym/lunar_lander_gan_control/.
The frozen design below is retained for provenance. This replaced the proposed
correction-label experiment; no correction training or extra sweep was run.

## Graph and eligibility

```text
prior -> z -> G1 -> st
           -> G2 -> at
           -> G3 -> st+1

observed st -> E -> z -> G1 / G2 / G3
playback: observed st -> E -> z -> G2 -> at -> simulator.step(at)
```

Terrain enters E and each G. E receives no action, previous action, successor,
or missingness mask. All Gs, E, and the MoG prior train jointly from scratch.
Adversarial generator and discriminator updates run throughout training, starting
at update 1. There are no detached probes or later imitation-only fine-tunes.
Paired reconstruction losses remain alongside adversarial losses.

The new leaderboard admits only verified GAN-trained controllers. Evaluate the
older pretrained joint adversarial controller as a clearly identified reference;
it has 47 labeled episodes and a different training history, so it is not part of
the matched comparison. Historical non-GAN reports remain intact but their rows
and heuristic scores do not enter this leaderboard or choose its default.
GAN pretraining followed by an imitation-only fine-tune does not qualify for this
leaderboard; adversarial updates must remain active in the controller training.

## First matched comparison

Two runs differ only in discriminator roles:

- **joint:** a masked joint transition discriminator.
- **marginals:** the same joint discriminator, plus an action discriminator and
  one shared discriminator for current/next state marginals.

Both retain the fixed sparse-action dataset: all 9,297 state/successor pairs from
47 expert training episodes, with 1,010 explicit action labels from episode IDs
3,15,53,5,22. The deterministic hash selection, state normalization from all
available states, and action normalization from labeled actions are unchanged.
Hidden actions remain absent from materialized training arrays and all losses.

The joint discriminator sees two observed views, each with fixed terrain/mask
context: complete triplets from the labeled batch; state/successor pairs from
the all-record batch with action coordinates masked to zero. The identical mask
is applied to corresponding fake records. Structural masking inside the critic
also prevents the gradient penalty from using hidden action coordinates.

For each view, fake triplets come both from sampled prior codes and from
E(observed state). Average the two paths and the two joint views. This trains
the generative graph and the encoder-driven graph adversarially. Marginal action
critics use only labeled actions; shared state critics use all state pairs.
G marginal weight is 1 on the mean of action/current/next marginal objectives.
D sums role objectives, with joint views and fake paths averaged.

Both use the default MoG relativistic GAN loss and bcap gradient penalty, MoG
1,024, z32, independent width128 generators, width128 state encoder, bounded
particle offsets, default optimizer/prior regularizer/EMA, and 2,500 updates.
G1/G3 paired state losses and G2 labeled action MSE retain weight 1. Apply the
prior regularizer once per G update. No synthetic cycle reconstruction is added.
Sampled contacts use the existing binary treatment with straight-through
gradients for G updates. Masks and terrain are not gradient-penalty coordinates.

Separate matched streams supply D labeled/all batches, G labeled/all batches,
prior samples, and contact draws. Reconstruction reuses G batches. Additional
critics must not alter shared generator/encoder/prior/joint-critic initialization
or shared draws. D steps detach fake generation; G steps freeze D parameters
while retaining input gradients. Record GAN updates, per-role losses/penalties,
all record draws, parameter counts, time, and sources.

## Evaluation and implementation

Freeze validation 991000–991019 and test 1091000–1091049 after verifying no
overlap with data or previous evaluations. Select checkpoints 250/1,000/2,500
by validation landings, then mean return, then earlier step. Test selected/final,
reusing identical checkpoints. Reevaluate the legacy joint GAN on these same
worlds; do not import historical scores. Preserve termination outcomes, Wilson
intervals, paired returns, traces, inference costs, and G1/G3 diagnostics with
persistence. The same caveat remains: G3 has no alternative-action input.

The GAN-only leaderboard and latest playable default must verify adversarial
training provenance. New checkpoints include trained discriminator weights and
GAN update counts. Keep old viewer options as explicitly labeled historical
controllers; the latest default comes only from this GAN comparison.

Use new source/report directories to preserve frozen previous experiments:
`lib/gym_gan_control.py`, `experiments/*gym_gan_control.py`,
`configs/gym/lunar_lander_gan_control/`,
`results/gym/lunar_lander_gan_control/`, and
`reports/gym/lunar_lander_gan_control/`.

Use GPU 1 only, sequential full training runs, and a flushed
`results/gym/lunar_lander_gan_control/live.log`. Run focused CPU tests and GPU 1
smokes before full training. Tests cover adversarial gradient reach through
E/all Gs/prior, discriminator updates, missing-action masking and nonleakage,
matched initialization/draws, and rejection of non-GAN leaderboard entries.
Freeze the protocol after sources are stable and before full runs. No seed-only
repeats, automatic weight sweeps, new label collection, or unrelated changes.
