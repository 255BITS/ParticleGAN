# AE-GAN handoff — checkpoint capacity scouts

Branch: `feat/cifar-ae-gan-pretrained-encoder`. User target remains CIFAR-10 generation FID50k below 13. User authorized stopping the two-D continuation and running capacity scouts on both GPUs through the pipeline. They explicitly requested a subagent to implement the change; `/root/capacity_trainer` implemented the standalone trainer, primary agent integrated and tested it.

## Current work

Four scouts run from the **original one-D 50k checkpoint to 70k**, using one D update throughout:

| Arm | G expansion | D head expansion |
|---|---|---|
| control | no | no |
| grow_g | yes | no |
| grow_d | no | yes |
| grow_both | yes | yes |

Pipeline PID at launch: **245618**, GPUs `0,1`, one worker each. Check current logs/processes before action; do not duplicate or restart completed jobs.

- Tail: `tail -F runs/cifar_particle_ae/growth_scout/PIPELINE.log`
- Command: `bash experiments/cifar_ae_growth_pipeline.sh growth_scout 0,1`
- Configs: `configs/cifar_particle_ae/growth_scout/manifest.json`
- Trainer: `experiments/train_cifar_ae_growth.py`
- Automatic final report: `reports/cifar-particle-ae/growth_scout/LEADERBOARD.md`
- Plan, validation and launch records: `growth_scout/{PLAN.md,VALIDATION.json,TESTS.txt,LAUNCH.json}` under this report directory.

FID50k plus held-out reconstruction10k and full checkpoints every 5k (55/60/65/70k). Rank final endpoints and report best intermediate separately. Analyzer reports FID against training time, deltas to contemporaneous control, and factorial interaction. No automatic long promotion. User dislikes paying substantial compute for approximately one FID point.

Parent: `runs/cifar_particle_ae/duration_100k/n08/checkpoint_050000.pt`, SHA256 `10fe8bbc22afb29ff6838ad1ede86e142320e5d7bce43c745b97de24e23ee8d6`, FID50k **18.9012**. All variants retain full old model/EMA/Adam/RNG state and shared seed. Growth initialization uses separate fixed CPU RNG streams without changing training RNG. Reconstruction still updates E/G/prior; exact bcap every8 D updates, coefficient times8. Frozen ResNet18 discriminator backbone unchanged.

## Architecture and validation

G already had GroupNorm, latent affine conditioning and normalized residual sums. Its lack of normalization was not the issue. The new G adds one same-channel residual refinement after each 8/16/32 resolution block. Its final conv starts at zero, and the sum is `x + residual` without dividing by sqrt(2), so initial outputs are unchanged. G parameters increase **645123 → 1091331**.

D adds one 64-channel residual refinement before pooling in each of its three trainable feature heads, also zero-final-conv identity initialized. Adds **222336** trainable parameters. Total D parameters (including frozen features) increase 3583204 → 3805540. This tests feature-head capacity, not a larger pretrained backbone. E and prior are unchanged.

Standalone trainer preserves old state keys. `install_growth` appends parameters to existing optimizer groups after parent restore. Grown-parent resume installs its architecture before strict restoration; removing growth is rejected. EMA new G branches copy live initialization and stay frozen. Audit verifies G/EMA/D outputs and input gradients, finite D double backward, frozen backbone, and identity initialization. Audit locally disables TF32 and selects deterministic cuDNN, restoring backend settings and D context cache afterward. Production training settings are unchanged.

**Five CUDA tests passed**, covering all four real-parent growth states, old tensor/Adam/RNG preservation, fresh new-parameter Adam state, new-layer learning, idempotence, and deterministic8vs4+4 expanded-model full-state replay. Four16-update real50k pipeline smokes certified. Old Adam steps correctly reach50016; new parameter steps16; all four finish with identical training RNG states. Parent hash unchanged. Audit outputs matched exactly, D input derivatives differed at most about7.5e-9 in smokes.

Do not change the active trainer or `lib/`, `particlegan/`, `experiments/run_grid.py`, or `experiments/config.py` while runs depend on their source certificates. Historical and active checkpoints archive source files. New work should preserve these certificates, normally through isolated trainers.

## Stopped two-D result

User explicitly stopped the previous 200k job. Last logged update **172100**; latest and best completed evaluation **FID18.3010 at170k**. It did not complete200k. All saved checkpoints retained, including `runs/cifar_particle_ae/plateau_200k/d2/checkpoint_170000.pt`. Curve and stop record: `plateau_200k/{LEADERBOARD.md,STOPPED.json}`. No two-D process remains active.

The trajectory improved modestly within18–19 but did not approach13. Two D updates gave roughly0.95FID atmatched60k while reducing throughput from~22 to~14.6Gupdates/s. User judged that cost unattractive and wanted checkpoint-based architectural interventions.

## Prior evidence and preferences

Prior six50k→60k scouts: d2 19.0289, unchanged historical19.9770, no-reconstruction-to-prior20.2151, reconstruction-only-E20.2712, reconstructionweight.1 20.5006, LR×.25 20.7400, combined20.8273. Lowering reconstruction or detaching it did not improve short continuations. Critic gradients weakened with training, but objective conflict is not proven as sole cause. Earlier fresh larger-G20k runs lost to small G; this round asks whether growth of a trained G behaves differently.

Previous detailed handoff: `HANDOFF_PLATEAU.md`. Investigation: `plateau/{FINDINGS.md,DIAGNOSIS.md}`. Earlier architecture/lazy-bcap/pretrained-E history: `HANDOFF_100k.md`.

User instructions: no seed experiments, be token efficient, easy-to-tail logs, summarize completed experiments with explanations, leaderboards and recommendations. Keep unrelated `.claude/`, `results/hopfield*`, `results/motion/`, `sparse-ucd.log` and run artifacts untouched. Launch persistent background jobs via `subprocess.Popen(start_new_session=True, stdin=DEVNULL, stdout=log, stderr=STDOUT)`.
