# AE-GAN handoff — pretrained features and selective reconstruction scouts

Branch `feat/cifar-ae-gan-pretrained-encoder`. User accepted the next experiments after capacity growth failed to improve final FID. Both GPUs are now assigned to the **features_scout** pipeline. Target remains CIFAR-10 generation FID50k below13. No seed experiments; easy-to-tail logs; summarize completed runs with leaderboard, interpretation, cost and recommendation. No automatic long promotion.

## Active scouts

- Pipeline PID at launch: 247178. Recheck processes/logs before action; do not duplicate work.
- Tail: `tail -F runs/cifar_particle_ae/features_scout/PIPELINE.log`
- Command: `bash experiments/cifar_ae_features_pipeline.sh features_scout 0,1`
- Manifest: `configs/cifar_particle_ae/features_scout/manifest.json`
- Trainer: `experiments/train_cifar_ae_features.py`
- Automatic completion report: `reports/cifar-particle-ae/features_scout/LEADERBOARD.md`
- Detailed plan and evidence: `features_scout/{PLAN.md,VALIDATION.json,TESTS.txt,LAUNCH.json}`.

All four run from original one-D50k to70k, with FID50k/reconstruction10k/checkpoints every5k. Same full original parent state and seed, one D update throughout, lazy bcapN8×8, same rates and reconstruction weight.

| Arm | Pretrained D | Added G branches |
|---|---|---|
| control | original ResNet18 | none |
| resnet34 | frozen ResNet34 replacement | none |
| grow_g_adv | original ResNet18 | reconstruction gradients blocked only on new parameters |
| both | frozen ResNet34 replacement | same selective new-G routing |

Parent: `runs/cifar_particle_ae/duration_100k/n08/checkpoint_050000.pt`, SHA256 `10fe8bbc22afb29ff6838ad1ede86e142320e5d7bce43c745b97de24e23ee8d6`, FID18.9012.

ResNet34 replacement preserves trained D pixel branch, heads and Adam state, but changes pretrained feature coordinates and thus D scores immediately. Fixed initial probe maximum score change~0.22477. No head reset, extra D warmup or G freezing: this tests replacement plus joint adaptation, not an isolated function-preserving capacity increase. Only frozen backbone parameters change; D total parameters3583204→8970724. Feature channels and spatial shapes remain compatible, pretrained weights cached. Runtime backbone audit logs before/after weights and G/EMA preservation.

G uses the previously tested identity refinements, increasing645123→1091331 parameters. Temporarily freezing only `g.growth` parameters during the reconstruction forward removes their reconstruction gradients; branch activations remain differentiable into old G, E and prior. The already-built adversarial graph still updates new parameters. Reconstruction evaluation uses the same full generator. No other gradient routing changed. New G parameters alone get fresh Adam state; full old model/EMA/Adam/RNG state restored.

## Validation and implementation

Four tests passed: two selective-gradient cases (allowed/blocked), real-parent CUDA pretrained swap preserving G/EMA, learned D/Adam state and RNG, and deterministic8vs4+4 combined-model full-state resume. ResNet34 passes double backward and stays frozen through train/requires_grad toggles. Four real-parent16-step pipeline smokes certified, old Adam50016/new G Adam16, all end with identical training RNG states; parent SHA unchanged. All tests done before scout launch.

Standalone trainer copied growth trainer to preserve historical source certificates; adds `d_backbone` and `recon_growth_grad` flags. Parent backbone is installed before strict restore; requested replacement follows restoration. Metadata and end-of-run frozen-feature checks use the installed backbone. Selective routing flags are journaled interventions. Grown and replaced-backbone checkpoints support future full-state continuation.

Do not modify active/historical trainers, `lib/`, `particlegan/`, `experiments/run_grid.py` or `experiments/config.py` while their source certificates are needed. New isolated trainers preserve them. New pipeline/analyzer produce final rankings, intermediate curves, compute costs, control deltas and factorial interaction; no automatic long continuation.

## Prior completed results

Growth scouts50k→70k: control19.2033, expanded D heads19.2402 (+11%train time), both20.1437 (+28%), larger G22.8981 (+14%). G reconstruction improved5.4% while FID worsened3.69. D-only best intermediate18.5407at65k rebounded by70k. None promoted. Detailed results: `growth_scout/{LEADERBOARD.md,FINDINGS.md}`. Current selective-G arm comparison to old all-gradient G is historical context, not a contemporaneous paired control.

Two-D continuation explicitly stopped by user at logged172100; best/latest completed FID18.3010at170k. It did not complete200k. Checkpoints retained. `plateau_200k/{LEADERBOARD.md,STOPPED.json}`.

Prior reconstruction-only-E/no-prior/low-weight/LR scouts failed to improve short continuations. Weak D gradients implicated feedback quality, but reconstruction conflict is not a proven sole cause. Earlier fresh larger-G/pretrained-E scouts also lost to baseline. BigGAN target comparison has protocol caveats; ours unconditional with pretrained D.

History: `HANDOFF_GROWTH.md`, `HANDOFF_PLATEAU.md`, `HANDOFF_100k.md`; investigation `plateau/{FINDINGS.md,DIAGNOSIS.md}`. Unrelated `.claude/`, `results/hopfield*`, `results/motion/`, `sparse-ucd.log` and runs untouched. Use persistent subprocess.Popen with start_new_session=True and redirected streams for background jobs.
