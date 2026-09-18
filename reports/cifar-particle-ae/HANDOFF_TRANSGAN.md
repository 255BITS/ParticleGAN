# AE-GAN handoff — scratch transformer and E-only CNN scouts complete

Branch `feat/cifar-ae-gan-pretrained-encoder`. User requested TransGAN-style G with a subagent implementing it, reusing the historical full-reconstruction CNN benchmark. User explicitly added a fresh E-only CNN because the earlier E-only result was a checkpoint intervention. All three scratch50k runs completed and were re-certified; no continuation is queued. Goal remains FID50k below13. No seed experiments. Keep logs easy to tail and summarize results, leaderboard, explanations and recommendations when complete.

## Results and recommendation

Latest user request: prepare to compact; user suspects pretrained discriminator robustness. No new experiment is authorized by this compaction request. Current task is to preserve the state and discuss hypotheses; do not launch training before the next instruction. Results analysis committed as8e02224; implementation/launch as27e52a1.

Final FID50k: CNN E-only23.1749, transformer full reconstruction24.4911, transformer E-only24.6440, versus historical full-reconstruction CNN18.9012. Transformer full reconstruction improves test MSE16.5% (.03357 versus.04022) for29.2× G parameters and7.6× training time, without improving FID. Training hours: CNN E-only0.61, transformer full4.72, transformer E-only4.54.

E-only CNN curve10/20/30/40/50k:19.4482/20.3610/19.7293/20.0023/23.1749. Transformer full:25.7557/27.2631/24.4107/26.3416/24.4911. Transformer E-only:20.3919/22.9247/25.4700/71.8706/24.6440;40k visibly repeats a few appearances, with diversity recovering by50k. Similar final transformer scores conceal much worse E-only instability. No new observed minimum beats historical50k CNN.

Main inference: direct reconstruction gradients on G/prior are not necessary for the plateau, so L2 competition alone cannot explain it. More G capacity is usable for reconstruction but does not resolve unconditional generation under this recipe. Shared discriminator feedback/regularization and prior/sampling remain suspects; current losses near.693 do not by themselves prove weak gradients. No new gradient probes in this review.

Recommend against long promotion of these endpoints. Proposed next diagnostic (NOT launched): freeze G/prior at an inexpensive CNN10k checkpoint and test whether D can learn real/fake separation under current regularization, using held-out draws and input gradients to assess useful feedback. Await user direction; bothGPUs idle. See `transgan_scout/FINDINGS.md` for caveats, curves and recommendation.

## Discriminator hypothesis for the next session

Distinguish three possibilities: (1) frozen pretrained feature representations have blind spots that G exploits; (2) the trainable heads/pixel branch fail to extract or maintain useful separation; (3) regularization/optimization suppresses useful D feedback even when information is available. These are hypotheses, not established causes. The entire discriminator is NOT frozen: ImageNet ResNet18 features are frozen, but feature heads and a pixel branch train. Therefore a feature blind spot alone need not explain failure of the whole critic.

Evidence: plateau persists across generator families and when reconstruction updates onlyE; previous read-only probes found weak D input and G adversarial gradients in some plateau checkpoints; larger frozen ResNet34 did not help. Caveat: that replacement retained heads trained on different feature coordinates and resumed joint training immediately, so it did not isolate pretrained feature quality. Current D/G loss values nearln2 alone do not prove an ineffective critic or useful GAN equilibrium. The particle prior remains an alternative shared cause.

Proposed staged diagnostic after user resumes work:
1. Restore `runs/cifar_particle_ae/transgan_scout/cnn_e_only/checkpoint_010000.pt` (FID50k19.4482; SHA256 `d75fca4bc42ec09f1423ce1a671b4cbd10caefe0abccae3ac2bdb05d5d93237c`), freezing G/E/prior/EMA. This removes moving-generator and reconstruction-update confounds. Record parent hash and preserve all original states; use standalone diagnostic/trainer to preserve source certificates.
2. First measure held-out real/fake separation and image-gradient norms, attributing score/gradient contributions to pixel versus pretrained-feature branches. Use fresh fake draws and held-out real images, not only D training batches. Keep the FID reference protocol separate from diagnostic validation.
3. Short matched D-only continuations: current bcap versus a weaker setting, changing only regularization. If D learns useful separation when G stops moving, investigate adaptation speed; if weaker regularization is required, investigate overregularization. Separation alone is insufficient: inspect gradients and held-out generalization. Do not silently combine head resets, feature replacement and regularization changes.
4. If separation remains poor, assess what is recoverable from fixed pretrained features versus the pixel branch before choosing a feature/backbone change. Only a promising diagnostic should lead to a short joint checkpoint continuation, then a longer run if FID warrants it.

No seed sweeps, no automatic long promotion, no rerun of the historical full-reconstruction CNN. Do not claim these planned tests have happened. Both GPUs idle; all checkpoints preserved.

## Completed pipeline

- Launched UTC: 2026-09-18T11:52:16.922473+00:00. Pipeline PID **254269**; recheck status rather than assuming active or relaunching.
- Command: `bash experiments/cifar_ae_transgan_pipeline.sh transgan_scout 0,1`.
- TwoGPU workers: transformer all-gradient and transformer E-only start together; CNN E-only queued next.
- Tail: `tail -F runs/cifar_particle_ae/transgan_scout/PIPELINE.log`.
- Manifest: `configs/cifar_particle_ae/transgan_scout/manifest.json`.
- Trainer: `experiments/train_cifar_ae_transgan.py`.
- Runs: `runs/cifar_particle_ae/transgan_scout/{transgan_all,transgan_e_only,cnn_e_only}/`.
- All start from scratch,50k steps, FID50k/test10k reconstruction every10k, checkpoint saved each evaluation.
- Automatic analyzer: `experiments/analyze_cifar_ae_transgan.py`; report `reports/cifar-particle-ae/transgan_scout/LEADERBOARD.md`.
- Both GPUs are idle. Pipeline exited0 and its PID no longer exists. No other long jobs remain active.

## Design and implementation

TransGAN-style G: width512→128→32,8/16/32 token grids, depths5/4/2, four-head global SDPA, MLP ratio4, pre-LayerNorm, learned absolute and relative2D positions, pixel shuffle. Final token LayerNorm and a low-gain tanh RGB head (gain0.1) adapt the unbounded official head. G18,851,023params versus CNN645,123. Official architecture reference is VITA-Group/TransGAN commit6b85440ca56716fd7a60bac964466cc0296ce663. This is a generator-family/capacity comparison inside AE-GAN, not a full TransGAN reproduction.

Config API: `generator_arch` cnn/transgan, `recon_grad` all/encoder_only, `transgan_dim`, `transgan_depths`, `transgan_heads`, `transgan_mlp_ratio`, `transgan_rgb_gain`. CNN stays exactly the original architecture. Alternative G creation preserves shared D/E initialization using a forked RNG stream after advancing the historical G constructor. Both transformer routes share identical initialization. E-only detaches prior means and temporarily freezes G parameters during reconstruction forward; gradients pass through G into E. GAN loss still updates G/prior. G/E/prior/D/EMA/optimizers and all RNG streams checkpoint/resume fully.

Shared recipe: seed24002, batch64, latent64, particles1024, width32 scratchE, frozen ImageNetResNet18 D features and original trainable heads/pixel branch, sigma_rel0.025, temperature0.125, recweight1, LR G/E0.0003, prior0.003, D0.00045, oneD update, EMA0.995, lazy double-backprop bcap every8 with coefficient×8, constantLR. LR and shared objectives were not altered to stabilize the transformer.

## Preflight evidence

Subagent `/root/transgan_impl` implemented trainer/tests and is finished. Primary owns configs/pipeline/analyzer/validation/reports/launch. Ten tests pass (`TESTS.txt`), including both real-CIFAR CUDA full-state8vs4+4 replays, selective routing, unchanged CNN/sharedD/E initialization, upstream Linear initialization and output stability under100× residual activation scaling. Three512step real-size pipeline smokes certified; read-only audit `experiments/validate_cifar_ae_transgan.py` verifies initialization/RNG/optimizer counters/source/checkpoint hashes.

Initial port applied blanket Linear Xavier and saturated100% of live output pixels by128steps. Correcting Linear initialization to official defaults improved initial output but still failed in training; documented preflight failures retained. Final output normalization+gain0.1 resolves immediate saturation in both64step and512step pilots. At512, transformer live/EMA probes have0% pixels|x|>.99 and nontrivial variation. NoFID measured in smokes. This does not guarantee longer-term stability.

Steady preflight speed: transformer all2.92updates/s, E-only3.03, CNN23.53. Estimated50k training time4.75h/4.58h/0.59h, plus evaluations. Peak allocation11.06/10.50/1.55GiB. Whole queue roughly5–6h. First transformer FID aroundonehour afterlaunch. `PREFLIGHT.md`, `VALIDATION.json`, `LAUNCH.json`, `PLAN.md` contain evidence and interpretation.

## Historical comparison and next decision

Historical full-reconstruction CNN scratch trajectory: FID50k19.6110at10k,20.1046at20k,19.4391at30k,19.4241at40k,18.9012at50k. Resumed at30k with full state and unchanged recipe; source certificates and early audit hashes verified. Reference `HISTORICAL_CNN.json`. Do not retrain full-reconstruction CNN (user preference). New CNN E-only provides the fresh architecture comparison; historical CNN all-gradient contrasts are less controlled. Transformer is much larger, so gains cannot be assigned to attention alone. E-only blocks bothG andprior rec updates, so it cannot distinguish their individual effects.

Results review complete: all3 runs re-certified, final sample grids and transformer E-only40k grid inspected. Analyzer report and detailed `transgan_scout/FINDINGS.md` saved. All checkpoint options preserved; no automatic200k promotion.

## Prior work and source safety

Feature/selective-growth scout completed: final70k control20.3672, Gselective22.6733, ResNet34+G27.8205, ResNet3486.5698. Gradient probes suggested unreliable D feedback, with feature coordinate replacement a confound. No previous endpoint promoted. Prior detailed handoff `HANDOFF_FEATURES.md`; earlier history `HANDOFF_GROWTH.md`, `HANDOFF_PLATEAU.md`, `HANDOFF_100k.md`. TwoD longrun stopped by user at172100; best/latest completedFID18.3010at170k. Its200k target was not reached.

Do not edit active trainer or historical/shared `lib/`, `particlegan/`, run_grid.py/config.py: source certificates hash them. New standalone trainer preserves historical certificates. Unrelated `.claude/`, `results/hopfield*`, `results/motion/`, `runs/`, `sparse-ucd.log` are untouched. Use persistent subprocess.Popen(start_new_session=True, stdin=DEVNULL, stdout=log, stderr=STDOUT) for background jobs. Generic follow_grid replays stale log offsets on rerunning existing output directories; this affected superseded smoke logs only. The new scout starts with fresh directories and logs correctly.
