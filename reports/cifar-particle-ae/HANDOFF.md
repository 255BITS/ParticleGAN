# AE-GAN handoff — scratch transformer and E-only CNN scouts running

Branch `feat/cifar-ae-gan-pretrained-encoder`. User requested TransGAN-style G with a subagent implementing it, reusing the historical full-reconstruction CNN benchmark. User explicitly added a fresh E-only CNN because the earlier E-only result was a checkpoint intervention. Three scratch50k runs are launched; no automatic long promotion. Goal remains FID50k below13. No seed experiments. Keep logs easy to tail and summarize results, leaderboard, explanations and recommendations when complete.

## Active pipeline

- Launched UTC: 2026-09-18T11:52:16.922473+00:00. Pipeline PID **254269**; recheck status rather than assuming active or relaunching.
- Command: `bash experiments/cifar_ae_transgan_pipeline.sh transgan_scout 0,1`.
- TwoGPU workers: transformer all-gradient and transformer E-only start together; CNN E-only queued next.
- Tail: `tail -F runs/cifar_particle_ae/transgan_scout/PIPELINE.log`.
- Manifest: `configs/cifar_particle_ae/transgan_scout/manifest.json`.
- Trainer: `experiments/train_cifar_ae_transgan.py`.
- Runs: `runs/cifar_particle_ae/transgan_scout/{transgan_all,transgan_e_only,cnn_e_only}/`.
- All start from scratch,50k steps, FID50k/test10k reconstruction every10k, checkpoint saved each evaluation.
- Automatic analyzer: `experiments/analyze_cifar_ae_transgan.py`; report `reports/cifar-particle-ae/transgan_scout/LEADERBOARD.md`.
- BothGPUs available to this task, but currently occupied. No other long jobs remain active.

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

On completion, re-certify all3 runs, inspect final curves and samples, report finalFID ranking/cost and historical contrasts. Analyzer runs automatically after pipeline; partial reports deliberately refuse winner selection. Preserve checkpoint options; no automatic200k promotion.

## Prior work and source safety

Feature/selective-growth scout completed: final70k control20.3672, Gselective22.6733, ResNet34+G27.8205, ResNet3486.5698. Gradient probes suggested unreliable D feedback, with feature coordinate replacement a confound. No previous endpoint promoted. Prior detailed handoff `HANDOFF_FEATURES.md`; earlier history `HANDOFF_GROWTH.md`, `HANDOFF_PLATEAU.md`, `HANDOFF_100k.md`. TwoD longrun stopped by user at172100; best/latest completedFID18.3010at170k. Its200k target was not reached.

Do not edit active trainer or historical/shared `lib/`, `particlegan/`, run_grid.py/config.py: source certificates hash them. New standalone trainer preserves historical certificates. Unrelated `.claude/`, `results/hopfield*`, `results/motion/`, `runs/`, `sparse-ucd.log` are untouched. Use persistent subprocess.Popen(start_new_session=True, stdin=DEVNULL, stdout=log, stderr=STDOUT) for background jobs. Generic follow_grid replays stale log offsets on rerunning existing output directories; this affected superseded smoke logs only. The new scout starts with fresh directories and logs correctly.
