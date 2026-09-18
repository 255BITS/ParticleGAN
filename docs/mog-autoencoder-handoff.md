# Particle autoencoder continuation handoff

> Historical experiment/planning handoff. The 0.5.0 release API and final naming
> are documented in [the particle autoencoder guide](particle-autoencoders.md).
> `vae_gan` is the constant-KL default; reconstruction never adds KL.

## Current direction: prepare PR, documentation, and release after compaction

The user has enough experimental evidence and selected **particle AE-GAN,
particle VAE-GAN, and AE-DDGAN** for a PR. After compaction, document these,
prepare the package/API and PR, then bump a release. **Do not start further
experiments by default**; this supersedes older next-experiment recommendations.

Start with [the PR/release handoff](particle-autoencoders-pr-handoff.md). It
records exact definitions, evidence, package exposure gap, outstanding checks,
current uncommitted work, and release workflow. New implementation currently
lives outside the shipped `particlegan` package, so a release-ready public API
still needs attention. Local version is0.4.0; no next version has been selected.
No jobs remain running/queued. No commit, PR, push, tag, version bump, or release
was made in this preparation turn. Main tracked source remains unchanged.

## Latest: lower-LR controls plus constant-KL particle VAE completed (2026-09-17)

User accepted the lower-LR round and asked if we can be variational without a KL
regularizer. Answer: yes for a restricted family with constant KL, not by simply
removing variable KL from our soft categorical posterior. Completed five200-step
pilots and five6000-step full runs on both GPUs. Nothing remains running/queued.
No seed sweep, no commit requested; new and previous work remains uncommitted.

- Read [results and formulation](../reports/mog-vae/stability/README.md),
  [leaderboard](../reports/mog-vae/stability/LEADERBOARD.md),
  [protocol](../reports/mog-vae/stability/PROTOCOL.md).
- New hard posterior: `E(X)->one k; z=p[k]+fixed_sigma*eps`. One-hot q(k|X),
  uniform p(k), q(z|k,X)=p(z|k). JOINT KL(k,z)=log(K), independent of parameters.
  Drop constant from optimization, retain in ELBO. No local learned offset/var.
  Related to VQ-VAE's established constantKL argument (paper linked in protocol).
  This is not a claim marginal KL(q(z|X)||p(z)) is logK. Hard encoder uses BIASED
  ST soft-routing query gradient; valid bound does not mean unbiased optimization.
- Controls: GAN, deterministic AE-GAN, soft categorical no-KL stochastic AE-GAN,
  soft categorical VAE-GAN. Same construction/seed24002/K400/z2/width128/batch256.
  Fixedsigma.0023566126; G/E LR.0003,D.00045,prior.003 (allhalved). Tau=.03 in all
  likelihoods; soft categorical temp.0025, AE/hard surrogate .25. Two posterior
  samples/train input. All generation evals100k at2k/4k/6k;8draws/8192heldout.
- Final: AE97modes/89.71%HQ/MSE.002341; stochasticnoKL96/92.92%/.007220;
  GAN96/75.49%; hardconstantKLVAE88/85.21%/.003264; softVAE86/82.50%/.003884.
  Ranking coverage-first, not universalbest. GANbestSW1.1685. Widthtoo narrowall.
- Hard posterior KL5.9915,effconditionalK1; aggregateeff194; pairRMS.01066,
  all65536drawssameinputmode; shuffleMSE16.50. SoftVAEpair.02946,same100%;
  noKLpair.03216,same98.03%. Hardvariationsmall; neitherpromisebroaddiversity.
  Hardactual likelihood G(z)+.03noise:99modes/75.76%HQ; boundnegative4.443nats
  vssoftVAE3.916. These differfromdecodermeanmetrics above.
- AE2k->4k->6k improves82/96/97modes and62.80/87.49/89.71%HQ. Otherstability
  unresolved: noKL4kdips60modes/49.38%HQ; hard96->88latecoverage,soft95->86.
- Fullqueue2.0minwall,3.12GPUtrainmin,3.25processmin.24tests+13subtests passed;
  fivepilots/fullfive passed,15checkpoint hashes/arrays/certificates/init/data/
  prior RNGs verified. Newhard test caught dtype handling, fixed beforepilots.
- Trainer `experiments/train_mog_vae_stability.py` copied/adapted to preserve
  oldtrainer provenance. Analyzer `analyze_mog_vae_stability.py`, test
  `test_mog_vae_stability.py`, configs`configs/mog_vae/stability{,_pilot}`.
  Logs`runs/mog_vae/stability.live.log`, `stability_pilot.live.log`,
  `stability_analysis.log`. Previousscout certificates remainvalid; oldtrainer
  and mainworktree trackedsource untouched.
- Suggested next (NOT launched): fixed-size uniform candidate-set posterior,
  `E(X)->exactly m particles; uniform samplewithinset + fixedpriornoise`.
  JointKLlog(K/m) constant, preservesparticlechoiceuncertainty for m>1 (trym4).
  Stillneedsapproximategradientforsetselection. Keep lowerLR AE as practical
  reconstructionbaseline, inspectstability/usage/width; don'tpromoteimagesyet.

## Previous: genuine particle VAE toy queue completed (2026-09-17)

User authorized starting VAE experiments through the config pipeline and queuing
a bunch. Completed 12 prespecified 6,000-update arms plus five 200-step pilots,
using both GPUs, one worker each. No seed-only experiments. Nothing running or
queued. Feature worktree/branch unchanged; this and previous DDGAN work remain
uncommitted. Main worktree tracked source unchanged.

- Read [results/recommendations](../reports/mog-vae/README.md),
  [leaderboard](../reports/mog-vae/LEADERBOARD.md), and
  [exact protocol](../reports/mog-vae/PROTOCOL.md).
- Main genuine variational family: `E(X) -> q(k|X); k~q; z=p[k]+fixed_sigma*eps`.
  `q(z|k,X)=p(z|k)`, so continuous KL is zero. Only exact categorical KL to
  Uniform(400). Gaussian decoder likelihood N(G(z),tau^2 I). Two iid posterior
  draws and unbiased leave-one-out score gradient; no ST categorical estimator,
  no exhaustive decoding. Objective MSE+tau^2*KL+GAN+particle spread. Prior sigma
  remains .0023566126. Reparameterization and KL are separate concepts.
- Optional local posterior learns u/log_s with analytic Gaussian KL; no-KL arm
  is explicitly stochastic AE-GAN, not VAE. Pure VAE retains spread regularizer.
- Matched seed24002, K400/z2/width128, batch256, G/E .0006, D .0009, prior .006,
  same toy spatial-query skip. Temps .0025/.025; tau .03/.1/.3. AE uses legacy
  .25 soft-backward nearest-forward and bounded offset. Same six-output E
  construction across all new arms. Historical toy trajectories are not reused.
- Final 100k decoder means: no-KL stochastic AE-GAN95 modes/76.01% HQ/MSE.004246;
  best coverage-first VAE sharp/.03:90/72.92%/.005626; deterministic AE84/73.50%/
  .004629; GAN80/59.31%. GAN still best SW1 .2755. All have width/fit problems.
- Best VAE uses3.23 effective particles/input; posterior pairRMS.03864 and all
  65,536 held-out draws retain input mode. Shuffled MSE16.23. KL4.8195 nats,
  categorical MI4.6036, aggregate categorical TV.248. Actual likelihood samples
  including tau=.03 noise have67.12%HQ. Large tau likelihood calibration poor.
- Local posterior std/prior std stays .982/.997; localKL tiny. Broader routing
  and no-GAN VAE worse. No KL benefit established over stochastic control.
- Matched-count read-only 4k audit: AE99modes/92.67%HQ ->84/73.50%at6k;
  GAN94/76.78%->80/59.31%; noKL91/71.81%->95/76.01%; categorical sharp/.03
  76/48.87%->90/72.92%. Late instability means final ranking is not robust
  superiority evidence. Same100k/RNG; checkpoints unchanged; no retraining.
- 12-arm full queue4.4minwall;7.57 summedGPUtrainmin;7.89 processmin including
  eval/I/O.22tests+13subtests passed; all5pilots/full12 passed; certificates,
  same init/data/prior RNGs, fixedsigma,36checkpoint hashes and saved arrays
  verified. Dry run12done/0to run. Initial pilots failed two evaluation bugs
  (unpacking, sparse undefined width); fixed before full queue, attempts archived.
- Entry `experiments/train_mog_vae.py`; analyzer `analyze_mog_vae.py`; audit
  `audit_mog_vae.py`; tests `tests/test_mog_vae.py`; configs `configs/mog_vae/`.
  Logs `runs/mog_vae/scout.live.log`, `pilot.live.log`, `analysis.log`,
  `late_audit.log`. Raw runs/source/checkpoints ignored; portable report retained.
- Recommended next (NOT launched): matched four-arm GAN/AE/noKL/categoricalVAE
  sharp tau=.03 with allLR halved, fixed-count intermediate evaluations, fixed
  sigma. Establish stability before more complexity/images. User has not yet
  selected next work or requested a commit of this round.

## Previous handoff: return to the toy problem for a genuine VAE

The user explicitly corrected the model name and requested a subagent-assisted
rename throughout the feature worktree. Canonical names are **particle AE-GAN**
and **DDGAN + particle AE**. Current E(X) and reconstruction are deterministic;
post-training latent jitter does not make the model variational. Display labels,
reports, plots, and the report generator have been corrected. Stable run IDs
(`bounded`, `direct_bounded`, `ddgan_bounded`) remain accurate and preserve
checkpoint/config/source provenance.

After compaction, return to the existing 2D 100-Gaussian toy experiment to
formulate and test an actual VAE against the deterministic particle AE-GAN.
The specific posterior family and objective have not been selected yet. Start
by spelling out `E(X) -> q(z|X)`, how sampling enters training, and the variational
objective/likelihood before assigning a VAE name. Merely adding sampled noise
to the existing reconstruction/GAN loss is not enough to establish an ELBO.
Fixed prior sigma can remain fixed; distinguish it from posterior variance.
Preserve the user's interest in particle-based inference without assuming a
standard Gaussian KL formulation is the final choice.

Use `experiments/train_mog_autoencoder.py`, the existing toy configs and
`reports/mog-autoencoder/` as the starting point. Keep the experiment cheap,
matched, measurable, and queued, with easy-to-tail logs and a leaderboard.
Include mode coverage, sample quality/distribution fit, held-out reconstruction,
conditional variation, posterior/prior diagnostics appropriate to the selected
objective, and runtime. No seed-only experiments. This turn only renames and
prepares the handoff; launch the toy experiments on continuation after compaction.

## Current status: direct / DDGAN comparison complete (2026-09-17)

User authorized DDGAN versus DDGAN + particle AE versus direct particle AE-GAN at 10k updates,
using both GPUs and the config pipeline, with additional experiments allowed.
Completed four arms by adding direct GAN control and using half the previous
learning rates throughout. No jobs remain running or queued. No seed sweep.

- Worktree `/home/martyn/dev/ParticleGAN-mog-autoencoder`, branch
  `feature/mog-autoencoder`; main tracked source untouched. This round's new
  implementation/results are not yet committed. Previous HEAD is `b06f01f`.
- Read [new report](../reports/cifar-particle-ddgan/README.md),
  [leaderboard](../reports/cifar-particle-ddgan/LEADERBOARD.md), and
  [protocol](../reports/cifar-particle-ddgan/PROTOCOL.md).
- **Final FID50k after 10k updates:** direct GAN19.483, direct particle AE-GAN20.054,
  DDGAN + particle AE43.233, DDGAN49.475. Training7.91/8.77/10.65/9.10 minutes.
  These are deterministic particle autoencoders, with no KL/ELBO/learned variance.
- All same-count FID5k curves improve through10k, avoiding previous late decline.
  Direct GAN35.093→28.344→26.175→24.549;
  particle AE-GAN32.212→27.304→25.230→24.691;
  DDGAN85.443→64.139→59.835→53.585;
  DDGAN + particle AE71.607→53.199→49.392→47.380.
- New comparison is unconditional, K1024/z64/fixedsigma.212616; matched
  G/D/E/prior initialization and all RNG streams within architecture pairs.
  Data/prior RNGs also match across architectures. G/E LR.0003, D.00045,
  prior.003; constantLR; reconstruction coefficient1; full-table VICReg1.
  DDGAN uses existing width32 U-Net, four-step schedule, four time-head feature
  critic, no auxiliary class loss. Historical conditional DDGAN31.56 at10k
  differs in labels/prior/LR/auxiliary loss and is not the matched control.
- DDGAN encoder sees clean X only: `E(X) -> z_X`; reconstruction
  `G(z_X, X_t, t) -> X`. GAN training/sampling still uses uniform MoG prior,
  independent latent per reverse step; encoder not needed for generation.
- DDGAN + particle AE improves matched baseline FID12.6%, costs16.9% extra training.
  Noisiest-t reconstructionMSE.097218 versus shuffled-code.398152,
  zero-offset.317364, shuffled-particle.151379. Fix X_t across ablations.
  These are clean-image predictions with side information, not latent-only or
  full encoded reverse-chain reconstructions.
- Direct particle AE-GAN testMSE.078996, PSNR17.04, zero-offset.304904,
  shuffled-particle.128659; lower LR worsens reconstruction versus previous
  .063365 despite better generation. Adds10.9% training time vs direct GAN.
  Direct/ DD encoder effective particles295/433, offset saturation31.1%/50.7%.
  Encoded offsets/particle usage still not matched to the sampling prior.
- Frozen variation audit:512 inputs,8 draws,0/.5/1sigma perturbation; DD noisy
  inputs/t fixed and context hashes matched across DD checkpoints. No audit
  images saved/viewed. At.5sigma all8 distinct for every input. Direct pixel
  pairRMSE6.96/255, MSE+1.93%, own-anchor retrieval99.78%, feature diversity
  12.31% of unrelated reconstructions. DD: RMSE.86–4.06, MSE+.08–.48%, retrieval
  100%, feature diversity1.35–5.01%. Retrieval is not semantic identity, and
  injected latent noise is not a learned posterior. Audits took123.7s total.
- Full four-arm queue:26.8min wall,52.11 summed GPU-processmin including eval,
  36.42 trainingmin; pilots1.02 processmin. Peak5.34GiB inclFID.
- 28 tests +13 subtests passed; all4 pilots/full runs succeeded. Analyzer
  verifies certificates, budgets, initializations/RNGs, fixedsigma/frozen
  features, saved per-image errors/usage,16 checkpoint hashes, variation arrays
  and matching DD noisy-input hashes. Dry-run:4 already done,0 to run.
- Entry point `experiments/train_cifar_particle_ddgan.py`, model adapters
  `lib/image_particle_ddgan.py`, configs `configs/cifar_particle_ddgan/`.
  Analyzer `experiments/analyze_cifar_particle_ddgan.py`, audit
  `experiments/measure_cifar_ddgan_variation.py`. Original direct trainer unchanged.
  Queue sources were held fixed; all raw configs/source archives/checkpoints
  retained under ignored `runs/cifar_particle_ddgan/`.
- Numbered checkpoints at2500/5000/7500/10000 now retained; checkpoint.pt is a
  latest symlink. No exact resume implemented. Logs easy to tail:
  `runs/cifar_particle_ddgan/scout.live.log`, `variation_*.log`, `analysis.log`.
- Recommendation from the image round: direct GAN for generation, particle
  AE-GAN for encoding/reconstruction. The user's next direction supersedes the
  previously proposed continuous-encoder image ablation: return to the toy
  problem after compaction and test a genuinely variational formulation.
  No further experiments are running or queued.

## Previous round: first CIFAR pair complete (2026-09-17)

The user authorized a basic matched CIFAR image experiment and a config queue
using both GPUs. Implemented and completed; no more training is queued.

Latest follow-up: the user requested numerical image-variation metrics without
visual inspection. Completed a 29.8-second read-only audit of the bounded EMA
checkpoint: 512 test inputs, eight draws, noise multipliers 0/.25/.5/1/2 around
`z_X`, plus sampling directly around the selected particle center. At .5sigma:
all8 outputs distinct, feature pair distance23.1% of unrelated reconstructions,
MSE+5.8%, own-reconstruction nearest99.34%. At2sigma MSE+80.7%, retention28.27%.
Center-only sampling loses the input (MSE+484.3%). Recommend .5sigma as a
starting inference setting, .25 for conservative variation; no semantic/VAE
posterior guarantee. See [variation report](../reports/cifar-particle-ae/variation/README.md).
No images were saved/viewed in this audit. Checkpoint/model hashes unchanged;
two known-answer tests pass; scripts are `measure_cifar_particle_variation.py`
and `analyze_cifar_particle_variation.py`. Log `runs/cifar_particle_ae/variation.log`.

- Read [CIFAR report](../reports/cifar-particle-ae/README.md) and its protocol.
- Direct unconditional CIFAR32, K1024, latent64, fixed sigma0.212616; same
  initialization/data/prior draws. GAN versus bounded particle autoencoder+GAN,
  10k updates each, one shared seed24002. Existing queue used unchanged with
  `--workers_per_gpu 1 --gpus 0,1`; combined log under
  `runs/cifar_particle_ae/scout.live.log`.
- Final FID50k: GAN81.411, bounded33.266. Training7.99 vs8.61min (+7.8%).
  Main pair total including eval21.2 summed process minutes,10.9 concurrent
  wall minutes. Peak5.32GiB including Inception.
- Important: both regress late. Same-count FID5k at7500->10000:
  GAN25.951->85.361; bounded25.267->37.918. Final-checkpoint read-only audit
  confirms this is not an evaluation sample-count artifact. Do not describe
  the large endpoint margin as an established stable generation advantage.
- Bounded testMSE.063365,PSNR18.00; blurry reconstructions. Zero-offset
  MSE.364833 (5.76x); shuffledparticle.073319 (+15.7%). Offsets carry much of
  the information, unlike the toy. Used363/effective109.5 of1024; offsetRMS2.04,
  11.3% saturation; no aggregate matching to the sampling Gaussian.
- Recommend next: same pair and budget, halve all learning rates to test
  stability; retain numbered checkpoints (currently only latest survives).
  This recommendation has not been launched. Continuous-encoder control is
  useful after baseline stability. Keep oracle training deferred.
- Scripts: `train_cifar_particle_ae.py`, `analyze_cifar_particle_ae.py`,
  `audit_cifar_particle_ae.py`; model `lib/image_particle_autoencoder.py`;
  configs `configs/cifar_particle_ae/{pilot,scout}`. Runs/checkpoints/source
  archives are in `runs/cifar_particle_ae/`; reports contain portable results.
- 62 tests +13 subtests pass. Analyzer verifies certificates, matched hashes,
  fixed sigma/frozen D features, complete budgets and reconstruction errors.
  First bounded audit terminated code143, preserved then evaluation-only retry
  succeeded. Two short pilots sharedGPU0 accidentally; full scouts used0/1.
- Main worktree source remains untouched. Continue in this feature worktree.

## Previous direction (implemented by the CIFAR round above)

User requested a commit and pause for compaction before further experiments.
Next: test the scalable bounded particle autoencoder + GAN on simple images,
against a matched MoG GAN with reconstruction disabled. No image experiment
has started. Joint oracle-supervised GAN training is deferred.

```text
Reconstruct: E(X) -> (k, u) -> p[k] + fixed_sigma * 3*tanh(u/3) -> G -> X_hat
Generate:   uniform k + Gaussian noise -> p[k] + fixed_sigma * noise -> G -> X_new
Train:      reconstruction + GAN objective + existing particle regularizer
```

This is a particle autoencoder + GAN, without a per-example VAE posterior or KL.
Hard selection uses nearest particle to an encoder query; a soft routing
surrogate supplies the query gradient. The baseline uses the original global
surrogate, not the later local-routing, balancing, or oracle additions.
Routing still costs O(batch × particles × latent dimension); it is not free,
but avoids exhaustive comparisons in image space.

Choose dataset, latent dimension, image architecture, loss scaling, budget, and
image-appropriate quality/diversity metrics before launching. A small digit-image
task is a candidate, not a locked protocol. Inspect existing image infrastructure
and available data first. Match G/D/prior, initialization, data streams, and update
budgets across arms. Calibrate sigma once for the new prior, then keep it fixed.
Record held-out reconstruction, generation quality/diversity, hard particle usage,
offset RMS and zero/random-offset ablations, runtime, and peak memory. Keep flushed
logs and publish a leaderboard plus interpretation. No seed-only experiments.

## Completed evidence

- Nine-arm 2D 100-Gaussian scout: bounded offsets give 92/100 covered modes,
  82.23% HQ, reconstruction MSE 0.002775. Best HQ/reconstruction; local routing
  reaches 93 modes with worse quality. Other metrics have different winners.
- Exhaustive zero-offset oracle: decode all 400 centers and pick the closest
  output to each observed X. Bounded encoder MSE 0.00280830 versus oracle
  0.00188026 on 100k held-out examples. Exact only among current center outputs;
  not an optimum over offsets, future decoders, or unconditional generation.
- Frozen encoder fitting: two 6,000-update arms from the bounded checkpoint,
  with G/prior/sigma frozen, zero offsets, fresh Adam, and identical continuation
  data. Oracle query regression gives MSE 0.00209430: 25.42% improvement and
  76.94% of the available gap closed. Reconstruction control gives 0.02755613;
  6.247% wrong-grid cases account for 94.10% of its error. Generation is unchanged.
- Oracle fitting demonstrates learnability of much of the selection gap. It
  does not establish image-scale performance. Oracle search was cheap here
  because 400 tiny decoded centers could be cached with G frozen.

## Workspace and artifacts

- Worktree: `/home/martyn/dev/ParticleGAN-mog-autoencoder`
- Branch: `feature/mog-autoencoder`; leave the main worktree untouched.
- Python: `/home/martyn/dev/ParticleGAN/.venv/bin/python`
- Commits: `40a55a7` initial scout; `9235c3b` local routing; `bb6cc0a` oracle audit;
  `c04615d` frozen encoder fitting. This handoff is committed separately.
- Reports: [scout](../reports/mog-autoencoder/README.md),
  [oracle](../reports/mog-autoencoder/ORACLE.md),
  [encoder fitting](../reports/mog-autoencoder/encoder-fit/README.md).
- Runs/checkpoints: `runs/mog_autoencoder/scout/` and
  `runs/mog_autoencoder/encoder_fit/` (gitignored; retained locally).
- Latest logs: `runs/mog_autoencoder/encoder_fit.console.log` and
  `runs/mog_autoencoder/encoder_fit.audit.log`; use `tail -F`.
- Existing image entry points include `experiments/train_cifar_ddgan.py`,
  `lib/image_ddgan.py`, and `lib/cifar_metrics.py`; suitability for a simple-image
  particle autoencoder has not yet been assessed.
- Latest code verification: 38 tests pass across `test_mog_encoder_fit.py`,
  `test_mog_oracle.py`, `test_mog_autoencoder.py`, `test_mog.py`, `test_mog_api.py`.
  Frozen state hashes, original checkpoint integrity, matched continuation RNG,
  complete budgets, and saved-source hashes were verified. No runs are active.
