# CIFAR AE-GAN plateau: current handoff

Branch: `feat/cifar-ae-gan-pretrained-encoder`. Latest user: "ok cool lets plan some experiments for that next, compacting". Detailed planning-only next-round specification is saved in `particle_expansion/PLAN.md`; read it before implementation. No new experiment was launched for this compaction request. This round completed the proposed G-only LR scout, endpoint D probes, and an additional read-only sampling/support investigation. All jobs finished; both GPUs idle. No long job or further scout is queued. Root cause remains unproven and FID below 13 remains unmet.

Read `generator_balance/FINDINGS.md` and `particle_support/FINDINGS.md`. Previous detailed discriminator investigation is archived in `HANDOFF_DISCRIMINATOR.md`; earlier architecture history in `HANDOFF_TRANSGAN.md`. Implementation commit for G LR scout: `b0cbfce`.

## Completed matched training

Both restore the identical CNN E-only 10k parent (FID50k 19.4482), full model/Adam/EMA/RNG. Only G LR changes: 0.0003 -> 0.00015. E 0.0003, prior 0.003, D 0.00045; original bcap coefficient 1 lazy8 x8, one D update. No seed experiments.

| Arm | FID50k 15k | FID50k 20k | Test MSE | Training minutes |
|---|---:|---:|---:|---:|
| Control | 19.5589 | 20.3044 | 0.14846 | 7.20 |
| Half G LR | 20.7412 | 20.6996 | 0.14310 | 7.34 |

Half-G loses at both evaluations; neither beats parent. Final 0.3952 gap is small, so claim absence of demonstrated gain, not that all lower G rates are harmful. No promotion.

Read-only 2048-image endpoint D probes: control/half-G test AUC 0.4743/0.5575, fake image gradient 0.1558/0.4921, G adversarial gradient 0.1435/0.5724. Half-G increases measured feedback ~4x without improving FID. This joins negative D-warmup/weaker-bcap results: feedback strength alone has not fixed the plateau. Post-G endpoint AUC is phase dependent; do not treat it as a unique diagnosis. Sources/results in `generator_balance_probes/`.

## Read-only support investigation

EMA G/prior remain fixed. At each noise multiplier use original particle IDs and underlying Gaussian draws, 50k FID protocol unchanged. Baseline x1 reproduces original FID within 0.001 for all three checkpoints. All three certified.

| Checkpoint | Centers only | Original noise | Double noise |
|---|---:|---:|---:|
| Original 10k | 41.8580 | 19.4481 | 21.7837 |
| Control 20k | 42.4706 | 20.3039 | 21.7199 |
| Half-G 20k | 43.4404 | 20.7005 | 23.5012 |

Noise is used and contributes to image variation; broader inference noise is not a fix. Center-only FID describes a finite 1024-output generator, so its high score is not itself proof of poor prototypes or collapse. In inspected grouped samples, draws within each particle retain object/pose/layout and vary mostly locally. Balanced ANOVA (128 particles x16 draws) assigns 32–33% Inception-feature variation but about 5% pixel variation within particles. These are descriptive fractions, not semantic coverage metrics. Latent covariance effective rank stays ~64/64 at 10k and 50k, ruling against global latent dimension collapse in these snapshots.

Grouped grid: `runs/cifar_particle_ae/particle_support/control_20k/within_particle.png`. Other checkpoints have corresponding grids. Midpoint/final joint sample grids and control grouped grid inspected; no total collapse claim.

## Recommended next test — NOT launched

Concrete two-wave plan: `particle_expansion/PLAN.md`. First compare 1024 unchanged versus 4096 cloned/trainable centers on the two GPUs, 10k ->20k with FID50k at15k/20k. Mandatory preflight covers normalization, fixed sigma, per-row Adam/EMA mapping, regularizer differences, paired RNG and initial image/FID equivalence. Conditional follow-ups distinguish useful expansion from slow symmetry breaking; no automatic long promotion.

Test expanding trainable particles 1024 -> 4096 from the same checkpoint, alongside a matched 1024 control. This tests whether more centers can learn distinct image configurations; wider inference noise did not. Preserve G/D/E and their Adam/EMA, saved sigma, and explicitly map expanded prior/EMA/Adam state. Audit initial generated distribution/FID before training to distinguish expansion initialization from training effects. **Simply repeating raw particle rows does not exactly preserve standardized means because `prior.means()` uses unbiased std**; account for this rather than claiming exact identity. Preserve historical source certificates via a new standalone trainer.

Discriminator feedback quality/robustness remains the competing hypothesis, not a ruled-out explanation. Existing D can classify fixed-target data but useful joint gradients are unproven. Every-step bcap has only been tested D-only, not in matched joint FID. Do not freeze E/prior or grow G solely on these observations. No automatic long promotion on small or reversing differences.

## Files and validation

Parent: `runs/cifar_particle_ae/transgan_scout/cnn_e_only/checkpoint_010000.pt`, SHA256 `d75fca4bc42ec09f1423ce1a671b4cbd10caefe0abccae3ac2bdb05d5d93237c`.
Endpoint checkpoint paths/hashes: `generator_balance/CHECKPOINTS.json` (canonical saved 20k copies also exist in each run directory).

Standalone trainer `experiments/train_cifar_ae_balance.py` adds `g_lr_scale`, defaults 1, applied only to G optimizer group after restoration on every step. Existing global `lr_scale` affects all groups and is not equivalent. Actual group LRs recorded and audited. `cifar_ae_generator_balance_pipeline.sh`, `analyze_cifar_ae_balance.py` implement grid/reporting. `tests/test_cifar_ae_balance.py`: 2 deterministic CUDA tests passed, exact original full-state continuation and isolated actual first half-G update (unchanged D/E/prior and all Adam moments). Two actual-parent eight-step smokes and both full scouts certified; matched RNG use, intervention metadata and optimizer counts checked.

Read-only `probe_cifar_ae_support.py`, `cifar_ae_support_pipeline.py` implement sampling/ANOVA probes with pinned historical architecture dependency, full source archives and parent/frozen state checks. Initial small smoke computed correctly but was rejected by grid because summary lacked required `final` block. Added block and reran in fresh `particle_support_smoke_v2`; passed. Original failed attempt/source archive retained, documented in `particle_support/PREFLIGHT.md`. Small-smoke FIDs are not benchmarks. All 3 full probes then certified. Two endpoint D probes reuse unchanged original diagnostic trainer and are certified.

Completed logs:
- `tail -F runs/cifar_particle_ae/generator_balance/PIPELINE.log` (PID 264755, exit0, 9.8min)
- `tail -F runs/cifar_particle_ae/particle_support/PIPELINE.log` (PID 265248, exit0, about7min)

No subagents used. User preferences: no seed experiments, token efficient, easy-to-tail logs, completed experiment leaderboard/explanations/recommendations. They prefer checkpoint interventions and consider two D updates too expensive for ~1 FID point. Historical full-reconstruction CNN should not be retrained without reason. Persistent jobs use detached Popen with redirected logs. Historical/shared code unchanged. Unrelated `.claude/`, `results/hopfield*`, `results/motion/`, `runs/`, `sparse-ucd.log` preserved; generated `results/failures.txt` records the support preflight formatting failure.
