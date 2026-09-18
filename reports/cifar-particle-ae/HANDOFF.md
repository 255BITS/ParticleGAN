# CIFAR AE-GAN plateau: particle expansion improves FID

Branch: `feat/cifar-ae-gan-pretrained-encoder`. Latest user: "lets continue to find the reason why we are plateauing". Completed the planned 1024 versus 4096 particle scout; 4096 wins at both evaluations. Both respective endpoints are now continuing to40k on the two GPUs. No jobs beyond40k are queued. Root cause remains non-unique; target FID50k<13 is still unmet.

Read `particle_expansion_scout/FINDINGS.md`, `LEADERBOARD.md`, `CHECKPOINTS.json`. Implementation commit `4a370de`. Earlier detailed state archived in `HANDOFF_BALANCE.md`; prior D and architecture work in `HANDOFF_DISCRIMINATOR.md`, `HANDOFF_TRANSGAN.md`.

## Completed scout

Same CNN E-only10k parent: `runs/cifar_particle_ae/transgan_scout/cnn_e_only/checkpoint_010000.pt`, SHA256 `d75fca4bc42ec09f1423ce1a671b4cbd10caefe0abccae3ac2bdb05d5d93237c`.

| Arm | Initial FID50k | FID15k | FID20k | Training minutes |
|---|---:|---:|---:|---:|
| control1024 |19.44808|19.86990|19.89316|7.25|
| split4096 |19.44812|18.93574|18.52071|7.37|

Both certified. Expansion gains0.9342/1.3724FID for1.71% extra training time, and improves versus parent. Wall10.82/10.94min includes three FID50k passes. No seed repeats. Production CUDA nondeterminism explains modest variation among historical contemporary controls; exact deterministic unchanged-control test passes.

Sibling RMS per coordinate reaches0.09140/0.11283 at15k/20k, or0.43/0.53 times sigma. All4096 rows sampled250–397 times by20k. Coupled-noise image/feature differences grow, while inspected sibling grids still largely preserve parent object/pose/layout. This supports an effective prior-flexibility/learning-dynamics intervention, not proof of4096 semantic modes or a unique support ceiling. Skip small sibling jitter for now. D feedback quality remains an additional candidate; dense bcap is not yet joint-FID tested.

## Running persistence check

PID **266873**, launched with detached Popen; orchestration `experiments/cifar_ae_expansion_extend.py`.

`tail -F runs/cifar_particle_ae/particle_expansion_40k/PIPELINE.log`

GPU0 control1024, GPU1 split4096. Full optimizer/EMA/RNG continuation from respective20k checkpoints. Steps20k→40k, FID50k25k/30k/35k/40k, numbered full checkpoints. Unchanged rates G/E.0003, prior.003, D.00045, oneD update, bcapcoeff1 every8×8, E-only reconstruction. Both confirmed training past21k, GPUs100%. Expected total~20minutes from launch around16:10MDT on2026-09-18.

Config paths: `configs/cifar_particle_ae/particle_expansion_40k/`. Upon completion pipeline certifies both, audits rates/RNG pairing, and writes `reports/cifar-particle-ae/particle_expansion_40k/{results.json,LEADERBOARD.md,FINDINGS.md}`. Inspect curve and sibling grids, then update this handoff. No automatic200k promotion. Sustained4096 benefit could justify further matched training or8192; fading benefit would return priority to discriminator feedback robustness. Do not repeat same-seed experiments as a proxy for seed uncertainty.

## Implementation/preflight

Standalone `experiments/train_cifar_ae_expansion.py` copies balance trainer, leaving every historical/shared certified source untouched. Config `num_particles=1024` is reference initialization count; `expansion_factor=4` means4096 live rows. Build/calibrate original prior first; clone only after restoring/mapping saved state, preserving sigma0.2126164287 and d08.50465679. Do not construct/recalibrate on coincident clones.

ExpandedPrior corrects unbiased std by sqrt(N*(M−1)/(M*(N−1))); ReferenceRegularizer corrects variance and covariance similarly. Live/EMA initial center error<1e-6, coupled image max error2.44e-6, summed regularizer-gradient error<2.13e-11. Full initialFID difference only0.0000458. Prior Adam row moments copied, steps retained, noLR compensation. Expansion necessarily changes exposure/optimizer dynamics, so it is not a pure abstract capacity isolation.

Separate persisted clone-choice RNG preserves original parent-ID/noise/data streams. Evaluation resets/restores child RNG; exact full-state resume tested. Extra state records reference count, clone RNG, exposure. Expanded exposure accumulates since expansion; control exposure covers current continuation. Direct selection does not enumerate gradients through shared standardization/regularization. Read expanded checkpoints with this trainer's helpers, not an unmodified MoG constructor of4096 rows.

Three CUDA tests in `tests/test_cifar_ae_expansion.py` passed11.90s: exact original full-state control replay; real-parent mapping/sampling/image equivalence; exact expanded save/resume and E-only reconstruction gradients (E only, G/prior none). Two8-update pipeline smokes passed, allAdam counters10008. Small smoke128-image FID~143 is not a benchmark. Early unit-test-only panel-size assumption fixed before benchmark runs. Harmless scalar-conversion warning in preflight logs documented.

`experiments/cifar_ae_expansion_pipeline.py` creates/certifies scout and smoke tracks. `experiments/cifar_ae_expansion_extend.py` requires >0.5FID gain at both scout evaluations plus endpoint beating parent before creating40k configs, then certifies/reports. Detailed validation: `particle_expansion/PREFLIGHT.md`, `TESTS.txt`, historical center movement and hypotheses there.

Completed scout PID266496, exit0; logs in `runs/cifar_particle_ae/particle_expansion_scout/PIPELINE.log`. Keep all parent/checkpoint/source hashes intact. All operations remain on the feature branch; no subagents used. User preferences: no seed experiments, token efficient, tail-able logs, completed leaderboard/explanations/recommendations. Unrelated untracked `.claude/`, `results/failures.txt`, `results/hopfield*`, `results/motion/`, `runs/`, `sparse-ucd.log` preserved.
