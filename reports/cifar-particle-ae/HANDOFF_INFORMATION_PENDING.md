# CIFAR AE-GAN plateau: particle expansion improves FID

Branch: `feat/cifar-ae-gan-pretrained-encoder`. Latest user request: add the discussed feature-information/quality metrics in a subagent. Particle scaling remains the priority, target~13; best observed now16.5033 at35k. Completed the planned 1024 versus 4096 particle scout; 4096 wins at both evaluations. Both respective endpoints completed40k;4096 best16.5033 at35k, final17.2350, versus1024 final26.0216. 8192/16384 scouts are now running; read-only information/quality probes are queued after them. Root cause remains non-unique; target FID50k<13 is still unmet.

Read `particle_expansion_scout/FINDINGS.md`, `LEADERBOARD.md`, `CHECKPOINTS.json`. Implementation commit `4a370de`. Earlier detailed state archived in `HANDOFF_BALANCE.md`; prior D and architecture work in `HANDOFF_DISCRIMINATOR.md`, `HANDOFF_TRANSGAN.md`.

## Completed scout

Same CNN E-only10k parent: `runs/cifar_particle_ae/transgan_scout/cnn_e_only/checkpoint_010000.pt`, SHA256 `d75fca4bc42ec09f1423ce1a671b4cbd10caefe0abccae3ac2bdb05d5d93237c`.

| Arm | Initial FID50k | FID15k | FID20k | Training minutes |
|---|---:|---:|---:|---:|
| control1024 |19.44808|19.86990|19.89316|7.25|
| split4096 |19.44812|18.93574|18.52071|7.37|

Both certified. Expansion gains0.9342/1.3724FID for1.71% extra training time, and improves versus parent. Wall10.82/10.94min includes three FID50k passes. No seed repeats. Production CUDA nondeterminism explains modest variation among historical contemporary controls; exact deterministic unchanged-control test passes.

Sibling RMS per coordinate reaches0.09140/0.11283 at15k/20k, or0.43/0.53 times sigma. All4096 rows sampled250–397 times by20k. Coupled-noise image/feature differences grow, while inspected sibling grids still largely preserve parent object/pose/layout. This supports an effective prior-flexibility/learning-dynamics intervention, not proof of4096 semantic modes or a unique support ceiling. Skip small sibling jitter for now. D feedback quality remains an additional candidate; dense bcap is not yet joint-FID tested.

## Completed persistence check (launch details retained)

PID **266873**, launched with detached Popen; orchestration `experiments/cifar_ae_expansion_extend.py`.

`tail -F runs/cifar_particle_ae/particle_expansion_40k/PIPELINE.log`

GPU0 control1024, GPU1 split4096. Full optimizer/EMA/RNG continuation from respective20k checkpoints. Steps20k→40k, FID50k25k/30k/35k/40k, numbered full checkpoints. Unchanged rates G/E.0003, prior.003, D.00045, oneD update, bcapcoeff1 every8×8, E-only reconstruction. Both confirmed training past21k, GPUs100%. Expected total~20minutes from launch around16:10MDT on2026-09-18.

Config paths: `configs/cifar_particle_ae/particle_expansion_40k/`. Upon completion pipeline certifies both, audits rates/RNG pairing, and writes `reports/cifar-particle-ae/particle_expansion_40k/{results.json,LEADERBOARD.md,FINDINGS.md}`. Inspect curve and sibling grids, then update this handoff. No automatic200k promotion. Sustained4096 benefit could justify further matched training or8192; User now explicitly prioritizes further particle-count scaling; do not divert into a Gaussian-prior comparison. Do not repeat same-seed experiments as a proxy for seed uncertainty.

## Implementation/preflight

Standalone `experiments/train_cifar_ae_expansion.py` copies balance trainer, leaving every historical/shared certified source untouched. Config `num_particles=1024` is reference initialization count; `expansion_factor=4` means4096 live rows. Build/calibrate original prior first; clone only after restoring/mapping saved state, preserving sigma0.2126164287 and d08.50465679. Do not construct/recalibrate on coincident clones.

ExpandedPrior corrects unbiased std by sqrt(N*(M−1)/(M*(N−1))); ReferenceRegularizer corrects variance and covariance similarly. Live/EMA initial center error<1e-6, coupled image max error2.44e-6, summed regularizer-gradient error<2.13e-11. Full initialFID difference only0.0000458. Prior Adam row moments copied, steps retained, noLR compensation. Expansion necessarily changes exposure/optimizer dynamics, so it is not a pure abstract capacity isolation.

Separate persisted clone-choice RNG preserves original parent-ID/noise/data streams. Evaluation resets/restores child RNG; exact full-state resume tested. Extra state records reference count, clone RNG, exposure. Expanded exposure accumulates since expansion; control exposure covers current continuation. Direct selection does not enumerate gradients through shared standardization/regularization. Read expanded checkpoints with this trainer's helpers, not an unmodified MoG constructor of4096 rows.

Three CUDA tests in `tests/test_cifar_ae_expansion.py` passed11.90s: exact original full-state control replay; real-parent mapping/sampling/image equivalence; exact expanded save/resume and E-only reconstruction gradients (E only, G/prior none). Two8-update pipeline smokes passed, allAdam counters10008. Small smoke128-image FID~143 is not a benchmark. Early unit-test-only panel-size assumption fixed before benchmark runs. Harmless scalar-conversion warning in preflight logs documented.

`experiments/cifar_ae_expansion_pipeline.py` creates/certifies scout and smoke tracks. `experiments/cifar_ae_expansion_extend.py` requires >0.5FID gain at both scout evaluations plus endpoint beating parent before creating40k configs, then certifies/reports. Detailed validation: `particle_expansion/PREFLIGHT.md`, `TESTS.txt`, historical center movement and hypotheses there.

Completed scout PID266496, exit0; logs in `runs/cifar_particle_ae/particle_expansion_scout/PIPELINE.log`. Keep all parent/checkpoint/source hashes intact. All operations remain on the feature branch; no subagents used. User preferences: no seed experiments, token efficient, tail-able logs, completed leaderboard/explanations/recommendations. Unrelated untracked `.claude/`, `results/failures.txt`, `results/hopfield*`, `results/motion/`, `runs/`, `sparse-ucd.log` preserved.


## Next particle-count scaling queued (latest steering)

Read `particle_scaling_scout/PLAN.md` and `LAUNCH.json`. Queue PID267443 waits for both40k continuations and their certification, then runs CUDA tests, two8-update smokes, and8192/16384 scouts on GPUs0/1. Log: `tail -F runs/cifar_particle_ae/particle_scaling_scout/PIPELINE.log`. Persistent stage status: `particle_scaling_scout/QUEUE_STATUS.json`.

New counts start from the same original1024-center10k checkpoint used by the completed4096 scout; continue10k→20k with initial/15k/20kFID50k. Existing1024/4096 benchmarks reused, no seed repeats. No further promotion is queued. CPU actual-parent mapping tests passed2/2; CUDA tests and smokes are queued and must pass before full training.

Current intermediate40k measurements:4096 FID25k17.8876,30k17.4796;1024 FID25k21.8973,30k21.0193. Best known17.4796 is4.4796 above target13. Current run still in progress; check live reports for later points before responding.

Standalone `experiments/train_cifar_ae_scaling.py` preserves previous certified trainer, generalizes expansion factors to1/4/8/16 and descendant-panel dimensions. Config num_particles remains reference1024. `experiments/cifar_ae_scaling_pipeline.py` generates/certifies new count scouts and reports a combined scaling curve with old benchmarks. `experiments/queue_cifar_ae_scaling.py` handles dependency, preflight and launches; failure stops later stages. `tests/test_cifar_ae_scaling.py` adds8/16 actual-parent mapping/output and full-state resume checks. No historical/shared sources were edited.


## Information metrics implemented via requested subagent

`/root/particle_information_metrics` implemented standalone `experiments/probe_cifar_ae_information.py` and5passingCPUtests in `tests/test_cifar_ae_information.py`. Root reviewed the definitions, split separation, null controls and cache/source handling, and added `experiments/cifar_ae_information_pipeline.py` plus `experiments/queue_cifar_ae_information.py`.

QueuePID268218 waits for8192/16384 scout queue to finish, then rerunsCPUtests, runs a4096checkpoint GPU smoke, and seven full read-only probes. Log: `tail -F runs/cifar_particle_ae/particle_information/PIPELINE.log`. Current actual stage is `reports/cifar-particle-ae/particle_information/QUEUE_STATUS.json`; do not assume GPUvalidation has run until checked. Read `particle_information/PLAN.md`.

Checkpoints:1024/4096/8192/16384 at20k;1024/4096 at40k;4096 at35k for comparison with finalFIDrise. Fixed10000real/fake images,k5; same32originalparents with64train/32validation/64test examples per child and16additional variance draws. Density/coverage use cached, locked, SHA-validated realInception features; exact chunkedFP32distances. Read-only decoder uses train-only centroids/diagonalvariance, validation shrinkage/temperature selection including uniform, fresh test noise. Bits=log2K−testCEbits, negative estimates retained. Controls: independently shuffled balanced labels per split and exact synthetic feature clones. NestedANOVA separates parent/sibling/noise variation; fractions descriptive with sampling-noise caveat. Full moments include generated/realtrace ratio and squaredfeaturemean distance. NoFIDrecomputed: attach existing50kscore for exact checkpointstep.

Summary `final={information,variance,density_coverage}`; these also top-level. `information.observed/shuffled_labels/identical_clones` eachcontain decodable_bits,test_ce_bits,test_accuracy,parent_standard_error. `available_sibling_bits`,per_parent rows andbudgets retained. `variance` has within_child/between_siblings/between_parents fractions. `density_coverage` includesdensity,coverage,k,samples,feature_variance_trace_ratio_to_real,feature_mean_distance_squared. Parent/frozenstatechecks and pinned scalingtrainer sourceSHA keep historicalsources intact. Savedfeaturepanels support laterCPUanalysis. Conditionalinformation is a restricted-decoder lower-bound estimate, not exactentropy orsemanticcoverage. Information alone can rewardartifacts; interpret jointlywithquality/FID.

Completed40k reports now in `particle_expansion_40k/`: leaderboard/findings/results/checkpoint hashes/plot. Both finalsamplegrids inspected; variedoutputs withshape/detailerrors, no totalcollapseclaim.8192/16384GPUpreflight7tests passed19.51s;both8-update smokespassed;at15kFID18.6479/18.3401 versusprior4096 at15k18.9357. These newscouts are notcomplete yet. No furthertrainingpromotions queued; focuscurrentworkonadding/validatingmetrics asuserrequested.
