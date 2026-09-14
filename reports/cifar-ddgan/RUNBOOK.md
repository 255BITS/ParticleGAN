# Handoff after schedule round and joint-UCD toy scout

**All runs finished; nothing active or queued. Ready for compact.** User asked
for a commit regardless, constant LR as CIFAR default, and joint(t,c) UCD as
the preferred candidate if competitive. Baseline committed before experiments
as62843b9. A second commit records results and the new candidate. No push requested.

## Read first

- [READOUT.md](READOUT.md): CIFAR schedule leaderboard, curves and sample grids.
- [Toy scout](../denoising-toy/joint_ucd/READOUT.md): joint UCD comparison/tradeoffs.
- [NEXT_ROUND.md](NEXT_ROUND.md): exact prepared next experiment and launch command.

## Results

CIFAR:30k constant class-only UCD FID50k43.678 (33.95 training minutes),
30k cosine49.390 (31.29 minutes), old10k cosine62.819. Both GPUs,39.2min pair,
zero failures. Constant won the final endpoint; cosine led briefly before late
regression. Same seed24002; no seed sweep; nondeterministic production kernels
caused trajectories to differ before schedules diverged. See readout caveats.

Toy:56k class-only versus joint(t,c), both GPUs,8.3min pair, zero failures.
JointHQ91.250% versus91.905%, all100 modes, classacc96.185% versus96.325%,
conditionalSW1 .1040 versus.1190, conditionalTV .0590 versus.0623, posteriorSW1
.18977 versus.19028. Competitive, with tradeoffs, not a universal win. User's
preference for a unified conditioning rule breaks the practical tie.

## Current CIFAR candidate default

No args: `.venv/bin/python experiments/train_cifar_ddgan.py` reads
`configs/cifar_ddgan/default.yaml`, mirrored by literalDEFAULTS.
30k updates; constantLR (`lr_floor:1`); FID every10k; final50k samples;
`ucd_target:time_class`;40 heads, no D timestep embedding. D uses candidate and
xt; index `(t-1)*10+c` selects the adversarial head and real/fake CE target.
Class-only mode remains available and historical repository YAMLs explicitly
select it. The43.678 measured winner is class-only; **full joint CIFAR FID is
unmeasured**. Two100-update real-CIFAR smokes validated the new image option.

Unchanged: width32 U-Net G/GroupNorm D; G gets t,c; learned20k x128 latent prior;
Gaussian step noise; T4 and alpha_bar[1,.9,.5,.05,.0001]; clean prediction and
shared posterior from train_denoising.py; Rp logistic; candidate-only bcap1/kappa1;
CE.02; unique-row VICReg1; batch64; G LR.0006,D.0009,prior.006; Adam(0,.999);EMA.995.
No reconstruction/pure-diffusion objective. Core shared sampler unchanged.

Toy uses `ucd_target:class|time_class` too (4 versus16 heads); joint removes D's
one-hot timestep input. Toy no-argument default remains class-only,56k with its
original cosine schedule. The ablation changes only the D inputs/heads and CE
label; G, posterior, bcap and particle objective stay fixed.

## Validation and provenance

25 focused checks before CIFAR schedule pair;28 before toy scout;31 after image
port including joint image bcap/sample independence, head invariance, and CUDA
checkpoint replay. Exact pre-change G/D states/outputs verified in old toy/image
modes. Two toy100-update GPU smokes and two image100-update GPU smokes certified.
All four full runs certified. FID SciPy deprecation warning is benign and pinned.

Source changed only after active grids finished. Historical strict resume needs
matching source.zip + saved config. Never rewrite saved-source certificates.
Source archives/checkpoints remain ignored in results; report exports are tracked.
Current image configs add an explicit ucd_target key; archived run configs do not
get rewritten. Analyzer validates saved config/source metadata independently.

## Workflow and environment

Next pair prepared under configs/cifar_ddgan/joint_ucd, **not launched**. Both
GPUs authorized. Continue combined tail at `results/cifar_ddgan/live.log`.
Runtime around35–37min with lower FID frequency. Do not edit training sources
while a grid is active. User wants token-efficient waits until runs finish.

Two RTX A6000 GPUs; GPU1 also drives a desktop, don't disturb it. CIFAR data and
FID cache already verified; no download needed. .venv torch2.14+cu130,
torchvision0.29+cu130,torch-fidelity0.3,scipy1.17.1. FID50k unaugmented CIFARtrain
reference, TF-compatible Inception2048, float64 moments, extractorTF32off.
Labels are used; globalFID doesn't establish class fidelity or equivalence to
published unconditional~3.75 DDGAN. Existing checkpoints save full image state.

Root AGENTS.md: no seed experiments; token-efficient; easy-tail logs; summarize
leaderboard/explanations/recommendations. No new subagents authorized. Preserve
unrelated .claude/ and sparse-ucd.log. Original toy commit9a4aff0 already pushed.
