# Handoff: CIFAR particle DDGAN after normalization round

Prior runs finished. User authorized committing and launching the prepared
30k schedule pair on 2026-09-14. The previous experiment round ran on
both GPUs and emphasized keeping the solution/formulation from train_denoising.py.
We tested per-image GroupNorm in D using the two previously tried rate recipes.
Read [READOUT.md](READOUT.md) for leaderboard, figures, diagnostics and next steps.

## Selected no-argument default

`experiments/train_cifar_ddgan.py` loads `configs/cifar_ddgan/default.yaml`.
Width32 U-Net G; GroupNorm D with additive time conditioning; UCD;
learned20k x128 prior; Gaussian initial image and step noise; T4;
alpha_bar [1,.9,.5,.05,.0001]; 10k updates, batch64; G LR .0006,
D multiplier1.5 (.0009), prior multiplier10 (.006). These are restored toy rates.
Same Rp logistic, UCD CE .02, candidate-only bcap coeff/kappa1, unique-row
VICReg1, Adam(0,.999), EMA .995, cosine after60% to .05 floor.
No reconstruction objective, pure diffusion or flow. G predicts clean x0;
the shared posterior creates xprev. D keeps continuous xt and time inputs.

`lib/image_ddgan.py` now supports d_norm:none/group. ResBlock distinguishes
normalization from affine conditioning: G uses affine scale/shift; D retains
additive shifts with either norm setting. None path initialization, state keys
and outputs match archived original source exactly. Historical repository YAMLs
now explicitly say d_norm:none so selecting them does not inherit the new
GroupNorm default. Archived result/config/source artifacts are unchanged.
The normalization change is implemented; the obsolete draft patch was removed.

## Completed final FID50k

All runs seed24002, 10k updates, batch64. No seed sweep (AGENTS.md).

| Run | FID | Train min |
|---|---:|---:|
| normalized_d/toy_lr (new winner, UCD32) | 62.819 | 10.40 |
| normalized_d/image_lr (UCD32) | 76.863 | 11.26 |
| image_lr/concat32 | 180.069 | 8.76 |
| image_lr/ucd32 | 186.564 | 8.18 |
| baseline/width32 (UCD toy LR) | 191.516 | 8.23 |
| conditioning/concat32 (toy LR) | 198.464 | 8.18 |
| baseline/width64 (UCD toy LR) | 220.189 | 17.98 |

New pair completed in14 minutes wall time using both A6000 GPUs. Two new
100-step smoke runs also certified. Combined tail history remains:
`tail -F results/cifar_ddgan/live.log`.
Normalized configs: `configs/cifar_ddgan/normalized_d/{toy_lr,image_lr}.yaml`;
manifest JSON in same directory. Outputs under results/cifar_ddgan/normalized_d.
Reports export config/summary/metrics/samples/provenance/environment/certificates;
large checkpoints and source.zip remain ignored under results.

## Evidence and next recommendation

GroupNorm improves both matched rate recipes. Candidate gradients in final raw
D on a fixed256-real-image probe increased from roughly .011/.011/.009/.002
(t1..4) to1.024/1.022/.652/.062 with toy rates. Soft cap permits small >1 values.
Samples improve strongly but animals/class fidelity/diversity remain weak.
Latent changes alter outputs, but no CIFAR prior/noise benefit is established.

User accepted a revised next plan: **30k width32 cosine versus constant LR**,
then spatial particle injection, then pretrained D features. This supersedes the
width32/width64 recommendation. Full configs and manifest are prepared under
`configs/cifar_ddgan/schedule_30k/`; validated, not launched. See
[NEXT_ROUND.md](NEXT_ROUND.md) for exact settings, command, hypotheses and followups.
No-argument default remains the measured10k winner until new results justify a
change. Keep the train_denoising.py formulation throughout.

## Verification / integrity

Before launch:6 image/resume +19 toy/regularizer tests passed; both GPU smoke
runs passed. After default promotion:6 image/resume tests passed again, including
normalized D CUDA interruption/replay. Test enforces deterministic CUDNN; production
uses TF32/benchmarking and does not promise bitwise equality. FID pinned SciPy
deprecation warning is benign; logging scalar warning fixed in prior round.

Default promotion changed trainer hashes after certification. Historical strict
resume needs matching source.zip and saved config.yaml, including this new round.
Never rewrite source metadata/certificates to current hashes. Analyzer validates
saved provenance and rejects mixed source hashes within a group.

Dependencies/data/cache unchanged: torch2.14+cu130, torchvision0.29+cu130,
torch-fidelity0.3, scipy1.17.1. CIFAR data verified and cached; do not redownload.
FID uses50k unaugmented CIFAR train reference, TF-compatible Inception2048,
uint8 rounding, extractor TF32 disabled, float64 moments. Balanced generated
classes. No independent class classifier. Published~3.75 DDGAN is unconditional;
protocol equivalence to that benchmark is not established.

Repo: master, previous9a4aff0 pushed before image work. Image changes still
being committed before the authorized schedule round. Preserve unrelated .claude/
and sparse-ucd.log. Root AGENTS.md: no seed experiments, token-efficient,
easy-tail logs, summarize leaderboard/explanations/recommendations. No new
subagents authorized or spawned. Prior audit_bcap agent is historical/idle.
