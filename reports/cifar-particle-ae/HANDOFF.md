# AE-GAN current handoff — plateau investigation and active 200k run

The investigation requested after compaction is complete. **The selected 200k job is running on GPU 0; do not launch duplicate work or stop it.** GPU 1 is idle. Branch: `feat/cifar-ae-gan-pretrained-encoder`.

## Active job

- Pipeline PID: 242245. Trainer: `experiments/train_cifar_ae_plateau.py`.
- Config: `configs/cifar_particle_ae/plateau_200k/d2.yaml`.
- Run: `runs/cifar_particle_ae/plateau_200k/d2/`.
- Tail: `tail -F runs/cifar_particle_ae/plateau_200k/PIPELINE.log`.
- Restored final two-D scout checkpoint at 60k; verified running beyond 60,100 with finite losses, correct init/hash and GPU 0 utilization. Goal: global step 200,000 (140k more G updates, 280k more D updates).
- Parent SHA256: `c31c0f703de54ef0fc6d281cbf7a92c14e8e9a3553fc2f4d5460558263d2a56a`.
- Two D updates per G, all previous loss/optimizer/EMA settings unchanged. Reconstruction updates E/G/prior. No learning-rate decay; the quarter-rate scouts were worse. N=8 exact bcap scaled by eight, indexed by D updates.
- FID50k and saved checkpoints every 10k, including final 200k; 10k held-out reconstruction diagnostics. Automatic report: `reports/cifar-particle-ae/plateau_200k/LEADERBOARD.md`.
- Expected runtime roughly three hours; six-hour training cap. Final result is PENDING. Launch evidence: `reports/cifar-particle-ae/plateau/LAUNCH.json`.

## Completed investigation

Main report: [plateau/FINDINGS.md](plateau/FINDINGS.md). Frozen diagnostics and image grids: [plateau/DIAGNOSIS.md](plateau/DIAGNOSIS.md).

All six scouts resumed the same historical 50k checkpoint for 10k G updates; all have source/config certificates. FID50k at 60k:

| Recipe | FID50k | Test MSE |
|---|---:|---:|
| Two D updates | **19.0289** | .04120 |
| Historical unchanged control | 19.9770 | .03861 |
| Reconstruction detached from prior | 20.2151 | .05266 |
| Reconstruction only updates E | 20.2712 | .07487 |
| Reconstruction weight .1 | 20.5006 | .04231 |
| All LRs ×.25 | 20.7400 | .03926 |
| Weight .1 and LRs ×.25 | 20.8273 | .04371 |

Best observed intermediate: two-D at 55k, FID18.7229. Final-endpoint winner selected for long continuation, not the intermediate checkpoint. Below-13 target remains unmet.

The user asked whether reconstruction moves particles and suggested it should not, then suggested reconstruction only on E. We confirmed the existing gradient path, implemented and tested both alternatives in standalone `experiments/train_cifar_ae_routing.py`, and ran both scouts. We explicitly explained that neither improved this continuation and selected the stronger-critic winner with existing reconstruction routing. These are changes after 50k of coadaptation; they do not settle from-scratch routing choices. No detached long run is queued.

Evidence: live critic gradients weaken drastically at 100k; the adversarial G gradient is ~4× smaller than at 50k. Reconstruction becomes comparable and locally opposed on G. At 100k reconstruction's prior-gradient norm is 1.83× adversarial and reaches all 1024 rows via mean/std normalization, but their directions are mostly orthogonal. Stronger D training helped the matched FID endpoint; lowering reconstruction or LR did not. Thus conflict/mismatch is not proven the sole cause. Two-D at 60k gave ~2.4× the G adversarial-gradient norm of the unchanged 60k control.

Frozen checkpoint original prior FIDs reproduce within 0.00003. Encoder-frequency sampling worsens FID; Gaussian fits to train offsets worsen it dramatically. Actual train-code replay has diagnostic FID148.35/136.50 at 50k/100k despite low reconstruction MSE. This demonstrates poor perceptual reconstruction quality; replay and held-out reconstruction FID are NOT unconditional benchmarks. All fit data came from the train split.

Ten tests passed (plateau six, routing four), including full-state deterministic replay and gradient-recipient tests. Six real-checkpoint 16-update smokes passed. Live source/config checks and frozen-feature/sigma checks passed on all six completed scouts. No seed sweeps.

## Files and operational notes

- Control pipeline: `experiments/cifar_ae_plateau_pipeline.sh`; routing pipeline: `experiments/cifar_ae_routing_pipeline.sh`.
- Per-GPU routing coordinator: `experiments/queue_cifar_ae_routing.py`; completed. It uses the existing grid runner and central log.
- Analyzer for both trainers: `experiments/analyze_cifar_ae_plateau.py` with optional `--trainer` for routing. Reports final ranking, best observed checkpoint and curves.
- Diagnostics: `experiments/diagnose_cifar_ae_plateau.py`, `experiments/probe_cifar_ae_gradients.py`. Probe reconstruction gradients are hypothetical before routing; `applied_recon_g_norm` and `gradient_routing` distinguish actual recipients. In particular, do not misread hypothetical large prior gradients for detached runs as applied training gradients.
- Reports: `reports/cifar-particle-ae/{plateau_scout,routing_scout}/LEADERBOARD.md`; aggregate JSON, gradient probes and narrative under `plateau/`.
- Do not modify active trainer or shared lib/particlegan files: certificates hash them. Isolated files preserved all old sources.
- Earlier history (100k baseline, capacity scouts, hashes) is archived in [HANDOFF_100k.md](HANDOFF_100k.md).
- Unrelated `.claude/`, `results/hopfield*`, `results/motion/`, `sparse-ucd.log` and run artifacts remain untouched/untracked.
