# AE-GAN current handoff — plateau investigation and active 200k run

The investigation requested after compaction is complete. **The selected 200k job is running on GPU 0; do not launch duplicate work or stop it.** GPU 1 is idle. Branch: `feat/cifar-ae-gan-pretrained-encoder`.

## Latest user direction — 2026-09-18, before compaction

The user likes starting from saved checkpoints and testing controlled interventions and wants to use that approach again. They judge two D updates not worth roughly one FID point. **Do not assume two D updates should be retained for the next experiment round.** They asked to prepare compaction, then they will provide ideas to discuss. Wait for those ideas before queuing new experiments; do not automatically extend the current recipe or launch another sweep. The current run was not canceled.

At this snapshot the active run is at **159,000 / 200k**, with final result still pending. Latest evaluated FID50k is **18.7748 at 150k**, test reconstruction MSE **0.037819**. Best observed is **18.3879 at 130k**. The trajectory is a slight drift within roughly 18.5–19, not a sustained approach to 13. Recheck current logs after compaction: the process may have advanced or finished.

FID50k at 70/80/90/100/110/120/130/140/150k:
19.151 / 18.893 / 19.134 / 19.142 / 18.852 / 19.260 / 18.388 / 19.033 / 18.775.

Checkpoint candidates (do not select until the next ideas are discussed):

- **original_single_d_50k**: FID50k 18.9012, `runs/cifar_particle_ae/duration_100k/n08/checkpoint_050000.pt`; SHA256 `10fe8bbc22afb29ff6838ad1ede86e142320e5d7bce43c745b97de24e23ee8d6`.
- **best_two_d_130k**: FID50k 18.3879, `runs/cifar_particle_ae/plateau_200k/d2/checkpoint_130000.pt`; SHA256 `610f02784fb0f17e2678b09a950040413ac95e43462f7757b74f163d561b241f`.
- **latest_evaluated_two_d_150k**: FID50k 18.7748, `runs/cifar_particle_ae/plateau_200k/d2/checkpoint_150000.pt`; SHA256 `7aa9df6be3c543373b943c209b42a51723fb7cc677e81cb977ea150a6245c37f`.

The original 50k checkpoint has the one-D recipe and was the common parent of the completed scouts. The 130k/150k checkpoints inherit two-D model/Adam state; changing the D ratio is an explicit intervention, not a fresh baseline. Full model/EMA/Adam/RNG continuation and the existing source/config certificates must be preserved. No seed sweeps. Keep centralized, tail-friendly logs and publish leaderboards and interpretations.

The comparison was ~0.95 FID improvement at matched 60k G updates (19.03 versus 19.98). Two-D throughput is ~14.6 G updates/s versus ~22 for one D, so the extra compute is material. Ten-k scouts from coadapted checkpoints do not settle from-scratch choices.

The detailed live snapshot is [plateau/COMPACTION_SNAPSHOT.json](plateau/COMPACTION_SNAPSHOT.json). Prior launch evidence remains historical, not current status.

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

Best intermediate in the completed short scouts: two-D at 55k, FID18.7229 (superseded by the active long run’s 130k result above). Final-endpoint winner selected for long continuation, not the intermediate checkpoint. Below-13 target remains unmet.

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
