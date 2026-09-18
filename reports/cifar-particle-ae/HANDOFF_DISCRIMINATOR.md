# Compaction handoff: fix the AE-GAN FID plateau

## Latest user direction

User: "ok i'm going to compact then we'll set the north star to fixing this and continue".

Prepare for compaction now; do not launch another experiment in this turn. Next session's intended north star is to identify and fix the AE-GAN FID plateau, with sustained improvement toward the existing FID50k target below13. Judge interventions by matched FID trajectories and compute cost, not discriminator AUC, raw gradient magnitude, reconstruction MSE, or a selected intermediate minimum alone. The root cause is not established; update balance is a hypothesis to test, not an assumed diagnosis.

Branch: `feat/cifar-ae-gan-pretrained-encoder`. Implementation commit `38d9ef8`; completed analysis commit `32b96df`. All experiments finished; no training queued. Both GPUs were idle after final probes. Recheck actual status before launching. Prior architecture history is in `HANDOFF_TRANSGAN.md`, which links older rounds.

## Read first

`discriminator_joint/FINDINGS.md` contains the conclusions and recommendation. `discriminator_joint/LEADERBOARD.md` and `curves.png` contain the matched FID results. Detailed diagnostics are in `discriminator_diagnosis/FINDINGS.md`.

All three joint runs restored the same CNN E-only10k checkpoint (FID50k19.4482), then added10k joint updates:

| Arm | FID50k at15k | FID50k at20k |
|---|---:|---:|
| Control | 19.7875 | 20.0119 |
| Weaker bcap | 20.1344 | 19.7848 |
| D-only warmup | 27.4657 | 25.2598 |

Weaker changes only bcap coefficient1→0.1, retaining lazy8. Warmup installs D and its Adam state after2048 D-only updates with original regularization, then restores the original joint recipe. One D update per G update throughout joint training. Test reconstruction MSE stays near0.148 in all arms. Weaker/control ranking reverses; the final0.23-point gap is unconvincing. Warmup recovers somewhat but remains much worse. No endpoint promoted. All three scouts certified; queue took19.3 minutes.

## What the diagnosis established

- Holding G/prior fixed, existing D learns held-out real/fake AUC0.5255→0.9194 in2048 updates with original bcap. Weaker coefficient0.1 lazy8 reaches0.9552; original coefficient1 every step reaches0.9471. Training times48.6/47.7/159.5 seconds. Train/test ranking agrees. Both pixel and pretrained feature heads learn, weakening a simple feature-information failure explanation.
- Every-step regularization was NOT tested in joint FID; deferred because of cost. Do not claim it was ruled out. Original10k/50k image gradients are below cap1 at the probe, but sparse logs show the penalty can activate elsewhere in training.
- Warmup initially retains strong feedback after8 joint updates (AUC0.9389, G gradient4.7574). By20k that advantage disappears (AUC0.4322, G gradient0.0589). Better D-only classification did not yield better joint FID.
- No strong pixel/feature cancellation in image, G-parameter or prior-parameter gradients. Parameter decomposition uses the actual combined GAN loss derivative, in FP32, with additive errors around1e-6. Production settings unchanged.
- Eight-step traces at20k show D improves held-out ranking and G/prior then reduces it on8/8 observed updates. Counterfactual mean AUC: afterD0.5524, only prior update0.5519, only G update0.4616, both0.4613. Immediate ranking change is mostly from G. This is expected adversarial behavior, NOT itself proof of a bad learning-rate ratio. Joint game dynamics and usefulness of feedback are stronger suspects; the sole cause remains open.

## Proposed next controlled test — NOT launched

Resume the same10k parent and halve **G-only** learning rate0.0003→0.00015. Keep D/E/prior rates fixed, one D update, lazy8, original bcap coefficient1, and compare against an unchanged control. Existing `lr_scale` changes all optimizer groups and cannot isolate this: add an explicit G-only setting in a new standalone trainer to preserve source certificates. This tests update balance without the extra cost of two D updates.

Do not automatically freeze or replace the backbone, restart the historical full-reconstruction CNN, or promote another endpoint for a small isolated gain. User dislikes paying for D2 for about one FID point and prefers controlled checkpoint interventions. No seed experiments.

## Checkpoints and implementation

Parent: `runs/cifar_particle_ae/transgan_scout/cnn_e_only/checkpoint_010000.pt`.
SHA256: `d75fca4bc42ec09f1423ce1a671b4cbd10caefe0abccae3ac2bdb05d5d93237c`.

New joint20k checkpoints and hashes: `discriminator_joint/CHECKPOINTS.json`.
D-only delta: `runs/cifar_particle_ae/discriminator_diagnosis/current/D_delta.pt`; contains D/Adam only, not a joint checkpoint. When continuing a warmstart joint checkpoint, clear `d_warmstart` and `d_warmstart_sha256` to avoid reapplying the old warmup.

Standalone trainers:
- `experiments/diagnose_cifar_ae_discriminator.py`: D-only/probes; pins historical TransGAN trainer hash.
- `experiments/train_cifar_ae_discriminator.py`: copy of original trainer plus explicit coefficient and D warmstart.
- Pipelines: `cifar_ae_discriminator_pipeline.sh`, `cifar_ae_discriminator_joint_pipeline.sh`; matching `analyze_cifar_ae_discriminator*.py` analyzers.
- Read-only checkpoint pipeline: `probe_cifar_ae_discriminator_joint.py` (early/final stages).
- Gradient decomposition: `probe_cifar_ae_discriminator_projection.py`.
- Eight-step instrumented diagnostics: `trace_cifar_ae_discriminator_game.py`, `probe_cifar_ae_update_components.py`. These temporary update traces are not FID/runtime benchmarks.

Historical/shared source files were not changed. Source certificates cover trainers and shared code: use standalone entry points for further interventions. Persistent jobs use detached `subprocess.Popen` with redirected logs. Completed pipeline log: `tail -F runs/cifar_particle_ae/discriminator_joint/PIPELINE.log`; PID262835 exited0.

## Validation and detailed evidence

Three D-only smokes and three actual-parent joint smokes passed. Four initial diagnostics, three joint scouts, and five endpoint probes certified. Exact unchanged old/new full-state continuation passed (one test,6.07 seconds) under deterministic CUDA in the test only. Parent/source hashes, frozen state, D Adam counts and nonmutating probes verified.

Initial exact replay test lacked full deterministic enforcement and failed tiny floating-point equality; fixed test setup only. An initial TF32 gradient decomposition failed its strict additive check (0.000367 error); the read-only calculation passed with TF32 disabled. Production training unchanged. No seed replicates, so small run-to-run uncertainty is not quantified.

Evidence: `discriminator_diagnosis/{results.json,GAME_TRACE.json,UPDATE_COMPONENTS.json,PROJECTED_GRADIENTS*.json,LOGGED_PENALTIES.json}`; joint endpoint probes in `discriminator_joint_probes_early/` and `discriminator_joint_probes_final/`. Final grids inspected; varied samples, no claim of total mode collapse.

No subagents spawned in this round. Applicable AGENTS.md: no seed experiments, be token efficient, make logs easy to tail, summarize findings/leaderboard/recommendations after completion. Unrelated untracked `.claude/`, `results/hopfield*`, `results/motion/`, `runs/`, `sparse-ucd.log` untouched.
