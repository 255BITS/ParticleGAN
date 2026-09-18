# Extend8192/16384 from20k to40k

User authorized: "ok lets extend them", referring to the two larger particle-count scouts. Both run on the existing feature branch, one worker perGPU. Full original optimizer/EMA/RNG state retained; no parameter, objective, noise or architecture change. No new expansion is performed: factors8/16 remain unchanged and expansion audits record intervention=false.

| GPU | Particles | Initial FID50k | Additional steps | Endpoint |
|---|---:|---:|---:|---:|
|0|8192|18.1602|20000|40000 total|
|1|16384|18.0136|20000|40000 total|

FID50k and numbered full checkpoints at25k/30k/35k/40k. G/E LR0.0003,prior0.003,D0.00045; oneDupdate,bcapcoeff1every8×8, E-only reconstruction,EMA0.995,sigma0.2126164287. Compare against the existing4096 duration trajectory; do not rerun it or add seed experiments.

The existing scaling trainer is unchanged; its factor8/16 exact full-state continuation tests already passed in the prior preflight. Parent source certificates and SHA256digests verified again before launch. Both actual runs confirmed restoration at20000, first update20001, GPUs100% active.

After training, the pipeline certifies both runs, verifies fixed rates and original RNG pairing, and writes results.json,CHECKPOINTS.json,LEADERBOARD.md,FINDINGS.md,curves.png. It then runs the existing read-only information/density/coverage diagnostic on both40k endpoints with full default budgets, reusing the4096 endpoint diagnostic and identical cached real reference. These reports go to `particle_scaling_40k_information/`.

Train orchestration: `experiments/cifar_ae_scaling_extend.py`. PID269541 launched around17:26MDT2026-09-18. Estimated training+diagnostics20–25minutes. No further training past40k is queued.

`tail -F runs/cifar_particle_ae/particle_scaling_40k/PIPELINE.log`

Endpoint diagnostic progress subsequently appears in `runs/cifar_particle_ae/particle_scaling_40k_information/PIPELINE.log`; launcher.log captures overall pipeline errors/results. Check final FID and the trajectory separately from the selected minimum. Target remainsFID50k<13.
