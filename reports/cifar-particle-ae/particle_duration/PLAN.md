# Continue32k and test longer16k training

User explicitly authorized continuing32k and delegated the second GPU choice. Choose longer16k because80k just produced the new bestFID15.7527 with improved density/coverage. This checks whether extra optimization time remains productive toward200k, while32k80k supplies a matched count comparison against the completed16k80k run.64k can follow if the count advantage persists.

| GPU | Particles | Resume → stop | Starting FID50k | Estimated elapsed time |
|---|---:|---|---:|---|
|0|16,384|80k →160k|15.7527|65–75 minutes|
|1|32,768|40k →80k|16.3451|35–40 minutes|

Both retain exact trainer/config recipe and full optimizer/EMA/RNG state, oneD update, E-only reconstruction, bcap every8steps×8, fixedsigma and learning rates. FID50k every10k, numbered full checkpoints, automatic full endpoint information/density/coverage probes on the same GPU. No seed repeats, no source edits, no subsequent promotion.

Parents and SHA256:16k80k `runs/cifar_particle_ae/particle_16k_80k/16k_80k/checkpoint_080000.pt`, `9ddcef1fb0bf82c47a581fa1b2d58872315d5d67b302eadc533c0abc41b7d5c7`;32k40k `runs/cifar_particle_ae/particle_32k_40k/32k_40k/checkpoint_040000.pt`, `533ff24ebb99cab375c46b19af58a4bff25a21848ed0c93c829a2dc3a25a6473`. Parent training/probe source certificates and hashes validated before launch. Existing exact-resume tests cover both factors; no trainer implementation change or redundant smoke training needed. New orchestration compiled; actual restoration/first updates checked on both GPUs.

Pipeline `experiments/cifar_ae_particle_duration.py --arm {16k_160k,32k_80k}`. Follow `tail -F runs/cifar_particle_ae/particle_duration/PIPELINE.log`. Per-arm launcher logs in that directory; PIDs inLAUNCH.json. Training reports under particle_16k_160k and particle_32k_80k; diagnostics under corresponding _information. After completion report full curves, endpoint/best-sampled leaderboard, density/coverage/bits, interpretation and next recommendation. Unequal endpoint durations are not a matched count comparison.
