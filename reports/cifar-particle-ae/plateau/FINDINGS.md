# AE-GAN plateau: findings and 200k continuation

The best tested continuation is two discriminator updates per generator update: **FID50k 19.0289 at 60k**, versus **19.9770** for unchanged continuation. The best observed intermediate score is **18.7229 at 55k**. The below-13 target remains unmet. All six scouts completed with valid source/configuration certificates.

## Completed leaderboard

Every scout starts from the same 50k checkpoint (FID50k 18.9012) and ends at 60k. These are different configurations with a common seed, not seed experiments. Ranking uses final endpoints.

| Rank | Change | D updates / G | Reconstruction updates | Final FID50k ↓ | Test MSE ↓ |
|---:|---|---:|---|---:|---:|
| 1 | d2 | 2 | E/G/prior | 19.0289 | 0.04120 |
| 2 | no_recon_prior | 1 | E/G | 20.2151 | 0.05266 |
| 3 | encoder_only_recon | 1 | E | 20.2712 | 0.07487 |
| 4 | recon01 | 1 | E/G/prior | 20.5006 | 0.04231 |
| 5 | lr025 | 1 | E/G/prior | 20.7400 | 0.03926 |
| 6 | recon01_lr025 | 1 | E/G/prior | 20.8273 | 0.04371 |
| — | Historical unchanged control | 1 | E/G/prior | 19.9770 | 0.03861 |

The ordinary prior and FID protocol are unchanged for every training result: 50k samples, EMA G/prior, CIFAR train50k reference, TF-compatible Inception. Test reconstruction uses the 10k held-out split. The historical control is a matched continuation endpoint, not an independent replication. Timing varied with diagnostic overlap and the extra D work; it is not a clean speed benchmark.

## What the evidence says

1. **Insufficient adversarial training is the leading actionable explanation.** The live 100k critic had almost no real/fake separation and much weaker image gradients than at 50k. Two D updates were the only tested intervention to improve the matched 60k FID. At 60k, the generator adversarial-gradient norm was 0.360 with two D updates versus 0.150 in the unchanged control. This supports giving the critic more training; it does not establish a unique root cause or guarantee continued gains.

2. **Reconstruction optimizes a different notion of quality and influences the prior substantially.** At 100k its prior-gradient norm was 1.83× the adversarial norm and reached all particle rows through mean/std normalization. Its generator gradient was also comparable to the adversarial gradient and locally opposed to it. However, reducing its weight worsened FID, and neither detachment scout improved FID. The negative cosine is a measured symptom, not proof that reconstruction is globally harmful. At the unchanged 60k checkpoint the G-gradient cosine was slightly positive.

3. **Detachment has a real reconstruction cost in this continuation.** Blocking reconstruction from the prior raised test MSE to 0.05266; sending it only to E raised MSE to 0.07487, versus 0.03861 in the unchanged control. FIDs were 20.2151 and 20.2712. The encoder-only implementation preserves input gradients through G while excluding G parameters from that reconstruction graph. It does not freeze G for adversarial training.

4. **A simple sampler change is not the fix.** Reproducing the ordinary FID agreed within 0.00003. Particle-frequency sampling worsened FID. Gaussian fits to train encodings worsened it substantially. Replaying real train encodings gave diagnostic FIDs 148.35/136.50 at 50k/100k, showing that low pixel MSE can coexist with blurry, perceptually poor outputs. This replay is not an unconditional benchmark. Full results and sample grids are in [DIAGNOSIS.md](DIAGNOSIS.md).

## Chosen long run

- Continue the **final 60k two-D-update checkpoint to global step 200,000**: 140k additional G updates, 280k additional D updates.
- GPU 0; existing experiment pipeline; identical model, objective, optimizer hyperparameters, EMA and RNG state. Constant learning rates are retained because the lower-rate scouts did not improve FID. No new untested combination is added.
- Reconstruction continues to update E/G/prior in this selected run. Both detached alternatives are implemented and tested, but their short continuation results do not support replacing the winner.
- Keep exact lazy bcap every eight D updates with coefficient multiplied by eight. Retain all 10k checkpoints and evaluate FID50k at 70k, 80k, …, 200k. The analyzer reports both final and best observed results.
- Expected duration: about **three hours**, estimated from 14.38 G updates/s plus 14 evaluations. Training time cap is six hours.
- Parent checkpoint SHA256: `c31c0f703de54ef0fc6d281cbf7a92c14e8e9a3553fc2f4d5460558263d2a56a`.

```bash
tail -F runs/cifar_particle_ae/plateau_200k/PIPELINE.log
```

Config: `configs/cifar_particle_ae/plateau_200k/d2.yaml`. Output: `runs/cifar_particle_ae/plateau_200k/d2/`. Automatic completion report: `reports/cifar-particle-ae/plateau_200k/LEADERBOARD.md`.

## Limits and next recommendation

The routing scouts changed a model already trained for 50k with reconstruction coupled to G/prior. They test this continuation, not whether encoder-only reconstruction or detached particles work better from initialization. Longer adaptation or combining detachment with two D updates remains untested. The selected 200k run tests the strongest measured improvement first. If it stalls, a focused next experiment should compare stronger critic training with/without reconstruction-to-prior gradients from the start, rather than assuming that lower reconstruction weight or a Gaussian latent fit will solve the problem.

## Validation and artifacts

Ten tests passed across the plateau/routing suites, including deterministic full-state continuation and exact reconstruction-gradient recipient checks. All six real-checkpoint 16-update configuration smokes passed. All six 10k-update scouts passed pipeline certificates and frozen-feature/sigma checks. Historical trainer/lib sources and source checkpoints were preserved.

[Four-control leaderboard](../plateau_scout/LEADERBOARD.md) · [Routing leaderboard](../routing_scout/LEADERBOARD.md) · [Plan](PLAN.md) · [Frozen-checkpoint diagnosis](DIAGNOSIS.md)

## Endpoint gradient probes

Values are snapshots across 16 batches. Potential reconstruction gradients are distinguished from applied gradients when routing is disabled.

| Run | G adversarial norm | Applied G reconstruction norm | D fake-input gradient norm | D paired accuracy |
|---|---:|---:|---:|---:|
| baseline_60k | 0.1496 | 0.0538 | 0.6641 | 0.614 |
| d2 | 0.3602 | 0.1137 | 0.7542 | 0.629 |
| no_recon_prior | 0.0697 | 0.0635 | 0.1006 | 0.449 |
| encoder_only_recon | 0.0323 | 0.0000 | 0.0451 | 0.441 |
| recon01 | 0.0296 | 0.0141 | 0.0434 | 0.426 |
| lr025 | 0.1363 | 0.0302 | 0.6744 | 0.615 |
| recon01_lr025 | 0.1168 | 0.0037 | 0.6424 | 0.602 |
