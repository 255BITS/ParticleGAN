# Residual-generator learning-rate scout

Authorized GPU1 only, sequential forks from the certified16k residual CNN80k full-state checkpoint. Existing unchanged90k/100k results reused. No seed experiments.

| Arm | G LR | E LR | Prior LR | D LR |
|---|---:|---:|---:|---:|
| existing unchanged | .0003 | .0003 | .003 | .00045 |
| half_g | .00015 | .0003 | .003 | .00045 |
| half_all | .00015 | .00015 | .0015 | .000225 |

Both80k->100k; FID50k every5k, reconstruction10k, fixed original sigma, moving16k centers, EMA.995, E-only reconstruction, oneDupdate, bcap every8 multiplied by8. These are constant-rate drops at the checkpoint, not a gradual decay schedule. Preserve full Adam, EMA, random streams and prior exposure. Keep every5k checkpoint. No automatic promotion beyond100k.

Hypotheses: (1) G changes too quickly late in training; (2) slowing the entire coupled system matters; (3) selected layers/conditioning pathways have disproportionately large effective updates. Successful lower-rate continuation supports update dynamics, not residual-specific causality. Equalized layer reparameterization and architecture changes are deferred.

Diagnostics: actual one-step Adam update RMS, parameter RMS, their ratio and gradient RMS per G parameter tensor at logs; initial and every5k live/EMA block activations, conditioning scale/shift, output saturation and fixed-input pixel drift. Use256 fixed latent draws from parent EMA prior with independent RNG so prior movement does not contaminate G drift. Compare existing90k/100k generator snapshots using identical inputs; historical controls lack per-step updates. Reduced pixel drift is expected mechanically and is not itself evidence of improved quality.

Verification: direct observational audit must preserve model/EMA/prior states, gradients, Adam state, modes and RNG bitwise; the next Adam update with common gradients must match bitwise. Certified16-update half-G and half-all smokes verify actual training and rates. Initial cross-run bitwise comparison was rejected because the historical trainer contains nondeterministic CUDA adaptive_avg_pool2d backward; even a shared-process comparison differed and PyTorch strict determinism explicitly rejected that operation. We do not claim bitwise-identical full training trajectories. Smokes have no FID benchmark. Old trainer/library sources remain untouched. Full jobs start only after these checks pass. Endpoint probes reuse the certified FID/coverage/overlap pipeline.

Tail:

```sh
tail -f runs/cifar_particle_ae/particle_lr_80k_scout/PIPELINE.log
```

Controller/status: `runs/cifar_particle_ae/particle_lr_80k_scout/launcher.log`, `reports/cifar-particle-ae/particle_lr_80k_scout/STATUS.json`. Failures halt the queue. Results, checkpoint hashes, leaderboard and conditional recommendations update after each arm and probe.
