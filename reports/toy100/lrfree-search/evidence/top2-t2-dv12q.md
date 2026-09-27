# t2-dv12q: dv12-ams-rc3 sample-quality / noise probe

Base dv12-ams-rc3 fails only img_intensity2 (2m, hq .84). Diagnosis:
- img_intensity2 quality threshold is rmse <= **.06** (not .1).
- With continuous_policy set, output_noise_std is a constant (.029). The harness `measure` adds it at
  evaluation (`_generate(model, prior.z, output_noise_std(...))`), and the dv12 controller also perturbs latents there.
  So the eval noise alone costs ~.029 of the .06 rmse budget. Base trace: a slow quality plateau (hq .78-.88 after step 425), not oscillation.
- The same constant noise is also used in training, and mode_hold depends on it (8-mode coverage).

Stage A (intensity2, bars4, v.unequal_mass, mode_hold); change vs base only:
| cand | change | int2 | bars4 | v.mass | mode_hold |
|---|---|---|---|---|---|
| ons0   | out_noise 0     | PASS 7/24 | FAIL | PASS | FAIL 7m hq.99 |
| ons010 | out_noise .010  | F sfx2 hq.91 | PASS | PASS | FAIL 7m hq.99 |
| ons018 | out_noise .018  | F 8/24 sfx4 hq.97 | PASS | PASS | PASS 7/24 @900 |
| ons020 | out_noise .020  | F hq.88 | PASS | PASS | FAIL 8m hq.83 |
| ons022 | out_noise .022  | F sfx4 hq.91 | PASS | PASS | FAIL 8m hq.93 sfx2 |
| ons025 | out_noise .025  | F sfx2 hq.91 | PASS | FAIL sfx2 | FAIL 8m hq.87 |
| b2p99  | betas (0,.99)   | F sfx1 | PASS | FAIL | FAIL 8m hq.81 |

None passes all of stage A, so no quick-11 or ring runs.
- Lowering output noise clearly lifts intensity2 HQ (up to .97-1.0). At <=.010, mode_hold loses a mode (7/8).
- .018-.025 is on a knife edge: mode_hold HQ with 8 modes lands anywhere from .83 to .94 and does not change monotonically with the noise level.
- beta2 .99 is worse everywhere.
- Best: ons018 (3/4; intensity misses the 5-check suffix by one because of a single .88 at step 500).
Next idea (needs a package change, not tried): use separate noise levels for training and for eval/sampling
(keep .029 in training for mode_hold, sample with less), but that changes what the frozen measure samples, so it needs sign-off first.
