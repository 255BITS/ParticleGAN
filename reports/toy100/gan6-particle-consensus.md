# GAN-6: particle consensus from critic disagreement

Host: neural. PyTorch 2.14.0+cpu, one thread. Draft only. No merge.

## Mechanism
After each PR84 generator step, a particle that shares a sharp-critic basin with another particle is moved toward an empty high-critic probe on the particles' own radial shell; with no empty probe the shared basin shrinks, and distinct basins stay put.

## Purity
GAN dynamics only — no coverage or likelihood term. The move uses the sharp critic on generated particle images and on probes on that cloud's own shell. It does not assign real samples, match nearest neighbors to data, use Chamfer, mode labels, or quotas.

## Gate table vs this harness's PR84 pin

PR84 pin is smoothed G critic + alternating curvature (`.25` / `3`). Cold trajectory MSE `.000942668` (PASS). Cold ring **7 modes, HQ `.9927`**, 0/24 checks (FAIL). The scheduled prefix at update 1000 is already **6 modes, HQ `.994`**, so the 200-step warm window cannot clear a 7-mode pass on this build. PR84 then holds that cloud: **6/6 modes on all 200 checks, min HQ `.935`**.

| Gate | PR84 pin | GAN-6 |
| --- | --- | --- |
| Warm 1001–1200 | 0/200 pass; holds 6 modes, min HQ `.935` | **Regress.** min HQ `.191`, 198/200 checks below HQ `.9`, modes 3–7 |
| Cold trajectory | PASS, MSE `.000942668` | not run |
| Cold ring 8 | FAIL, 7 modes, HQ `.993` | not run |
| Stay | not run | not run |

Warm regression stops the ladder. The separate move fired at the 0.35 cap on the already-covered warm cloud (false empty probes on the shell), which kicked particles out of HQ balls.

## Rank
GAN-native track only. Below the PR84 reference. Does not acquire ring 8 and does not stay on the warm solution.

## Keep / kill / next
**Kill** this rule. Do not retune the step cap or shell count on the same test.

Next single bet: separate only across a critic valley, and only when the probe score exceeds the donor basin, with the output step kept inside the HQ radius so a false hole cannot eject a particle. Stay idle when every high shell probe already has a nearby particle.
