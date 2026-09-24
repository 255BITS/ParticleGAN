# GAN-5 real-sample Laplacian basin — KILL

Host = neural. Torch `2.14.0+cpu`, one thread, AVX2. No merge and no production claim.

## Mechanism

The discriminator loss adds `0.1 * relu(ΔD)` at real planar samples, where `ΔD` is the finite-difference Laplacian of the sharp critic (`eps = 0.1`). A positive Laplacian is local convexity, so the relativistic generator is not pulled back to that sample. The penalty pushes real points toward local maxima of D. G remains the PR84 five-point stencil with curvature bounds `.25` / `3`. The stencil is defined only for planar critic inputs, so the trajectory host is the PR84 update.

## Purity

GAN dynamics only — no coverage/likelihood term. No particle assignment, Chamfer, mode quota, anchor, or data likelihood.

## Gate table vs this build's PR84 pin

Warm is the constant-rate continuation of updates 1001–1200 from the shared scheduled checkpoint. PR84 on that window is **196/200** (failures 1129–1132, minimum HQ `.866`, final eight / HQ `.999`). That matches the historical warm bar.

| Gate | PR84 pin | GAN-5 basin | Result |
| --- | --- | --- | --- |
| Warm 200 | 196/200, min HQ `.866`, final 8 / `.999` | **200/200**, min HQ `1`, final 8 / `1` | no regression |
| Cold trajectory 400 | published MSE `.000942662` PASS; this arm is the PR84 update on non-planar inputs | MSE `.000942668`, PASS | pass |
| Cold ring 1200 | **7 modes / HQ `.993`**, 0/24 checks, 23.7 s | **6 modes / HQ `.915`**, 0/24 checks, 31.3 s | **FAIL, worse acquisition** |
| Stay | not reached | not run | stopped |

Warm did not regress, so trajectory and ring were run. The ring never recorded an eight-mode check. Stay was not started.

## Rank

Unranked on the GAN-native board. Below the PR84 reference: borrowed warm retention improved, cold acquisition fell from seven modes to six.

## Kill

The concavity penalty deepens basins on samples the critic already treats as real. On a borrowed warm cloud that is enough to remove four short HQ dips. From scratch it leaves the generator in a six-mode subset and does not create the missing mode. Do not retune the coefficient. Do not continue this penalty.

Next single bet, not run here: a D-side term that changes the critic where the cloud is absent, rather than another occupied-sample curvature penalty.
