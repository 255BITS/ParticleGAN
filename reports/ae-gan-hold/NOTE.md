# AE-GAN hold toy (CPU)

One family. Seed 0, 250 Adam steps, batch 64, lr 2e-3. Two blobs at (±1.5, 0).
Logs are `ae-gan-hold arm=...` lines on stdout (`flush` each line).

This PASS is a CPU toy. It is not a Music or Anima GPU transfer.

## Formulation that is gated

Locked_shared adversarial shape, on top of `get_recipe("ae_gan")`:

| knob | pin | note |
|---|---|---|
| pairing | RpGAN logistic | `gan_mode=rp`, `loss_type=logistic` |
| penalty | `GradRegularizer` b_cap | coeff=1, kappa=1, norm=l2, lazy=1, anneal=none |
| FM | 0 | FM-on under b_cap is refused |
| cover | **1.5** | demo / Field3D pin, not Music 1.0 |
| particles | n=12, `particle_l2=0.02` | `particle_l2 * z.square().mean()` on raw rows |
| recon | weight 1 | `ParticleEncoding.reconstruction_loss` (no KL) |

Not used: Hub 128-particle mixture, the AE-GAN study default K=400, and the recipe's VICReg `prior_reg`. The package has no critic; the toy owns a 2-layer MLP and does not swap one.

`ae_gan`'s study preset uses lazy regularization every 4 steps. This gate pins lazy=1 to match locked_shared.

## Leaderboard (seed 0)

Bars: recon MSE ≤ 0.05, unconditional hold ≤ 0.15, and b_cap + RpGAN on every step. Hold is the mean distance from each blob to the nearest decoded prior sample. Init recon is about 1.28.

| arm | recon MSE | hold | b_cap steps | RpGAN steps | verdict |
|---|---:|---:|---:|---:|---|
| locked (recon + RpGAN b_cap, FM off, cover 1.5) | 0.0038 | 0.0239 | 250/250 | 250/250 | **PASS** |
| AE-only (recon + particle_l2) | 0.0024 | 0.0054 | 0/250 | 0/250 | **FAIL** |
| stranger pairing (`gan_mode=vanilla`) | — | — | refused | refused | **FAIL** |
| FM-on (`fm_weight=0.1`) under b_cap | — | — | refused | refused | **FAIL** |
| κ-hardcoded thin b_cap (center 0.1, or `kappa=0.1`) | — | — | refused | refused | **FAIL** |

## What the split means

Reconstruction fidelity is real: the locked arm moves recon from 1.28 to 0.0038, under the 0.05 bar.

AE-only also reconstructs (0.0024) and its unconditional samples sit on the blobs (hold 0.0054). On this toy the latent is the data plane, so reconstruction already places particle codes on the blobs. Hold does **not** separate the arms. The gate still FAILs AE-only because the locked adversarial shape never ran: zero RpGAN generator steps and zero b_cap applications, plus `adversarial_weight=0` and `cover_weight=0`.

Drifted adversarial recipes are refused before training. A subclass that advertises `kappa=1` but hardcodes center 0.1 is not a `GradRegularizer` champion, and neither is `kappa=0.1`.

## Recommendation

Keep this gate as a shape-and-fidelity check for the AE-GAN helpers. Do not read AE-only's low hold as evidence that the adversary is unnecessary, and do not promote the PASS to a Music (cover 1.0) or image claim. A later toy that wants the adversary to be the thing that covers modes needs a path where the encoder offset can reconstruct without the prior samples landing on the data.
