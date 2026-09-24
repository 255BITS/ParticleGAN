# Selected H verification: FAIL

**Cold acquisition replay: 10/10 PASS, with all live metrics and non-timing verdict fields exactly matching the archived run.** The subsequent strict 22-case sequence stopped at unipolar: **10 PASS, 1 FAIL, 11 SKIPPED**. An explicitly authorized one-time diagnostic of the other eight older hosts added three passes and five failures. Across all 19 measured older hosts: **13 PASS, 6 FAIL**. grid100, rotated100, and staggered100 remain SKIPPED.

The exact selected archive has 125 verified source files, SHA256 `ed4a61dcfebc13f1134630db096aad043d60e4846083b4a1384e871ee9ebb334`. All remain unchanged. Ten cold gates took 77.22 s on one CPU/AVX2 worker. Subsequent diagnostics used at most four one-thread workers. Fixed older seed 0; no seed sweep or candidate tuning.

The conditional-host extension exposes the existing lambda critic and noise stream to H’s mixup helper. A fixture proves identical forward values, gradients, penalty, and RNG state. The explicit pure-GAN overrides set AE reconstruction and unused-token hold weights to zero. Frozen architecture, budgets, measurement samples, metrics, thresholds, and sustained-gate rules are unchanged. Actual G/D LR is 0.0015, particle LR 0.003, Adam betas (0, 0.999), and discriminator input sigma 0.05 throughout all 19 measured hosts.

| Case | Strict sequence | Later diagnostic | Budget | Result detail |
| --- | --- | --- | ---: | --- |
| `trajectory` | PASS | — | 400 | MSE 0.002905043 |
| `mode_hold` | PASS | — | 1200 | 8 modes; HQ 0.999267578 |
| `residual_student` | PASS | — | 400 | MSE 0.002931164; success 1; wrong-pad 0 |
| `img_stripes2` | PASS | — | 600 | 2 modes; HQ 0.96875 |
| `img_bars4` | PASS | — | 600 | 4 modes; HQ 1 |
| `vector_overlap` | PASS | — | 1200 | SW1 0.055860723 |
| `img_blobs4` | PASS | — | 600 | 4 modes; HQ 1 |
| `img_intensity2` | PASS | — | 600 | 2 modes; HQ 0.96875 |
| `vector_unequal_mass` | PASS | — | 1200 | mass TV 0.041552730; covariance error 0.417076070 |
| `vector_unequal_width` | PASS | — | 1200 | mass TV 0.032226563; covariance error 0.307263948 |
| `unipolar` | FAIL | — | 400 | endpoint passes, but suffix 1/5; neutral hold 0.850326702 |
| `two_pole` | SKIPPED | FAIL | 80 | mean_abs 0.032636743 < 0.30 |
| `cover_leftover` | SKIPPED | FAIL | 800 | u_kept 0.4491; content 0.4419; pole errors 0.4156/0.4542 |
| `mid_scale_identity` | SKIPPED | FAIL | 800 | identity_at_0 0.806213874 < 0.85 |
| `vector_anisotropic` | SKIPPED | PASS | 1200 | SW1 0.058866644; covariance error 0.275090615 |
| `vector_two_broad` | SKIPPED | PASS | 1200 | SW1 0.040083604; covariance error 0.133632258 |
| `vector_spiral` | SKIPPED | PASS | 1600 | SW1 0.036653005; covariance error 0.069570340 |
| `unused_token_hold` | SKIPPED | FAIL | 200 | hold 0.729775608; concept_move 0.540403429; both require 0.85 |
| `ae_gan_hold` | SKIPPED | FAIL | 250 | reconstruction MSE 3.917919397 > 0.05; hold 0.002231841 passes |
| `grid100` | SKIPPED | — | 7000 | SKIPPED after older failures |
| `rotated100` | SKIPPED | — | 7000 | SKIPPED after older failures |
| `staggered100` | SKIPPED | — | 7000 | SKIPPED after older failures |

**Unipolar is a sustained-gate failure:** its final cover 0.922150949, off-caption 0.0000238003 and neutral hold 0.850326702 satisfy endpoint bounds, but its last passing run has only one check; the frozen minimum is five.

**Auxiliary-host blockers inferred from source:** [unused-token parameters](https://github.com/255BITS/ParticleGAN/blob/e2f168ddbd2d38f8f94fbb825f8a71d67dc8d960/benchmarks/locked_shared/hosts/unused_token_hold.py#L141), [embedding rule](https://github.com/255BITS/ParticleGAN/blob/e2f168ddbd2d38f8f94fbb825f8a71d67dc8d960/benchmarks/locked_shared/hosts/unused_token_hold.py#L147), [concept-only adversarial update](https://github.com/255BITS/ParticleGAN/blob/e2f168ddbd2d38f8f94fbb825f8a71d67dc8d960/benchmarks/locked_shared/hosts/unused_token_hold.py#L259), [hold loss](https://github.com/255BITS/ParticleGAN/blob/e2f168ddbd2d38f8f94fbb825f8a71d67dc8d960/benchmarks/locked_shared/hosts/unused_token_hold.py#L270), and [metrics](https://github.com/255BITS/ParticleGAN/blob/e2f168ddbd2d38f8f94fbb825f8a71d67dc8d960/benchmarks/locked_shared/hosts/unused_token_hold.py#L167) imply an invariant when hold_weight=0: shared and concept-slot parameters stay equal, and the unused correction stays zero. A concept_move of at least 0.85 therefore forces unused_hold at most 0.575, below its required 0.85. This applies to this preserved host and optimizer; critic tuning cannot remove that conflict.

The AE [encoder path](https://github.com/255BITS/ParticleGAN/blob/e2f168ddbd2d38f8f94fbb825f8a71d67dc8d960/benchmarks/locked_shared/hosts/ae_gan_hold.py#L210) feeds reconstruction only, while the [adversarial loss](https://github.com/255BITS/ParticleGAN/blob/e2f168ddbd2d38f8f94fbb825f8a71d67dc8d960/benchmarks/locked_shared/hosts/ae_gan_hold.py#L219) samples the prior directly. Removing reconstruction weight disconnects encoder training although the [frozen metric](https://github.com/255BITS/ParticleGAN/blob/e2f168ddbd2d38f8f94fbb825f8a71d67dc8d960/benchmarks/locked_shared/hosts/ae_gan_hold.py#L115) still evaluates the encoder. This is a missing training signal, not an observed instability of the unconditional generator. Its hold metric passes.

All 19 source/config/episode and optimizer/noise receipts independently regrade. Cold raw final draws regrade for trajectory/ring; restored frozen evaluation draws exactly reproduce every metric for the four image and three vector gates. `residual_student` has no checkpoint in the archived screen, so its evidence is the 24-check episode and full update/noise receipts. The original own-state continuation remains a prior reported FAIL (5 modes/HQ 0.209228516); this verification did not rerun it.

Evidence: [verification.json](verification.json), [cold parity](cold-parity.json), [cold evidence audit](cold-evidence-audit.json), [remaining audit](remaining-evidence-audit.json), [plumbing fixture](plumbing-audit.json), [cold artifacts](cold-replay/h_n05r06_mixup_c0p01_lr15/status.json), [first failure](remaining-replay/h_n05r06_mixup_c0p01_lr15/status.json), [diagnostic summary](diagnostics/summary.json). Exact commands are [run-cold.sh](run-cold.sh), [run-remaining-replay.sh](run-remaining-replay.sh), and [run-diagnostics.py](run-diagnostics.py). Tail `diagnostics.log` or individual candidate `run.log` files.

Two setup errors are retained: the supplied replay script omitted the exact frozen leading_profile.json dependency; then the unextended H helper rejected a conditional lambda before any optimizer step. Neither is counted as a GAN failure. All work is isolated here; no source-attempt/PR checkout edits or GitHub writes were performed.
