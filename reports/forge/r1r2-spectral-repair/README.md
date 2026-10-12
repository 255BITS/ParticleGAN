# R1/R2 spectral-update readout

**No complete Tier 1 repair.** Full and critic-only DualNorm each pass four of six ordinary tasks plus their own clock audit, and fail Ring16 and words. This finite comparison is closed. Keep the original baseline and failures; no selection, default promotion, Tier 2 or automatic refinement follows. The [technique inventory](../technique-inventory.md) remains the only current generated leaderboard.

This is a frozen comparison of one global recipe per row. Gaussian and word columns score independently confirmed acquisition; pole, token, AE and Ring retain their original sustained gates. A later endpoint does not replace acquisition evidence.

| Recipe | Gaussian | Pole | Token | AE | Ring16 | Words | Clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Scheduled reference | [PASS](publication/media/baseline-gaussian1d_smoke.gif) | [PASS](publication/media/baseline-two_pole.gif) | [PASS](publication/media/baseline-unused_token_hold.gif) | [PASS](publication/media/baseline-ae_gan_hold.gif) | [FAIL](publication/media/baseline-ring16_acquisition.gif) | [PASS](publication/media/baseline-five_word_joint_smoke.gif) | [FAIL](publication/media/baseline-clockfree_audit_measurement_v1.gif) |
| Constant Adam parent | [PASS](publication/media/adam-gaussian1d_smoke.gif) | [PASS](publication/media/adam-two_pole.gif) | [PASS](publication/media/adam-unused_token_hold.gif) | [PASS](publication/media/adam-ae_gan_hold.gif) | [FAIL](publication/media/adam-ring16_acquisition.gif) | [FAIL](publication/media/adam-five_word_joint_smoke.gif) | [PASS](publication/media/adam-clockfree_audit_measurement_v1.gif) |
| Cached AdaNSGDA | [PASS](publication/media/ada-gaussian1d_smoke.gif) | [PASS](publication/media/ada-two_pole.gif) | [PASS](publication/media/ada-unused_token_hold.gif) | [PASS](publication/media/ada-ae_gan_hold.gif) | [FAIL](publication/media/ada-ring16_acquisition.gif) | [FAIL](publication/media/ada-five_word_joint_smoke.gif) | [PASS](publication/media/ada-clockfree_audit_measurement_v1.gif) |
| Full DualNorm | [PASS](publication/media/spectral-gaussian1d_smoke.gif) | [PASS](publication/media/spectral-two_pole.gif) | [PASS](publication/media/spectral-unused_token_hold.gif) | [PASS](publication/media/spectral-ae_gan_hold.gif) | [FAIL](publication/media/spectral-ring16_acquisition.gif) | [FAIL](publication/media/spectral-five_word_joint_smoke.gif) | [PASS](publication/media/spectral-clockfree_audit_measurement_v1.gif) |
| Critic-only DualNorm | [PASS](publication/media/critic_spectral-gaussian1d_smoke.gif) | [PASS](publication/media/critic_spectral-two_pole.gif) | [PASS](publication/media/critic_spectral-unused_token_hold.gif) | [PASS](publication/media/critic_spectral-ae_gan_hold.gif) | [FAIL](publication/media/critic_spectral-ring16_acquisition.gif) | [FAIL](publication/media/critic_spectral-five_word_joint_smoke.gif) | [PASS](publication/media/critic_spectral-clockfree_audit_measurement_v1.gif) |

## Density remains the common blocker

| New global recipe | Best covariance error | Final covariance error | Final minimum eigen ratio | HQ |
| --- | ---: | ---: | ---: | ---: |
| Full DualNorm | 1.128805 at 1584 | 1.180543 | .288719 | .897461 |
| Critic-only DualNorm | 1.149246 at 1550 | 1.212486 | .144263 | .830322 |

Both recipes have zero covariance passes in 96 original evaluations and zero terminal full passes. The unchanged covariance bound is .85; the minimum eigen bound is .15. Full DualNorm retains all 16 modes and acceptable endpoint spread, but excessive covariance remains. Critic-only also fails endpoint minimum spread and HQ. Seeing all modes or a passing endpoint control does not satisfy sustained density.

The [saved-panel audit](density-readout.md) reconstructs all six original Ring gate metrics across 192 new evaluations. Its endpoint Jacobian geometry is a first-order diagnostic, not an exact nonlinear covariance decomposition or gate. [Prior geometry](prior/README.md) shows that removing mapped-center dispersion alone leaves a covariance or minimum-spread failure in the linear proxy. No distinct prior-policy change is supported by those saved endpoints.

Both new word runs have zero confirmed acquisitions: full DualNorm ends with four of five modes and failed reconstruction; critic-only ends with one mode. The Gaussian acquisition passes do not establish endpoint retention. Full versus critic-only bundles generator/encoder and learned-prior update changes, so this comparison cannot isolate generator motion alone.

## Mechanism, contracts and cost

The [frozen plan](PLAN.md), [registration](registration.json), [independent preflight](preflight-review.json) and [certified results](publication/results.json) bind the exact declarations. Only optimizer_family changes from the constant parent: full DualNorm uses polar G/E/D updates and sampled learned-prior row normalization; critic-only changes D and keeps native Adam G/E/P at .0085. Initial prior, capacity and sampling stay fixed; full prior update law is explicitly changed. Native direct pole remains generator-owned, and token has no sampled prior.

All tasks keep seed 0, public deterministic initialization, architecture, target/data law, actual initial tensors and ordered target batches, prior law/capacity, sampling, update budgets, evaluation cadence and gates. Every consumed named RNG stream is checkpointed. Full polar checkpoints retain step counters rather than raw gradients; sampled row IDs are cleared after each step. The audit checks required row ownership, bound source and RNG receipts, and does not claim to reconstruct every historical row set.

Execution commit: `220d37827e910c4f39c0d1c7afb1f337627c53ab`. Scientific source and runtime remain those in the registration. All 35 certified cases have actual-training GIFs; 21 exact cached control attempts cost zero new work. Fourteen new attempts cost **1,255.0448551499867 seconds**. The shared queue totals **189 unique attempts / 5,842.625037202932 seconds**, preserving all 175 earlier charges and their original owners. Full reservations remain 12,600 seconds and the aggregate paid cap remains 8,000. No scientific retries, seed changes, execution errors or shortened task allowances.

The [collection audit](collection-audit.json) independently verifies saved grades, checkpoints, actual optimizer laws, source/runtime, initial/data/RNG identities, caches, costs and media. Read-only collection adds no updates or draws. Bulk archive: `/mnt/ml7tb/ParticleGAN-forge/r1r2-spectral-repair-20261012`; raw logs, traces, tensor panels and checkpoints stay outside Git. [Validation receipt](validation.json) binds software and memory checks.

## Recommendation

Close the spectral comparison without adoption. The [global transport review](../r1r2-direction-floor-repair/transport/global-sliced-review.md) supports one from-initialization quantile-force hypothesis on the constant parent; the [optimizer memory review](optimizer/ams-role-note.md) supports testing global AMSGrad on exact m110, with weak density confidence. Neither is a measured repair. The separately bounded [objective/memory plan](../r1r2-objective-memory-repair/PLAN.md) keeps every task, gate and shared paid ceiling unchanged. Arbitrary smoothing scales, endpoint continuation and center-spread-only repair are unsupported by the saved diagnostics.
