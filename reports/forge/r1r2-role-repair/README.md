# R1/R2 optimizer-role repair comparison

No whole recipe passes the original Tier 1 suite. All seven new recipes pass
their own clock audit but fail sustained Ring16. Momentum plus AMSGrad passes
5/6 ordinary tasks; the other role subsets also fail word acquisition.
D/G+E momentum with zero sampled-prior momentum is the strongest density lead:
best covariance error 0.904193 still exceeds the original 0.85 bound.

The [plan](PLAN.md), [manifest](manifest.json), [registration](registration.json)
and [independent preflight](preflight-review.json) freeze the comparison.
It follows the closed [five-track campaign](../r1r2-five-track-repair/README.md).
The public tasks, actual initialization, data law and consumed batches, positive
MoG sampling, capacity, update/evaluation budgets and original gates are unchanged.
No seed variants, longer training, retries or higher-tier work occurred.

## Whole global recipes

Bits are D / G+E / sampled-prior beta1: 1 means .5 and 0 means 0. All constant
recipes keep beta2=.99, paired R1/R2 coefficient .1 every update, LR=.0085,
learned prior and zero extra prior regularization/noise/EMA. The direct pole
fixture uses G betas; its coordinates and the unsampled token fixture have no
sampled-prior beta factor. AMSGrad is a global flag. The three cached controls
retain their exact original recipe, revision, source and task identity.

PASS/FAIL links illustrate each numerical gate through an actual-training GIF.
This is an unranked frozen comparison; the [technique inventory](../technique-inventory.md)
remains the single current generated leaderboard.

| Global recipe | Gaussian | Pole | Token | AE | Ring16 | Words | Clock | Ordinary |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| Cached scheduled baseline | [PASS](publication/media/baseline-gaussian1d_smoke.gif) | [PASS](publication/media/baseline-two_pole.gif) | [PASS](publication/media/baseline-unused_token_hold.gif) | [PASS](publication/media/baseline-ae_gan_hold.gif) | [FAIL](publication/media/baseline-ring16_acquisition.gif) | [PASS](publication/media/baseline-five_word_joint_smoke.gif) | [FAIL](publication/media/baseline-clockfree_audit_measurement_v1.gif) | 5/6 |
| Cached all-zero constant Adam | [PASS](publication/media/m000-gaussian1d_smoke.gif) | [PASS](publication/media/m000-two_pole.gif) | [PASS](publication/media/m000-unused_token_hold.gif) | [PASS](publication/media/m000-ae_gan_hold.gif) | [FAIL](publication/media/m000-ring16_acquisition.gif) | [FAIL](publication/media/m000-five_word_joint_smoke.gif) | [PASS](publication/media/m000-clockfree_audit_measurement_v1.gif) | 4/6 |
| Cached all-.5 Adam | [FAIL](publication/media/m111-gaussian1d_smoke.gif) | [PASS](publication/media/m111-two_pole.gif) | [PASS](publication/media/m111-unused_token_hold.gif) | [PASS](publication/media/m111-ae_gan_hold.gif) | [FAIL](publication/media/m111-ring16_acquisition.gif) | [FAIL](publication/media/m111-five_word_joint_smoke.gif) | [PASS](publication/media/m111-clockfree_audit_measurement_v1.gif) | 3/6 |
| m100 | [PASS](publication/media/m100-gaussian1d_smoke.gif) | [PASS](publication/media/m100-two_pole.gif) | [PASS](publication/media/m100-unused_token_hold.gif) | [PASS](publication/media/m100-ae_gan_hold.gif) | [FAIL](publication/media/m100-ring16_acquisition.gif) | [FAIL](publication/media/m100-five_word_joint_smoke.gif) | [PASS](publication/media/m100-clockfree_audit_measurement_v1.gif) | 4/6 |
| m010 | [PASS](publication/media/m010-gaussian1d_smoke.gif) | [PASS](publication/media/m010-two_pole.gif) | [PASS](publication/media/m010-unused_token_hold.gif) | [PASS](publication/media/m010-ae_gan_hold.gif) | [FAIL](publication/media/m010-ring16_acquisition.gif) | [FAIL](publication/media/m010-five_word_joint_smoke.gif) | [PASS](publication/media/m010-clockfree_audit_measurement_v1.gif) | 4/6 |
| m001 | [PASS](publication/media/m001-gaussian1d_smoke.gif) | [PASS](publication/media/m001-two_pole.gif) | [PASS](publication/media/m001-unused_token_hold.gif) | [PASS](publication/media/m001-ae_gan_hold.gif) | [FAIL](publication/media/m001-ring16_acquisition.gif) | [FAIL](publication/media/m001-five_word_joint_smoke.gif) | [PASS](publication/media/m001-clockfree_audit_measurement_v1.gif) | 4/6 |
| m110 | [FAIL](publication/media/m110-gaussian1d_smoke.gif) | [PASS](publication/media/m110-two_pole.gif) | [PASS](publication/media/m110-unused_token_hold.gif) | [PASS](publication/media/m110-ae_gan_hold.gif) | [FAIL](publication/media/m110-ring16_acquisition.gif) | [FAIL](publication/media/m110-five_word_joint_smoke.gif) | [PASS](publication/media/m110-clockfree_audit_measurement_v1.gif) | 3/6 |
| m101 | [PASS](publication/media/m101-gaussian1d_smoke.gif) | [PASS](publication/media/m101-two_pole.gif) | [PASS](publication/media/m101-unused_token_hold.gif) | [PASS](publication/media/m101-ae_gan_hold.gif) | [FAIL](publication/media/m101-ring16_acquisition.gif) | [FAIL](publication/media/m101-five_word_joint_smoke.gif) | [PASS](publication/media/m101-clockfree_audit_measurement_v1.gif) | 4/6 |
| m011 | [PASS](publication/media/m011-gaussian1d_smoke.gif) | [PASS](publication/media/m011-two_pole.gif) | [PASS](publication/media/m011-unused_token_hold.gif) | [PASS](publication/media/m011-ae_gan_hold.gif) | [FAIL](publication/media/m011-ring16_acquisition.gif) | [FAIL](publication/media/m011-five_word_joint_smoke.gif) | [PASS](publication/media/m011-clockfree_audit_measurement_v1.gif) | 4/6 |
| All-.5 Adam + AMSGrad | [PASS](publication/media/ams_m111-gaussian1d_smoke.gif) | [PASS](publication/media/ams_m111-two_pole.gif) | [PASS](publication/media/ams_m111-unused_token_hold.gif) | [PASS](publication/media/ams_m111-ae_gan_hold.gif) | [FAIL](publication/media/ams_m111-ring16_acquisition.gif) | [PASS](publication/media/ams_m111-five_word_joint_smoke.gif) | [PASS](publication/media/ams_m111-clockfree_audit_measurement_v1.gif) | 5/6 |

[Final metrics and provenance](publication/results.json) ·
[Independent collection audit](collection-audit.json) ·
[Saved density readout](density-readout.md) · [Density receipt](density-audit.json).

## Failure and recommendation

All 672 saved Ring16 observations reproduce the original six metrics exactly,
with no new draws or updates. Every new recipe has 0/96 covariance passes,
0/96 full passes and terminal suffix 0/5. m110's best observation passes all
other bounds, but its last five covariance errors remain above 0.85. Its
endpoint covariance 1.083997 and minimum eigen ratio .111723 both fail.
All component tangent variances remain 1.600–3.466 times target; radial variance
ranges .133–2.496. A uniform contraction cannot be inferred to repair both.
m110 also fails Gaussian and word acquisition, so its density improvement
cannot qualify a per-task mixture. Momentum plus AMSGrad repairs those
acquisition gates but has endpoint covariance 2.009387.

Close this finite campaign with every failure visible. Keep the original
scheduled baseline and its original clock/density failures. The role factorization
narrows a possible momentum lead but supplies no complete replacement.
The saved [gradient-direction analysis](optimizer/normalization.md) and
[critic-denominator analysis](prior/denominator-readme.md) support two distinct
next hypotheses on the exact zero-momentum constant parent: the existing
AdaNSGDA norm graft, and critic-only epsilon 1e-4. Their separate and joint
recipes require a new finite registration and every unchanged gate. These
endpoint diagnoses are motivation, not proof of causality or future passes.

## Evidence and accounting

Execution commit `6f634cec5c3ec6e0dcda7c1508a2eafaf9fb2bfd`; scientific source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`;
runtime `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
Independent audits verify candidate/job/source/runtime and saved-grade bindings,
actual optimizer beta/AMSGrad histories, identical captured initial tensors
and ordered target batches, and every consumed named RNG stream checkpoint.
The explicit zero/identity/direct-coordinate fixtures remain separate cohorts;
the clock receipts retain their own five-branch comparisons.

49 new runs cost 1,269.2766638240719 seconds. All 21 cached cells add zero
paid cost. Aggregate shared accounting preserves all 105 earlier charges and
now contains 154 unique attempts totaling 4,000.9795464819763 seconds.
Full registration reservation is 25,200 seconds and aggregate paid cap 8,000.
The combined archives retain the 8 GiB cap and 5 GiB free-space floor.
These costs are accounting, not a comparable-speed ranking.

All 70 final cells have saved metrics/provenance and actual-training GIFs;
collection adds zero optimizer updates or sampling draws. The
[validation receipt](validation.json) binds checks and report hashes. Forty-one
workflow checks pass, and tier-report publication checks preserve the original
qualification tables. No task/source arithmetic or public default changes.
Provisional screening calibration grants no default promotion.

Reproduce from the frozen [plan commands](PLAN.md), with read-only collection
against the retained original queue. Bulk stdout, events, checkpoints and tensors
stay outside Git. Coordinator/test logs are in
`/mnt/ml7tb/ParticleGAN-forge/r1r2-role-repair-20261012/`:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/r1r2-role-repair-20261012/queue/events.jsonl
```
