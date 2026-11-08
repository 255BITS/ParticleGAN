# bispf-lazy-cadence-v1

One global BISPF public lazy policy at reg_every 4, 8 or 16 preserves every measured eager-control PASS across the unchanged tiered tasks, including held tests, with the base coefficient fixed and the built-in k-fold strength compensation. This measures coupled application-clock controller behavior, not isolated gradient frequency. Any matched PASS-to-FAIL falsifies preservation for that interval; unavailable evidence cannot pass. No seed study, per-task tuning, scientific default adoption or automatic qualification.

Stage: planned. No default adoption.

Reserved ceiling: 186480 seconds.

Public lazy policy including application-clock controllers; internal k-fold strength compensation. Base coefficient, optimizer updates and task protocols remain fixed.

New-paid cost is campaign-wide and shared across lane reports; never add it across lanes. Observed evidence cost retains its original attempt scope. Neither cost is a speed ranking.

## coverage

Cadence study bispf-lazy-cadence-v1-coverage compares reg_every against 1.

[The current technique inventory](../technique-inventory.md) remains the single leaderboard.
This comparison grants no qualification; diagnostic coverage supplies no ordinary qualification credit.

| Cadence | Tier | Required PASS / total | Held PASS / total | Diagnostic PASS / total | Lost PASS | Unavailable |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0/6 | 0/0 | 0/1 | 0 | 7 |
| 1 | 2 | 0/14 | 0/7 | 0/0 | 0 | 21 |
| 1 | 3 | 0/1 | 0/0 | 0/1 | 0 | 2 |
| 4 | 1 | 0/6 | 0/0 | 0/1 | 0 | 7 |
| 4 | 2 | 0/14 | 0/7 | 0/0 | 0 | 21 |
| 4 | 3 | 0/1 | 0/0 | 0/1 | 0 | 2 |
| 8 | 1 | 0/6 | 0/0 | 0/1 | 0 | 7 |
| 8 | 2 | 0/14 | 0/7 | 0/0 | 0 | 21 |
| 8 | 3 | 0/1 | 0/0 | 0/1 | 0 | 2 |
| 16 | 1 | 0/6 | 0/0 | 0/1 | 0 | 7 |
| 16 | 2 | 0/14 | 0/7 | 0/0 | 0 | 21 |
| 16 | 3 | 0/1 | 0/0 | 0/1 | 0 | 2 |

| Cadence | Task | Role | Control → candidate gate | Metric | Control | Candidate | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | ae_gan_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | clockfree_audit_measurement_v1 | diagnostic | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | five_word_joint_smoke | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | gaussian1d_smoke | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | ring16_acquisition | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | two_pole | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | unused_token_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | cover_leftover | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | five_word_joint_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | gaussian1d_stability | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | grid100 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | img_bars4 | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | img_blobs4 | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | img_intensity2 | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | img_stripes2 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | mid_scale_identity | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | mode_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | residual_student | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | rotated100 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | staggered100 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | trajectory | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | unipolar | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | vector_anisotropic | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | vector_overlap | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | vector_spiral | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | vector_two_broad | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | vector_unequal_mass | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | vector_unequal_width | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | ring_extension | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 4 | ring_hold | diagnostic | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | ae_gan_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | clockfree_audit_measurement_v1 | diagnostic | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | five_word_joint_smoke | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | gaussian1d_smoke | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | ring16_acquisition | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | two_pole | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | unused_token_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | cover_leftover | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | five_word_joint_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | gaussian1d_stability | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | grid100 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | img_bars4 | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | img_blobs4 | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | img_intensity2 | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | img_stripes2 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | mid_scale_identity | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | mode_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | residual_student | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | rotated100 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | staggered100 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | trajectory | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | unipolar | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | vector_anisotropic | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | vector_overlap | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | vector_spiral | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | vector_two_broad | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | vector_unequal_mass | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | vector_unequal_width | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | ring_extension | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 8 | ring_hold | diagnostic | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | ae_gan_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | clockfree_audit_measurement_v1 | diagnostic | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | five_word_joint_smoke | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | gaussian1d_smoke | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | ring16_acquisition | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | two_pole | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | unused_token_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | cover_leftover | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | five_word_joint_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | gaussian1d_stability | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | grid100 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | img_bars4 | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | img_blobs4 | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | img_intensity2 | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | img_stripes2 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | mid_scale_identity | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | mode_hold | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | residual_student | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | rotated100 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | staggered100 | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | trajectory | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | unipolar | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | vector_anisotropic | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | vector_overlap | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | vector_spiral | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | vector_two_broad | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | vector_unequal_mass | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | vector_unequal_width | held | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | ring_extension | required | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| 16 | ring_hold | diagnostic | UNKNOWN → UNKNOWN | UNAVAILABLE: candidate gate is UNKNOWN; control gate is UNKNOWN; no paired finite scalar metrics | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |

Recommendation: incomplete. Finish or repair unavailable matched evidence before selecting a cadence.

Cadence counters describe local audit windows, which may overlap across task rows; do not sum them.
Observed penalty calls can differ from controller applications when a host observes only some calls.

| Cadence | Task | Critic updates in window | Controller applications | Observed applied / skipped | First / last application update | Thermal min / max / last |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ae_gan_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | clockfree_audit_measurement_v1 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | five_word_joint_smoke | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | gaussian1d_smoke | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | ring16_acquisition | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | two_pole | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | unused_token_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | cover_leftover | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | five_word_joint_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | gaussian1d_stability | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | grid100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | img_bars4 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | img_blobs4 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | img_intensity2 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | img_stripes2 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | mid_scale_identity | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | mode_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | residual_student | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | rotated100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | staggered100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | trajectory | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | unipolar | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | vector_anisotropic | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | vector_overlap | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | vector_spiral | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | vector_two_broad | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | vector_unequal_mass | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | vector_unequal_width | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | ring_extension | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 1 | ring_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | ae_gan_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | clockfree_audit_measurement_v1 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | five_word_joint_smoke | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | gaussian1d_smoke | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | ring16_acquisition | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | two_pole | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | unused_token_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | cover_leftover | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | five_word_joint_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | gaussian1d_stability | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | grid100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | img_bars4 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | img_blobs4 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | img_intensity2 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | img_stripes2 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | mid_scale_identity | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | mode_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | residual_student | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | rotated100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | staggered100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | trajectory | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | unipolar | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | vector_anisotropic | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | vector_overlap | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | vector_spiral | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | vector_two_broad | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | vector_unequal_mass | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | vector_unequal_width | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | ring_extension | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 4 | ring_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | ae_gan_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | clockfree_audit_measurement_v1 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | five_word_joint_smoke | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | gaussian1d_smoke | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | ring16_acquisition | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | two_pole | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | unused_token_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | cover_leftover | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | five_word_joint_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | gaussian1d_stability | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | grid100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | img_bars4 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | img_blobs4 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | img_intensity2 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | img_stripes2 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | mid_scale_identity | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | mode_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | residual_student | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | rotated100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | staggered100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | trajectory | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | unipolar | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | vector_anisotropic | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | vector_overlap | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | vector_spiral | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | vector_two_broad | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | vector_unequal_mass | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | vector_unequal_width | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | ring_extension | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 8 | ring_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | ae_gan_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | clockfree_audit_measurement_v1 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | five_word_joint_smoke | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | gaussian1d_smoke | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | ring16_acquisition | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | two_pole | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | unused_token_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | cover_leftover | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | five_word_joint_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | gaussian1d_stability | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | grid100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | img_bars4 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | img_blobs4 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | img_intensity2 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | img_stripes2 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | mid_scale_identity | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | mode_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | residual_student | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | rotated100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | staggered100 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | trajectory | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | unipolar | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | vector_anisotropic | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | vector_overlap | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | vector_spiral | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | vector_two_broad | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | vector_unequal_mass | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | vector_unequal_width | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | ring_extension | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |
| 16 | ring_hold | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE | UNAVAILABLE / UNAVAILABLE / UNAVAILABLE |

| Cadence | Evidence worker seconds | New paid worker seconds |
| --- | --- | --- |
| 1 | 0 | UNAVAILABLE |
| 4 | 0 | UNAVAILABLE |
| 8 | 0 | UNAVAILABLE |
| 16 | 0 | UNAVAILABLE |

Frozen source: bb52868450b556a0457b6aaf484f777b14703f3aa22a60781fc36bfa5ea6b381.
Frozen protocol: 7b975bab8008c880a88e04d9e9bbdf4d1ab549d46d1d894b9e42369fdc0931b2.
Runtime and remaining provenance are retained in the companion JSON; worker wall time supplies no speed ranking.
Fixed-seed pass preservation supplies no statistical noninferiority claim.
Logs: /home/martyn/dev/ParticleGAN/runs/forge/events.jsonl.
