# Forge experiments by tier

Current task assignments, grouped by goal view and qualification tier. Required tasks gate progression; ranking and diagnostic tasks retain their declared roles.

Catalog: **47 tasks**; **44 assigned** to at least one view; **3 unassigned**. Showing **6/6 views**.

Regenerate from the repository root with `python -m experiments.forge experiments-by-tier --output reports/forge/EXPERIMENTS_BY_TIER.md`. Add `--json` for machine-readable output (use a `.json` output path when saving). Regeneration reads declarations and launches no training.

Tier 1 is smoke, Tier 2 is quality, and Tier 3 is endurance. Views may leave later tiers empty. Placement follows each view's policy.

Steps and timeouts are declared per task, rather than measured costs. Tasks in an uninterrupted execution group share one run; their budgets must not be added together. Continuation rows distinguish total steps from additional or extension steps.

| View | Revision | Tier 1 | Tier 2 | Tier 3 | Declared calibration |
| --- | ---: | --- | --- | --- | --- |
| [adaptation](../../configs/forge/views/adaptation.json) | 2 | 3 required | 19 required | 1 required | provisional |
| [clockfree_continuous](../../configs/forge/views/clockfree_continuous.json) | 2 | 4 required | 19 required | 6 required | provisional |
| [discriminator_stability](../../configs/forge/views/discriminator_stability.json) | 2 | 3 required | 19 required | 2 required | provisional |
| [formulation_comparison](../../configs/forge/views/formulation_comparison.json) | 1 | 3 required | 19 required, 15 diagnostic | 2 required | provisional |
| [host_profile_transfer](../../configs/forge/views/host_profile_transfer.json) | 4 | 3 required | 19 required, 13 diagnostic | 2 required | provisional |
| [quality_coverage](../../configs/forge/views/quality_coverage.json) | 2 | 3 required | 19 required | 0 tasks | provisional |

## adaptation

Declaration: [adaptation](../../configs/forge/views/adaptation.json); revision 2; goal: `adaptation`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/adaptation.md).

### Tier 1: smoke

3 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

1 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json) | required | paired_adaptation / paired_adaptation | 3600 | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate) |

## clockfree_continuous

Declaration: [clockfree_continuous](../../configs/forge/views/clockfree_continuous.json); revision 2; goal: `clockfree_continuous`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/clockfree_continuous.md).

### Tier 1: smoke

4 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) | required | clockfree_audit / clockfree_parity | 24 | 300 | — |

### Tier 2: quality

19 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

6 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |
| [grid100_14k](../../configs/forge/tasks/grid100_14k.json) | required | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [grid100](../../configs/forge/tasks/grid100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [rotated100_14k](../../configs/forge/tasks/rotated100_14k.json) | required | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [rotated100](../../configs/forge/tasks/rotated100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [staggered100_14k](../../configs/forge/tasks/staggered100_14k.json) | required | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [staggered100](../../configs/forge/tasks/staggered100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json) | required | paired_adaptation / paired_adaptation | 3600 | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate) |

## discriminator_stability

Declaration: [discriminator_stability](../../configs/forge/views/discriminator_stability.json); revision 2; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/discriminator_stability.md).

### Tier 1: smoke

3 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

2 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

## formulation_comparison

Declaration: [formulation_comparison](../../configs/forge/views/formulation_comparison.json); revision 1; goal: `host_profile_transfer`.

Declared calibration status: **provisional**.

Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/formulation_comparison.md).

### Tier 1: smoke

3 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required, 15 diagnostic.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json) | diagnostic | transfer_image / transfer_sustained | 600 | 1800 | — |
| [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json) | diagnostic | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json) | diagnostic | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json) | diagnostic | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json) | diagnostic | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json) | diagnostic | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json) | diagnostic | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json) | diagnostic | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json) | diagnostic | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json) | diagnostic | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) | diagnostic | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) | diagnostic | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) | diagnostic | native100 / native_accuracy | 7000 | 3600 | — |
| [grid100_affine_paired_laws_v1](../../configs/forge/tasks/grid100_affine_paired_laws_v1.json) | diagnostic | native100 / native_accuracy | 7000 | 3600 | — |
| [grid100_release07_cloud_named_v1](../../configs/forge/tasks/grid100_release07_cloud_named_v1.json) | diagnostic | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

2 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

## host_profile_transfer

Declaration: [host_profile_transfer](../../configs/forge/views/host_profile_transfer.json); revision 4; goal: `host_profile_transfer`.

Declared calibration status: **provisional**.

Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/host_profile_transfer.md).

### Tier 1: smoke

3 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required, 13 diagnostic.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json) | diagnostic | transfer_image / transfer_sustained | 600 | 1800 | — |
| [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json) | diagnostic | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json) | diagnostic | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json) | diagnostic | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json) | diagnostic | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json) | diagnostic | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json) | diagnostic | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json) | diagnostic | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json) | diagnostic | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json) | diagnostic | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) | diagnostic | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) | diagnostic | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) | diagnostic | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

2 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

## quality_coverage

Declaration: [quality_coverage](../../configs/forge/views/quality_coverage.json); revision 2; goal: `quality_coverage`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/quality_coverage.md).

### Tier 1: smoke

3 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required.

| Task | Importance | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

0 tasks.

No tasks assigned.

## Tasks unassigned to any view

These catalog tasks have no tier placement. Add an assignment to a view to include them in its policy.

| Task | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- |
| [grid100_affine_square_named_v1_14k](../../configs/forge/tasks/grid100_affine_square_named_v1_14k.json) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [rotated100_affine_square_named_v1_14k](../../configs/forge/tasks/rotated100_affine_square_named_v1_14k.json) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [staggered100_affine_square_named_v1_14k](../../configs/forge/tasks/staggered100_affine_square_named_v1_14k.json) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |

Declaration input digest: `e51d29ceaef7c602528a7bd17b967c926712bffde41bf8e343fd67f5c57afd46`. The JSON form includes the individual task and view file hashes.
