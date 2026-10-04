# Forge experiments by tier

Current task assignments, grouped by goal view and qualification tier. Required tasks gate progression; ranking and diagnostic tasks retain their declared roles.

Catalog: **49 tasks**; **46 assigned** to at least one view; **3 unassigned**. Showing **6/6 views**.

Declared priors across the catalog: **32 MoGParticlePrior**, **17 ParticlePrior** (including **3 nonsampled parameter controls**). Every experiment defines `execution.prior` explicitly; candidate and API defaults cannot supply it. `kind: mog` selects `MoGParticlePrior`; `kind: particle_cloud` selects `ParticlePrior`. Sigma alone does not identify the code path. Ordinary Forge MoG tasks require positive sigma; archived zero-sigma MoG evidence keeps its recorded kind. Task sigma is absolute; API demonstrations may instead record the recipe's relative `sigma_rel`.

Regenerate from the repository root with `python -m experiments.forge experiments-by-tier --output reports/forge/EXPERIMENTS_BY_TIER.md`. Add `--json` for machine-readable output (use a `.json` output path when saving). Regeneration reads declarations and published artifacts and launches no training.

Tier 1 is smoke, Tier 2 is quality, and Tier 3 is endurance. Views may leave later tiers empty. Placement follows each view's policy.

Steps and timeouts are declared per task, rather than measured costs. Tasks in an uninterrupted execution group share one run; their budgets must not be added together. Continuation rows distinguish total steps from additional or extension steps.

## Review experiments and solutions

Use the tier tables to see the current requirements, then open each task's experiment guide for its question, numerical gates, recorded configurations and related training GIFs.

- [Current solution leaderboard](technique-inventory.md): selected configurations and complete Forge tier denominators.
- [Research memory](EXPERIMENT_MEMORY.md): hypotheses, comparisons, failure explanations and recommendations.
- [All retained public-API questions](../toy_audit/api_contract/QUESTION_RANKING.md): the wider research catalog, explanations, results and goal GIFs.
- [Public-API experiment readout](../toy_audit/api_contract/RUN_REPORT.md): completed protocols and failed numerical bounds.
- [Public-API test contract](../toy_audit/api_contract/README.md): reproduction commands and exact scope of the demos.

Recorded Forge outcomes below are lookups by exact task ID from the current solution publication; their source revision and runtime remain explicit. Related API variants illustrate the question under their own gates, recipes, priors, initialization, budgets and sampling laws. Their PASS results do not qualify a different Forge task or the latest checkout. The solution leaderboard remains the single ranking for its goal.

This report follows changing declarations and published evidence; it selects no release winner. Release selection requires full Forge qualification and review of the winner. A smoke-qualified entry or historical pass alone does not establish release readiness.

## Current tier assignments

| View | Revision | Tier 1 | Tier 2 | Tier 3 | Declared calibration |
| --- | ---: | --- | --- | --- | --- |
| [adaptation](../../configs/forge/views/adaptation.json) | 2 | 3 required | 19 required | 1 required | provisional |
| [clockfree_continuous](../../configs/forge/views/clockfree_continuous.json) | 2 | 4 required | 19 required | 6 required | provisional |
| [discriminator_stability](../../configs/forge/views/discriminator_stability.json) | 3 | 5 required | 19 required | 2 required | provisional |
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

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

1 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | paired_adaptation / paired_adaptation | 3600 | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate) |

## clockfree_continuous

Declaration: [clockfree_continuous](../../configs/forge/views/clockfree_continuous.json); revision 2; goal: `clockfree_continuous`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/clockfree_continuous.md).

### Tier 1: smoke

4 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-clockfree-audit) | clockfree_audit / clockfree_parity | 24 | 300 | — |

### Tier 2: quality

19 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

6 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |
| [grid100_14k](../../configs/forge/tasks/grid100_14k.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-grid100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [grid100](../../configs/forge/tasks/grid100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [rotated100_14k](../../configs/forge/tasks/rotated100_14k.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-rotated100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [rotated100](../../configs/forge/tasks/rotated100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [staggered100_14k](../../configs/forge/tasks/staggered100_14k.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-staggered100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [staggered100](../../configs/forge/tasks/staggered100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | paired_adaptation / paired_adaptation | 3600 | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate) |

## discriminator_stability

Declaration: [discriminator_stability](../../configs/forge/views/discriminator_stability.json); revision 3; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Expanded five-task Tier 1 placement is provisional and requires a new bounded calibration; prior three-task profiles and revision 2 published qualification retain their original scope.

Candidate outcomes, metrics and measured costs: [leaderboard](technique-inventory.md).

### Tier 1: smoke

5 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-ring16-acquisition) | transfer_vector / transfer_sustained | 400 | 300 | — |
| [five_word_joint_acquisition](../../configs/forge/tasks/five_word_joint_acquisition.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / transfer_sustained | 20001 | 900 | — |

### Tier 2: quality

19 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

2 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

## formulation_comparison

Declaration: [formulation_comparison](../../configs/forge/views/formulation_comparison.json); revision 1; goal: `host_profile_transfer`.

Declared calibration status: **provisional**.

Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/formulation_comparison.md).

### Tier 1: smoke

3 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required, 15 diagnostic.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [grid100_affine_paired_laws_v1](../../configs/forge/tasks/grid100_affine_paired_laws_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [grid100_release07_cloud_named_v1](../../configs/forge/tasks/grid100_release07_cloud_named_v1.json) | diagnostic | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

2 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

## host_profile_transfer

Declaration: [host_profile_transfer](../../configs/forge/views/host_profile_transfer.json); revision 4; goal: `host_profile_transfer`.

Declared calibration status: **provisional**.

Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/host_profile_transfer.md).

### Tier 1: smoke

3 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required, 13 diagnostic.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

2 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

## quality_coverage

Declaration: [quality_coverage](../../configs/forge/views/quality_coverage.json); revision 2; goal: `quality_coverage`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/quality_coverage.md).

### Tier 1: smoke

3 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required.

| Task | Importance | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

0 tasks.

No tasks assigned.

## Tasks unassigned to any view

These catalog tasks have no tier placement. Add an assignment to a view to include them in its policy.

| Task | Prior code path | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- |
| [grid100_affine_square_named_v1_14k](../../configs/forge/tasks/grid100_affine_square_named_v1_14k.json) | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-grid100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [rotated100_affine_square_named_v1_14k](../../configs/forge/tasks/rotated100_affine_square_named_v1_14k.json) | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-rotated100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [staggered100_affine_square_named_v1_14k](../../configs/forge/tasks/staggered100_affine_square_named_v1_14k.json) | MoGParticlePrior (sigma=0.025) | [Question, results, GIFs](#experiment-staggered100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |

## Experiment guides

Task variants share a guide when their declarations name the same host or problem. This grouping is for navigation; it does not assert matching scientific contracts. Public-API variants join only through their explicit retained question IDs.

### Experiment: ae-gan-hold

Checks reconstruction/identity and an acquired adversarial edit during the declared hold.

Forge declarations: [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["recon_mse", "<=", 0.05], ["hold", "<=", 0.35]]
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "generated_and_reconstructed_prior_with_scheduled_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "public_recipe_schedule"

</details>

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| ae_gan_hold | [BCap](../../configs/forge/configurations/bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / bf859ff60898 | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [K3P](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / 7ebaeb278a75 | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [KA2](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / 7be3028bd4fc | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [R1/R2](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / 4ff453a8a3ed | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [GAN v3 release 0.7 (MoG)](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / f99998f9b0fa | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-ae-anchor-hold](../toy_audit/api_contract/media/api-ae-anchor-hold.gif) | Reconstruct the noisy two-anchor inputs while the independently sampled prior covers both anchors Scope: New public Recipe.encode particle AE with reconstruction weight1 and applied GAN/KA2; independently sampled prior quality is separately scored. | MoGParticlePrior (sigma_rel=0.025) | COMPLETE / FAIL; 250/250 updates; quality_fraction, quality_mass_tv, last 5 post-update metric observations do not all pass | ae_gan / cpu / 3efb003d4c86 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: clockfree-audit

Check saved public trainer state under step_label, horizon, evaluation_cadence, restart perturbations.

Forge declarations: [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[clockfree_audit](../../configs/forge/tasks/clockfree_audit.json)

- **kind**: "clockfree_parity"
- **conditions**: ["step_label", "horizon", "evaluation_cadence", "restart"]
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: cover-leftover

Checks target coverage plus the separate unwanted-remainder/content constraints.

Forge declarations: [cover_leftover](../../configs/forge/tasks/cover_leftover.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[cover_leftover](../../configs/forge/tasks/cover_leftover.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["u_kept", ">=", 0.85], ["content_kept", ">=", 0.75], ["leak_ratio", "<=", 0.2], ["pole_rel_err_plus", "<=", 0.2], ["pole_rel_err_minus", "<=", 0.2], ["same_dir", "<=", 0.25]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "This host uses two explicit learned clouds and separately declared host jitter; no implicit MoG kernel may replace that law.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "learned_parameter_measurement"
- **scoring_weights**: "live"
- **eval_output_noise**: "not_applied_to_measurement"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-guarded-leftover](../toy_audit/api_contract/media/api-guarded-leftover.gif) | Cover both signed poles while preserving content and removing the guarded leak Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 800/800 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: five-word-joint

Can a joint BiGAN generator, encoder and critic acquire five equally likely canonical words and reconstruct every correctly paired input with confident token probabilities, including padding?

Explanation, interpretation and reproduction: [experiment readout](five-word-joint/README.md).

Forge declarations: [five_word_joint_acquisition](../../configs/forge/tasks/five_word_joint_acquisition.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[five_word_joint_acquisition](../../configs/forge/tasks/five_word_joint_acquisition.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["sample_count", ">=", 1024], ["quality_fraction", ">=", 0.95], ["modes", "==", 5], ["mass_tv", "<=", 0.1], ["reconstruction_exact", "==", 1], ["minimum_reconstruction_token_probability", ">=", 0.9]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "Retained finite-vocabulary joint BiGAN question explicitly uses five learned 2D rows, one possible code per word; a MoG-width question needs its own protocol.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "generated_and_paired_reconstructed_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| five_word_joint_acquisition | [K3P](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | ParticlePrior (sigma=0) | FAIL | CHANGED; earlier contract | cuda / 2899099048c0 / 7ebaeb278a75 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_acquisition | [KA2](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | ParticlePrior (sigma=0) | FAIL | CHANGED; earlier contract | cuda / 2899099048c0 / 7be3028bd4fc | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [forge-five-word-joint-api-demo-v1](../toy_audit/api_contract/five_word_joint/goal.gif) | Can a joint BiGAN generator, encoder and critic acquire five equally likely canonical words and reconstruct every correctly paired input with confident token probabilities, including padding? Scope: One bounded shared-API integration demonstration, not an ordinary Forge run or release qualification. Score the declared bounds honestly at 32 updates and grade the evidence INCOMPLETE against the 20,001-update task. | ParticlePrior (sigma=0) | COMPLETE / INCOMPLETE; 32/20001 updates; quality_fraction, modes, mass_tv, reconstruction_exact, minimum_reconstruction_token_probability | ka2 / cpu / 997c7f01b99a | [definition](../toy_audit/api_contract/five_word_joint/publication.json); [readout](../toy_audit/api_contract/five_word_joint/publication.json); [recipe and provenance](../toy_audit/api_contract/five_word_joint/publication.json) |
| [image-five-words-joint-ae](../toy_audit/api_contract/media/image-five-words-joint-ae.gif) | Generate the five equally likely canonical words with confident normalized token probabilities, and reconstruct each of the five matched inputs including underscore padding. Scope: Finite vocabulary apple/grape/lemon/melon/berry only. Joint BiGAN inverse reconstruction; no unseen words or natural-language generation. New API-policy variant, not reuse of historical EMA PASS. | ParticlePrior (sigma=0) | COMPLETE / PASS; 20001/20001 updates | ka2 / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: grid100

Recover all 100 Gaussian components, balanced mass, centers and within-mode covariance/radial spread; distinguish clean from noisy served laws.

Forge declarations: [grid100](../../configs/forge/tasks/grid100.json), [grid100_14k](../../configs/forge/tasks/grid100_14k.json), [grid100_affine_paired_laws_v1](../../configs/forge/tasks/grid100_affine_paired_laws_v1.json), [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json), [grid100_affine_square_named_v1_14k](../../configs/forge/tasks/grid100_affine_square_named_v1_14k.json), [grid100_release07_cloud_named_v1](../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[grid100](../../configs/forge/tasks/grid100.json), [grid100_14k](../../configs/forge/tasks/grid100_14k.json), [grid100_affine_paired_laws_v1](../../configs/forge/tasks/grid100_affine_paired_laws_v1.json), [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json), [grid100_affine_square_named_v1_14k](../../configs/forge/tasks/grid100_affine_square_named_v1_14k.json)

- **kind**: "native_accuracy"
- **coverage_thresholds**: {"all_finite": true, "max_cov_eig_ratio": 1.7, "max_mass_tv": 0.1, "max_mode_mass": 0.02, "max_radial_median_ratio": 1.4, "min_cov_eig_ratio": 0.4, "min_hq_mode_mass": 0.005, "min_modes": 100, "min_precision": 0.97, "min_radial_median_ratio": 0.65, "min_samples": 20000}
- **accuracy_limits**: {"abs_cov_trace_bias": 0.1, "center_rms_sigma": 0.2, "mass_tv": 0.06, "radial_ks": 0.04}
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

[grid100_release07_cloud_named_v1](../../configs/forge/tasks/grid100_release07_cloud_named_v1.json)

- **kind**: "native_accuracy"
- **coverage_thresholds**: {"all_finite": true, "max_cov_eig_ratio": 1.7, "max_mass_tv": 0.1, "max_mode_mass": 0.02, "max_radial_median_ratio": 1.4, "min_cov_eig_ratio": 0.4, "min_hq_mode_mass": 0.005, "min_modes": 100, "min_precision": 0.97, "min_radial_median_ratio": 0.65, "min_samples": 20000}
- **accuracy_limits**: {"abs_cov_trace_bias": 0.1, "center_rms_sigma": 0.2, "mass_tv": 0.06, "radial_ks": 0.04}
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "Faithful v0.7.0 GAN v3 public ParticlePrior law; sigma_rel and standardize are inert on its particle branch. Separate diagnostic, no MoG qualification.", "kind": "particle_cloud", "learnable": true, "sigma": 0, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-grid100](../toy_audit/api_contract/media/api-grid100.gif) | Recover all 100 equal-weight Gaussian modes, their mass and local width, including independent density-fidelity bounds. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / PASS; 7000/7000 updates | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-bars4

Healthy location transfer: four horizontal/vertical bar positions test spatial coverage.

Forge declarations: [img_bars4](../../configs/forge/tasks/img_bars4.json), [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[img_bars4](../../configs/forge/tasks/img_bars4.json), [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["modes", ">=", 4], ["hq", ">=", 0.9]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "Frozen image evaluation enumerates finite learned centers; stochastic MoG sampling requires separately calibrated measurement.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "enumerated_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_bars4-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_bars4-residual_upsample16.gif) | Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. Scope: Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 600/600 updates; distribution_tv, finite_template_tv, last 5 post-update metric observations do not all pass | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_bars4-source-transpose12](../toy_audit/api_contract/media/image-develop-img_bars4-source-transpose12.gif) | Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. Scope: Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 600/600 updates; distribution_tv, finite_template_tv, modes, last 5 post-update metric observations do not all pass | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-blobs4

Healthy location transfer: four small corner patches test localized quality and coverage.

Forge declarations: [img_blobs4](../../configs/forge/tasks/img_blobs4.json), [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[img_blobs4](../../configs/forge/tasks/img_blobs4.json), [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["modes", ">=", 4], ["hq", ">=", 0.9]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "Frozen image evaluation enumerates finite learned centers; stochastic MoG sampling requires separately calibrated measurement.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "enumerated_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_blobs4-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_blobs4-residual_upsample16.gif) | Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. Scope: Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_blobs4-source-transpose12](../toy_audit/api_contract/media/image-develop-img_blobs4-source-transpose12.gif) | Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. Scope: Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 600/600 updates; distribution_tv, finite_template_tv, hq, modes, last 5 post-update metric observations do not all pass | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-intensity2

Healthy photometric transfer: two patch intensities require intensity fidelity as well as support coverage.

Forge declarations: [img_intensity2](../../configs/forge/tasks/img_intensity2.json), [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[img_intensity2](../../configs/forge/tasks/img_intensity2.json), [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["modes", ">=", 2], ["hq", ">=", 0.9]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "Frozen image evaluation enumerates finite learned centers; stochastic MoG sampling requires separately calibrated measurement.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "enumerated_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_intensity2-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_intensity2-residual_upsample16.gif) | Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. Scope: Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_intensity2-source-transpose12](../toy_audit/api_contract/media/image-develop-img_intensity2-source-transpose12.gif) | Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. Scope: Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-stripes2

Healthy orientation transfer: two distinct stripe orientations with an adequately sized convolutional GAN.

Forge declarations: [img_stripes2](../../configs/forge/tasks/img_stripes2.json), [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[img_stripes2](../../configs/forge/tasks/img_stripes2.json), [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["modes", ">=", 2], ["hq", ">=", 0.9]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "Frozen image evaluation enumerates finite learned centers; stochastic MoG sampling requires separately calibrated measurement.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "enumerated_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_stripes2-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_stripes2-residual_upsample16.gif) | Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. Scope: Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_stripes2-source-transpose12](../toy_audit/api_contract/media/image-develop-img_stripes2-source-transpose12.gif) | Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. Scope: Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: mid-scale-identity

Checks identity preservation and target edit magnitude at intermediate control strength.

Forge declarations: [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["concept_cos_plus", ">=", 0.85], ["concept_cos_minus", ">=", 0.85], ["concept_mag_plus", ">=", 0.75], ["concept_mag_plus", "<=", 1.25], ["concept_mag_minus", ">=", 0.75], ["concept_mag_minus", "<=", 1.25], ["identity_at_0", ">=", 0.85], ["identity_at_mid", ">=", 0.85]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "Deterministic scale-conditioned residual controls are parameter clouds, with no sampled latent prior.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "learned_parameter_measurement"
- **scoring_weights**: "live"
- **eval_output_noise**: "not_applied_to_measurement"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-midscale-identity](../toy_audit/api_contract/media/api-midscale-identity.gif) | Retain identity at half strength in addition to correct neutral and signed poles Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 800/800 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: mode-hold

Checks all eight ring modes and HQ through the sampled terminal hold; not within-mode density fidelity.

Forge declarations: [mode_hold](../../configs/forge/tasks/mode_hold.json), [ring_extension](../../configs/forge/tasks/ring_extension.json), [ring_hold](../../configs/forge/tasks/ring_hold.json), [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[mode_hold](../../configs/forge/tasks/mode_hold.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["modes", ">=", 8], ["hq", ">=", 0.9]]
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

[ring_extension](../../configs/forge/tasks/ring_extension.json)

- **kind**: "ring_extension"
- **thresholds**: [["modes", "==", 8], ["hq", ">=", 0.9], ["hq", "<=", 1.0]]
- **confirmation_checks**: 200
- **hold_budget**: 1200
- **extension_steps**: 300
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

[ring_hold](../../configs/forge/tasks/ring_hold.json)

- **kind**: "ring_hold"
- **thresholds**: [["modes", "==", 8], ["hq", ">=", 0.9], ["hq", "<=", 1.0]]
- **confirmation_checks**: 200
- **hold_budget**: 1200
- **extension_steps**: 300
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

[target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json)

- **kind**: "paired_adaptation"
- **recovery_deadline**: 400
- **stationary_checks**: 5
- **deadline_checks**: 81
- **minimum_frozen_passing**: 0
- **requires_pair_artifacts**: true
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-ring8-acquire](../toy_audit/api_contract/media/api-ring8-acquire.gif) | Acquire all eight equal-weight radius-three, sigma-.07 Gaussian modes, including their within-mode law. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; hq >= 0.9, max_cov_eigen <= 1.5, max_radial_ks <= 0.1, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [api-ring8-hold](../toy_audit/api_contract/media/api-ring8-hold-failure.gif) | Acquire by update1200, then retain the same full eight-mode law without resetting optimizer state through update2400. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | ERROR / FAIL; 1200/2400 updates; API execution or metric error: RuntimeError: scientific prerequisite failed at update1200: ['hq >= 0.9', 'max_cov_eigen <= 1.5', 'max_radial_ks <= 0.1'] | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/failed-runs.json) |
| [api-ring8-resolution12](../toy_audit/api_contract/media/api-ring8-resolution12.gif) | Retain the original 12-row resource as a low-resource public-API control; assess the actual perturbed served law. A separate unperturbed twelve-equal-atom witness has a mass/width obstruction, which does not prove this stochastic served law impossible. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; hq >= 0.9, mass_tv <= 0.075, max_cov_eigen <= 1.5, max_radial_ks <= 0.1, min_cov_eigen >= 0.5, modes == 8, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [api-ring8-shift](../toy_audit/api_contract/media/api-ring8-shift-failure.gif) | After qualified acquisition/hold, adapt to a +1 x translation at update2401; require recovery by update2800 and retained width/mass through3600. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | ERROR / FAIL; 1200/3600 updates; API execution or metric error: RuntimeError: scientific prerequisite failed at update1200: ['hq >= 0.9', 'max_cov_eigen <= 1.5', 'max_radial_ks <= 0.1'] | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/failed-runs.json) |

### Experiment: residual-student

Checks whether the intended residual moves toward the correct paired target.

Forge declarations: [residual_student](../../configs/forge/tasks/residual_student.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[residual_student](../../configs/forge/tasks/residual_student.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["identity_mse", "<=", 0.02], ["success_rate", ">=", 1.0], ["wrong_pad_rate", "<=", 0.0]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "The conditional residual objective directly enumerates prior.z; stochastic latent sampling needs a separately calibrated task.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "conditional_prior_centers_with_scheduled_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "public_recipe_schedule"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-trajectory-residual](../toy_audit/api_contract/media/api-trajectory-residual.gif) | Recover the fast trajectory with a residual head and reject a correct marginal with wrong identities Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 400/400 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: ring16-acquisition

Acquire all 16 equally weighted two-dimensional Gaussian clusters from scratch: radius 3, sigma 0.1; require meaningful occupancy in every cluster, roughly balanced mass and noncollapsed local spread within 400 updates. No extended hold phase.

Forge declarations: [ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["sample_count", ">=", 4096], ["modes", ">=", 16], ["mass_tv", "<=", 0.15], ["hq", ">=", 0.85], ["component_covariance_error", "<=", 0.85], ["component_min_eigen_ratio", ">=", 0.15]]
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| ring16_acquisition | [BCap](../../configs/forge/configurations/bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212.json) | MoGParticlePrior (sigma=0.025) | FAIL | CHANGED; earlier contract | cuda / 2899099048c0 / bf859ff60898 | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [K3P](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / 7ebaeb278a75 | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [KA2](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / 7be3028bd4fc | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [R1/R2](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json) | MoGParticlePrior (sigma=0.025) | FAIL | CHANGED; earlier contract | cuda / 2899099048c0 / 4ff453a8a3ed | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [GAN v3 release 0.7 (MoG)](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json) | MoGParticlePrior (sigma=0.025) | FAIL | CHANGED; earlier contract | cuda / 2899099048c0 / f99998f9b0fa | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-ring16-acquisition](../toy_audit/api_contract/ring16/goal.gif) | Acquire all 16 equally weighted two-dimensional Gaussian clusters from scratch: radius 3, sigma 0.1; require meaningful occupancy in every cluster, roughly balanced mass and noncollapsed local spread within 400 updates. No extended hold phase. Scope: From random initialization, acquire sixteen equal radius-three sigma-.1 Gaussian clusters within 400 updates. Five terminal acquisition checks add no hold phase. Tier 1 is provisional; this standalone K3P/API initializer and RNG cohort supplies no Forge promotion credit. | MoGParticlePrior (sigma_rel=0) | ERROR / FAIL; 400/400 updates; component_covariance_error <= 0.85, hq >= 0.85, last 5 post-update metric observations do not all pass, goal media/state error: ModuleNotFoundError: No module named 'matplotlib' | k3p / cuda:0 / 746146cbdc5a | [definition](../toy_audit/api_contract/ring16/publication.json); [readout](../toy_audit/api_contract/ring16/publication.json); [recipe and provenance](../toy_audit/api_contract/ring16/publication.json) |

### Experiment: rotated100

Recover all 100 Gaussian components, balanced mass, centers and within-mode covariance/radial spread; distinguish clean from noisy served laws.

Forge declarations: [rotated100](../../configs/forge/tasks/rotated100.json), [rotated100_14k](../../configs/forge/tasks/rotated100_14k.json), [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json), [rotated100_affine_square_named_v1_14k](../../configs/forge/tasks/rotated100_affine_square_named_v1_14k.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[rotated100](../../configs/forge/tasks/rotated100.json), [rotated100_14k](../../configs/forge/tasks/rotated100_14k.json), [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json), [rotated100_affine_square_named_v1_14k](../../configs/forge/tasks/rotated100_affine_square_named_v1_14k.json)

- **kind**: "native_accuracy"
- **coverage_thresholds**: {"all_finite": true, "max_cov_eig_ratio": 1.7, "max_mass_tv": 0.1, "max_mode_mass": 0.02, "max_radial_median_ratio": 1.4, "min_cov_eig_ratio": 0.4, "min_hq_mode_mass": 0.005, "min_modes": 100, "min_precision": 0.97, "min_radial_median_ratio": 0.65, "min_samples": 20000}
- **accuracy_limits**: {"abs_cov_trace_bias": 0.1, "center_rms_sigma": 0.2, "mass_tv": 0.06, "radial_ks": 0.04}
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-rotated100](../toy_audit/api_contract/media/api-rotated100.gif) | Recover all 100 equal-weight Gaussian modes, their mass and local width, including independent density-fidelity bounds. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / PASS; 7000/7000 updates | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: staggered100

Recover all 100 Gaussian components, balanced mass, centers and within-mode covariance/radial spread; distinguish clean from noisy served laws.

Forge declarations: [staggered100](../../configs/forge/tasks/staggered100.json), [staggered100_14k](../../configs/forge/tasks/staggered100_14k.json), [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json), [staggered100_affine_square_named_v1_14k](../../configs/forge/tasks/staggered100_affine_square_named_v1_14k.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[staggered100](../../configs/forge/tasks/staggered100.json), [staggered100_14k](../../configs/forge/tasks/staggered100_14k.json), [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json), [staggered100_affine_square_named_v1_14k](../../configs/forge/tasks/staggered100_affine_square_named_v1_14k.json)

- **kind**: "native_accuracy"
- **coverage_thresholds**: {"all_finite": true, "max_cov_eig_ratio": 1.7, "max_mass_tv": 0.1, "max_mode_mass": 0.02, "max_radial_median_ratio": 1.4, "min_cov_eig_ratio": 0.4, "min_hq_mode_mass": 0.005, "min_modes": 100, "min_precision": 0.97, "min_radial_median_ratio": 0.65, "min_samples": 20000}
- **accuracy_limits**: {"abs_cov_trace_bias": 0.1, "center_rms_sigma": 0.2, "mass_tv": 0.06, "radial_ks": 0.04}
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-staggered100](../toy_audit/api_contract/media/api-staggered100.gif) | Recover all 100 equal-weight Gaussian modes, their mass and local width, including independent density-fidelity bounds. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / PASS; 7000/7000 updates | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: trajectory

Checks the extracted trajectory edit while preserving identity in finite paired rows.

Forge declarations: [trajectory](../../configs/forge/tasks/trajectory.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[trajectory](../../configs/forge/tasks/trajectory.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["identity_mse", "<=", 0.02]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "The conditional host enumerates prior.z in its identity objective; a stochastic latent read changes the frozen host law.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "conditional_prior_centers_with_scheduled_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "public_recipe_schedule"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-trajectory-edit](../toy_audit/api_contract/media/api-trajectory-edit.gif) | Change angular speed while preserving each trajectory's radius and starting phase Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 400/400 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: two-pole

Checks nonzero travel and bounded median critic gradient; its gate does not require both poles or a correct distribution.

Forge declarations: [two_pole](../../configs/forge/tasks/two_pole.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[two_pole](../../configs/forge/tasks/two_pole.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["mean_abs", ">=", 0.3], ["grad_med", "<=", 1.0]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "This host optimizes explicit sample-particle coordinates directly, without a separate generator or latent sampling kernel.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "learned_particles_and_critic_gradient"
- **scoring_weights**: "live"
- **eval_output_noise**: "not_applied_to_measurement"

</details>

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| two_pole | [BCap](../../configs/forge/configurations/bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / bf859ff60898 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [K3P](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / 7ebaeb278a75 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [K3P without A2](../../configs/forge/ideas/k3p-a2-off-native-diagnostic.json) | ParticlePrior (sigma=0) | FAIL | CHANGED; earlier contract | cuda / 2899099048c0 / 31b4d36cc007 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [K3P without critic anchor](../../configs/forge/ideas/forge-onboarding-anchor-ablation.json) | ParticlePrior (sigma=0) | FAIL | CHANGED; earlier contract | cuda / 2899099048c0 / 08a552799580 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [K3P without critic penalty](../../configs/forge/ideas/forge-no-critic-penalty.json) | ParticlePrior (sigma=0) | FAIL | CHANGED; earlier contract | cuda / 2899099048c0 / 866cef3392e0 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [K3P without training output noise](../../configs/forge/ideas/k3p-no-output-noise-diagnostic.json) | ParticlePrior (sigma=0) | FAIL | CHANGED; earlier contract | cuda / 2899099048c0 / edc93bce30da | [source-bound receipt index](technique-inventory.json) |
| two_pole | [KA2](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / 7be3028bd4fc | [source-bound receipt index](technique-inventory.json) |
| two_pole | [R1/R2](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / 4ff453a8a3ed | [source-bound receipt index](technique-inventory.json) |
| two_pole | [GAN v3 release 0.7 (MoG)](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / f99998f9b0fa | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-two-pole-grid12](../toy_audit/api_contract/media/api-two-pole-grid12.gif) | Check six target offsets per pole in one full row-ID realization, beyond mere travel; full served output-law fidelity remains unmeasured when latent perturbation is active. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 80/80 updates; max_grid_quantile_error_halfwidth <= 0.1, support_fraction >= 0.95, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: unipolar

Checks an intended edit with preservation of unrelated content.

Forge declarations: [unipolar](../../configs/forge/tasks/unipolar.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[unipolar](../../configs/forge/tasks/unipolar.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["cover", ">=", 0.85], ["off_caption", "<=", 0.05], ["neu_hold", ">=", 0.85]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "Deterministic student residual controls are parameter clouds, not draws from a sampled latent prior.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "learned_parameter_measurement"
- **scoring_weights**: "live"
- **eval_output_noise**: "not_applied_to_measurement"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-unipolar-hold](../toy_audit/api_contract/media/api-unipolar-hold.gif) | Make the positive 4D edit while holding the free scale-zero origin Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 400/400 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: unused-token-hold

Checks that active controls move and unused controls remain unchanged.

Forge declarations: [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[unused_token_hold](../../configs/forge/tasks/unused_token_hold.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["unused_hold", ">=", 0.85], ["concept_move", ">=", 0.85]]
- **minimum_stable_checks**: 5
- **prior**: {"exception_reason": "Deterministic embedding controls are parameter clouds; there is no sampled latent prior or separate prior optimizer.", "kind": "particle_cloud", "learnable": true, "sigma": 0.0, "standardize": false}
- **sampling_law**: "learned_parameter_measurement"
- **scoring_weights**: "live"
- **eval_output_noise**: "not_applied_to_measurement"

</details>

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| unused_token_hold | [BCap](../../configs/forge/configurations/bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / bf859ff60898 | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [K3P](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / 7ebaeb278a75 | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [KA2](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / 7be3028bd4fc | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [R1/R2](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / 4ff453a8a3ed | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [GAN v3 release 0.7 (MoG)](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / 2899099048c0 / f99998f9b0fa | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-unused-token-hold](../toy_audit/api_contract/media/api-unused-token-hold.gif) | Move the concept slot on its target axis while keeping the unused slot fixed Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / FAIL; 200/200 updates; concept_move, last 5 post-update metric observations do not all pass | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-anisotropic

Checks covariance shape: a narrow axis cannot be rescued by a wide one.

Forge declarations: [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json), [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json), [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["sw1_normalized", "<=", 0.18], ["mass_tv", "<=", 0.15], ["hq", ">=", 0.85], ["component_covariance_error", "<=", 0.85], ["component_min_eigen_ratio", ">=", 0.15]]
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-anisotropic](../toy_audit/api_contract/media/api-vector-anisotropic.gif) | Checks covariance shape: a narrow axis cannot be rescued by a wide one. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, resolved_max_component_spill <= 0.05, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-overlap

Scores the observable distribution when latent components are not identifiable.

Forge declarations: [vector_overlap](../../configs/forge/tasks/vector_overlap.json), [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[vector_overlap](../../configs/forge/tasks/vector_overlap.json), [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["sw1_normalized", "<=", 0.18], ["mean_error", "<=", 0.15], ["covariance_error", "<=", 0.45]]
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-overlap](../toy_audit/api_contract/media/api-vector-overlap.gif) | Scores the observable distribution when latent components are not identifiable. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-spiral

Checks continuous curved mass rather than a finite list of target mode centers.

Forge declarations: [vector_spiral](../../configs/forge/tasks/vector_spiral.json), [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[vector_spiral](../../configs/forge/tasks/vector_spiral.json), [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["sw1_normalized", "<=", 0.18], ["mean_error", "<=", 0.15], ["covariance_error", "<=", 0.45]]
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-spiral](../toy_audit/api_contract/media/api-vector-spiral.gif) | Checks continuous curved mass rather than a finite list of target mode centers. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1600/1600 updates; last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-two-broad

Basic learnable multimodal distribution and within-mode spread.

Forge declarations: [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json), [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[vector_two_broad](../../configs/forge/tasks/vector_two_broad.json), [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["sw1_normalized", "<=", 0.18], ["mass_tv", "<=", 0.15], ["hq", ">=", 0.85], ["component_covariance_error", "<=", 0.85], ["component_min_eigen_ratio", ">=", 0.15]]
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-two-broad](../toy_audit/api_contract/media/api-vector-two-broad.gif) | Basic learnable multimodal distribution and within-mode spread. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-unequal-mass

Checks target occupancy including the rare 2% component, not uniformity.

Forge declarations: [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json), [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json), [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["sw1_normalized", "<=", 0.18], ["mass_tv", "<=", 0.15], ["hq", ">=", 0.85], ["component_covariance_error", "<=", 0.85], ["component_min_eigen_ratio", ">=", 0.15], ["min_mass_ratio", ">=", 0.25]]
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-unequal-mass](../toy_audit/api_contract/media/api-vector-unequal-mass.gif) | Checks target occupancy including the rare 2% component, not uniformity. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-unequal-width

Checks component-specific scales without imposing one shared Gaussian width.

Forge declarations: [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json), [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json).

<details>
<summary>Declared Forge numerical gates and sampling</summary>

[vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json), [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json)

- **kind**: "transfer_sustained"
- **thresholds**: [["sw1_normalized", "<=", 0.18], ["mass_tv", "<=", 0.15], ["hq", ">=", 0.85], ["component_covariance_error", "<=", 0.85], ["component_min_eigen_ratio", ">=", 0.15]]
- **minimum_stable_checks**: 5
- **prior**: {"kind": "mog", "learnable": true, "sigma": 0.025, "standardize": false}
- **sampling_law**: "public_prior_without_output_noise"
- **scoring_weights**: "live"
- **eval_output_noise**: "clean"

</details>

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-unequal-width](../toy_audit/api_contract/media/api-vector-unequal-width.gif) | Checks component-specific scales without imposing one shared Gaussian width. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, resolved_max_component_spill <= 0.05, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

## Keep this view current

After editing task/view declarations or publishing new compact results and media indexes, run `python -m experiments.forge experiments-by-tier --output reports/forge/EXPERIMENTS_BY_TIER.md` and commit this same report. New tasks and explicit API question mappings are discovered automatically; no training, rescoring or queue access is needed. The report freshness test detects stale generated content.

The wider question review also links standalone experiments outside the Forge tier catalog. Adding a diagnostic there does not assign it to a Forge tier.

- [Questions](../toy_audit/api_contract/QUESTION_RANKING.md)
- [Later questions](../toy_audit/api_contract/recent_prs/README.md)
- [Caption questions](../toy_audit/api_contract/caption_prs/README.md)

Declaration input digest: `e19612930696d10c45025db31cd5cd2dd7ab2891b0e2bbcb06ad793638eb419a`. The JSON form includes the individual task and view file hashes.

Published artifact input digest: `45daa6fd1be795c7bb7780f1334f19756f3eb11dec5f7f91b0ed13c5cb8da7b6`. Artifact hashes and exact recipe/source/runtime bindings are included in the JSON form.
