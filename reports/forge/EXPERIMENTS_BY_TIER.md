# Forge experiments by tier

Current task assignments, grouped by goal view and qualification tier. Required tasks gate progression; ranking and diagnostic tasks retain their declared roles.

Catalog: **66 tasks**; **59 assigned** to at least one view; **7 unassigned**. Showing **14/14 views**.

Declared priors across the catalog: **38 MoGParticlePrior**, **28 ParticlePrior** (including **4 nonsampled parameter controls**). Every experiment defines `execution.prior` explicitly; candidate and API defaults cannot supply it. `kind: mog` selects `MoGParticlePrior`; `kind: particle_cloud` selects `ParticlePrior`. Sigma alone does not identify the code path. Ordinary Forge MoG tasks require positive sigma; archived zero-sigma MoG evidence keeps its recorded kind. Task sigma is absolute; API demonstrations may instead record the recipe's relative `sigma_rel`.

Reproducible comparisons use the fixed screening seed `0` and candidate-independent named RNG streams. Within each task, candidates share architecture, data law, batch size, prior, initialization, training budget, evaluation cadence and sampling law. Only the declared trainer change varies. The initialization column exposes fixed controls and component policies that take precedence over the deterministic orthogonal fallback; these are separate comparison cohorts. Historical results retain their original bindings.

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
| [bcap_convolution_images](../../configs/forge/views/bcap_convolution_images.json) | 1 | 4 diagnostic | 0 tasks | 0 tasks | provisional |
| [clockfree_continuous](../../configs/forge/views/clockfree_continuous.json) | 3 | 4 required | 19 required | 7 required | provisional |
| [constraint_geometry-diagnostic-v1](../../configs/forge/views/constraint_geometry-diagnostic-v1.json) | 1 | 6 diagnostic | 0 tasks | 0 tasks | provisional |
| [constraint_geometry-round2-diagnostic-v1](../../configs/forge/views/constraint_geometry-round2-diagnostic-v1.json) | 1 | 6 diagnostic | 0 tasks | 0 tasks | provisional |
| [constraint_geometry-round3-diagnostic-v1](../../configs/forge/views/constraint_geometry-round3-diagnostic-v1.json) | 1 | 6 diagnostic | 0 tasks | 0 tasks | provisional |
| [discriminator_stability](../../configs/forge/views/discriminator_stability.json) | 8 | 6 required, 1 diagnostic | 21 required | 2 required | provisional |
| [formulation_comparison](../../configs/forge/views/formulation_comparison.json) | 1 | 3 required | 19 required, 15 diagnostic | 2 required | provisional |
| [host_profile_transfer](../../configs/forge/views/host_profile_transfer.json) | 4 | 3 required | 19 required, 13 diagnostic | 2 required | provisional |
| [k3p_two_pole_horizon](../../configs/forge/views/k3p_two_pole_horizon.json) | 1 | 2 diagnostic | 0 tasks | 0 tasks | provisional |
| [mass-allocation-round5-diagnostic-v1](../../configs/forge/views/mass-allocation-round5-diagnostic-v1.json) | 1 | 6 diagnostic | 0 tasks | 0 tasks | provisional |
| [projection_transport-round4-diagnostic-v1](../../configs/forge/views/projection_transport-round4-diagnostic-v1.json) | 1 | 8 diagnostic | 0 tasks | 0 tasks | provisional |
| [quality_coverage](../../configs/forge/views/quality_coverage.json) | 2 | 3 required | 19 required | 0 tasks | provisional |
| [tier1_policy_coverage](../../configs/forge/views/tier1_policy_coverage.json) | 1 | 7 required | 0 tasks | 0 tasks | undeclared |

## adaptation

Declaration: [adaptation](../../configs/forge/views/adaptation.json); revision 2; goal: `adaptation`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/adaptation.md).

### Tier 1: smoke

3 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

1 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | paired_adaptation / paired_adaptation | 3600 | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate) |

## bcap_convolution_images

Declaration: [bcap_convolution_images](../../configs/forge/views/bcap_convolution_images.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Four source-bound image diagnostics do not qualify a new source or adopt public defaults.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

4 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

## clockfree_continuous

Declaration: [clockfree_continuous](../../configs/forge/views/clockfree_continuous.json); revision 3; goal: `clockfree_continuous`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/clockfree_continuous.md).

### Tier 1: smoke

4 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [clockfree_audit_measurement_v1](../../configs/forge/tasks/clockfree_audit_measurement_v1.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-measurement-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |

### Tier 2: quality

19 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

7 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit) | clockfree_audit / clockfree_parity | 24 | 300 | — |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |
| [grid100_14k](../../configs/forge/tasks/grid100_14k.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [grid100](../../configs/forge/tasks/grid100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [rotated100_14k](../../configs/forge/tasks/rotated100_14k.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [rotated100](../../configs/forge/tasks/rotated100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [staggered100_14k](../../configs/forge/tasks/staggered100_14k.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [staggered100](../../configs/forge/tasks/staggered100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | paired_adaptation / paired_adaptation | 3600 | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate) |

## constraint_geometry-diagnostic-v1

Declaration: [constraint_geometry-diagnostic-v1](../../configs/forge/views/constraint_geometry-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Explicit mechanism diagnostic only; unchanged task gates, no ordinary-tier qualification.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

6 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

## constraint_geometry-round2-diagnostic-v1

Declaration: [constraint_geometry-round2-diagnostic-v1](../../configs/forge/views/constraint_geometry-round2-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Explicit mechanism diagnostic only; unchanged task gates, no ordinary-tier qualification.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

6 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

## constraint_geometry-round3-diagnostic-v1

Declaration: [constraint_geometry-round3-diagnostic-v1](../../configs/forge/views/constraint_geometry-round3-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Explicit mechanism diagnostic only; unchanged task gates, no ordinary-tier qualification.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

6 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

## discriminator_stability

Declaration: [discriminator_stability](../../configs/forge/views/discriminator_stability.json); revision 8; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

Candidate outcomes, metrics and measured costs: [leaderboard](technique-inventory.md).

### Tier 1: smoke

6 required, 1 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | required | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json) | required | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-acquisition) | transfer_vector / transfer_sustained | 1600 | 300 | — |
| [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |
| [clockfree_audit_measurement_v1](../../configs/forge/tasks/clockfree_audit_measurement_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-measurement-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |

### Tier 2: quality

21 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | required | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

2 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

## formulation_comparison

Declaration: [formulation_comparison](../../configs/forge/views/formulation_comparison.json); revision 1; goal: `host_profile_transfer`.

Declared calibration status: **provisional**.

Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/formulation_comparison.md).

### Tier 1: smoke

3 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required, 15 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [grid100_affine_paired_laws_v1](../../configs/forge/tasks/grid100_affine_paired_laws_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [grid100_release07_cloud_named_v1](../../configs/forge/tasks/grid100_release07_cloud_named_v1.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

2 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

## host_profile_transfer

Declaration: [host_profile_transfer](../../configs/forge/views/host_profile_transfer.json); revision 4; goal: `host_profile_transfer`.

Declared calibration status: **provisional**.

Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/host_profile_transfer.md).

### Tier 1: smoke

3 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required, 13 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

2 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

## k3p_two_pole_horizon

Declaration: [k3p_two_pole_horizon](../../configs/forge/views/k3p_two_pole_horizon.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

2 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 800 | 300 | — |
| [two_pole_800_schedule800_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 800 | 300 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

## mass-allocation-round5-diagnostic-v1

Declaration: [mass-allocation-round5-diagnostic-v1](../../configs/forge/views/mass-allocation-round5-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

One bounded allocation diagnostic; no ordinary qualification or promotion.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

6 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

## projection_transport-round4-diagnostic-v1

Declaration: [projection_transport-round4-diagnostic-v1](../../configs/forge/views/projection_transport-round4-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Explicit mechanism diagnostic only; unchanged task gates, no ordinary-tier qualification.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

8 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

## quality_coverage

Declaration: [quality_coverage](../../configs/forge/views/quality_coverage.json); revision 2; goal: `quality_coverage`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/quality_coverage.md).

### Tier 1: smoke

3 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

19 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

0 tasks.

No tasks assigned.

## tier1_policy_coverage

Declaration: [tier1_policy_coverage](../../configs/forge/views/tier1_policy_coverage.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **undeclared**.

Candidate outcomes, metrics and measured costs: [leaderboard](technique-inventory.md).

### Tier 1: smoke

7 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / transfer_sustained | 1000 | 120 | — |
| [two_pole_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-acquisition) | transfer_vector / transfer_sustained | 400 | 300 | — |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / transfer_sustained | 20001 | 900 | — |
| [clockfree_audit_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-tier1-policy-selected-cloud-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

## Tasks unassigned to any view

These catalog tasks have no tier placement. Add an assignment to a view to include them in its policy.

| Task | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [five_word_joint_acquisition](../../configs/forge/tasks/five_word_joint_acquisition.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / transfer_sustained | 20001 | 900 | — |
| [gaussian1d_acquisition](../../configs/forge/tasks/gaussian1d_acquisition.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / transfer_sustained | 1000 | 120 | — |
| [gaussian1d_shallow_smoke](../../configs/forge/tasks/gaussian1d_shallow_smoke.json) | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_shallow_stability](../../configs/forge/tasks/gaussian1d_shallow_stability.json) | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_shallow_smoke](../../configs/forge/tasks/gaussian1d_shallow_smoke.json) (checkpoint) |
| [grid100_affine_square_named_v1_14k](../../configs/forge/tasks/grid100_affine_square_named_v1_14k.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-grid100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [rotated100_affine_square_named_v1_14k](../../configs/forge/tasks/rotated100_affine_square_named_v1_14k.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-rotated100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [staggered100_affine_square_named_v1_14k](../../configs/forge/tasks/staggered100_affine_square_named_v1_14k.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-staggered100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |

## Experiment guides

Task variants share a guide when their declarations name the same host or problem. This grouping is for navigation; it does not assert matching scientific contracts. Public-API variants join only through their explicit retained question IDs.

### Experiment: ae-gan-hold

Checks reconstruction/identity and an acquired adversarial edit during the declared hold.

Forge declarations: [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json), [ae_gan_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Declared Forge numerical gates and sampling:

[ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json)

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = encoder, generator, prior, discriminator; mechanism exercised = True; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | generated_and_reconstructed_prior_with_scheduled_output_noise |
| Scoring weights | live |
| Evaluation output noise | public_recipe_schedule |

[ae_gan_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = encoder, generator, prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_generated_and_reconstructed_prior_with_scheduled_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | public_recipe_schedule |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| ae_gan_hold | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [K3P](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 737592c128ef / b5a03f23b4f5 | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [KA2](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 737592c128ef / 488b09cfca27 | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [R1/R2](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 737592c128ef / 34afb01fc624 | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [GAN v3 release 0.7](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 737592c128ef / 78833310ac5c | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-ae-anchor-hold](../toy_audit/api_contract/media/api-ae-anchor-hold.gif) | Reconstruct the noisy two-anchor inputs while the independently sampled prior covers both anchors Scope: New public Recipe.encode particle AE with reconstruction weight1 and applied GAN/KA2; independently sampled prior quality is separately scored. | MoGParticlePrior (sigma_rel=0.025) | COMPLETE / FAIL; 250/250 updates; quality_fraction, quality_mass_tv, last 5 post-update metric observations do not all pass | ae_gan / cpu / 3efb003d4c86 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: clockfree-audit

Check saved public trainer state under step_label, horizon, evaluation_cadence, restart perturbations.

Forge declarations: [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json).

Declared Forge numerical gates and sampling:

[clockfree_audit](../../configs/forge/tasks/clockfree_audit.json)

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: clockfree-audit-measurement-v1

Measure the original four clock/state parity conditions and zero-dependency condition for an explicitly scheduled recipe; diagnostic FAIL grants no clock-free claim or qualification.

Forge declarations: [clockfree_audit_measurement_v1](../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Declared Forge numerical gates and sampling:

[clockfree_audit_measurement_v1](../../configs/forge/tasks/clockfree_audit_measurement_v1.json)

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: clockfree-audit-tier1-policy-selected-cloud-v1

Check saved public trainer state under step_label, horizon, evaluation_cadence, restart perturbations.

Forge declarations: [clockfree_audit_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Declared Forge numerical gates and sampling:

[clockfree_audit_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json)

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: cover-leftover

Checks target coverage plus the separate unwanted-remainder/content constraints.

Forge declarations: [cover_leftover](../../configs/forge/tasks/cover_leftover.json).

Declared Forge numerical gates and sampling:

[cover_leftover](../../configs/forge/tasks/cover_leftover.json)

| Metric | Required bound |
| --- | --- |
| u_kept | >= 0.85 |
| content_kept | >= 0.75 |
| leak_ratio | <= 0.2 |
| pole_rel_err_plus | <= 0.2 |
| pole_rel_err_minus | <= 0.2 |
| same_dir | <= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| cover_leftover | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-guarded-leftover](../toy_audit/api_contract/media/api-guarded-leftover.gif) | Cover both signed poles while preserving content and removing the guarded leak Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 800/800 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: five-word-joint

Can a joint BiGAN generator, encoder and critic acquire five equally likely canonical words and reconstruct every correctly paired input with confident token probabilities, including padding?

Explanation, interpretation and reproduction: [experiment readout](five-word-joint/README.md), [experiment readout](five-word-tier-split/README.md).

Forge declarations: [five_word_joint_acquisition](../../configs/forge/tasks/five_word_joint_acquisition.json), [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json), [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json), [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json).

Declared Forge numerical gates and sampling:

[five_word_joint_acquisition](../../configs/forge/tasks/five_word_joint_acquisition.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[five_word_joint_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: exact optimizer updates = True; finite state = True; mechanism exercised = True; optimizer roles = generator, encoder, prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = False.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| five_word_joint_hold | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [K3P](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / b5a03f23b4f5 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [KA2](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / 488b09cfca27 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [R1/R2](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / 34afb01fc624 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [GAN v3 release 0.7](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / 78833310ac5c | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [forge-five-word-joint-api-demo-v1](../toy_audit/api_contract/five_word_joint/goal.gif) | Can a joint BiGAN generator, encoder and critic acquire five equally likely canonical words and reconstruct every correctly paired input with confident token probabilities, including padding? Scope: One bounded shared-API integration demonstration, not an ordinary Forge run or release qualification. Score the declared bounds honestly at 32 updates and grade the evidence INCOMPLETE against the 20,001-update task. | ParticlePrior (sigma=0) | COMPLETE / INCOMPLETE; 32/20001 updates; quality_fraction, modes, mass_tv, reconstruction_exact, minimum_reconstruction_token_probability | ka2 / cpu / 997c7f01b99a | [definition](../toy_audit/api_contract/five_word_joint/publication.json); [readout](../toy_audit/api_contract/five_word_joint/publication.json); [recipe and provenance](../toy_audit/api_contract/five_word_joint/publication.json) |
| [image-five-word-joint-hold-confirmed-v1](../toy_audit/api_contract/five_word_smoke_hold/media/five_word_joint_hold.gif) | Continue this candidate's earliest confirmed five-word acquisition state for 4,000 updates. Every scheduled generation and inverse check, including the exact restored state, must pass. Scope: task-only selected BCAP DualNorm verification; no ordinary family qualification | ParticlePrior (sigma=0) | COMPLETE / FAIL; 4000/4000 updates | bcap / unrecorded / 3c82db6c7b24 | [definition](../toy_audit/api_contract/five_word_smoke_hold/publication.json); [readout](../toy_audit/api_contract/five_word_smoke_hold/publication.json); [recipe and provenance](../toy_audit/api_contract/five_word_smoke_hold/publication.json) |
| [image-five-word-joint-smoke-confirmed-v1](../toy_audit/api_contract/five_word_smoke_hold/media/five_word_joint_smoke.gif) | Can public joint BiGAN training acquire all five words and confidently reconstruct every paired input at one independently confirmed scheduled state? Complete all 20,001 updates. Scope: task-only selected BCAP DualNorm verification; no ordinary family qualification | ParticlePrior (sigma=0) | COMPLETE / PASS; 20001/20001 updates | bcap / unrecorded / 3c82db6c7b24 | [definition](../toy_audit/api_contract/five_word_smoke_hold/publication.json); [readout](../toy_audit/api_contract/five_word_smoke_hold/publication.json); [recipe and provenance](../toy_audit/api_contract/five_word_smoke_hold/publication.json) |
| [image-five-words-joint-ae](../toy_audit/api_contract/media/image-five-words-joint-ae.gif) | Generate the five equally likely canonical words with confident normalized token probabilities, and reconstruct each of the five matched inputs including underscore padding. Scope: Finite vocabulary apple/grape/lemon/melon/berry only. Joint BiGAN inverse reconstruction; no unseen words or natural-language generation. New API-policy variant, not reuse of historical EMA PASS. | ParticlePrior (sigma=0) | COMPLETE / PASS; 20001/20001 updates | ka2 / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: gaussian1d-acquisition

Can the public ParticleGAN trainer acquire the scalar law N(2, 0.5^2) from random initialization within 1,000 updates, with correct location, width and CDF shape at five terminal checks?

Explanation, interpretation and reproduction: [experiment readout](gaussian-shallow/README.md), [experiment readout](gaussian-smoke-tier-split/README.md), [experiment readout](../toy_audit/api_contract/gaussian1d/README.md).

Forge declarations: [gaussian1d_acquisition](../../configs/forge/tasks/gaussian1d_acquisition.json), [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json), [gaussian1d_shallow_smoke](../../configs/forge/tasks/gaussian1d_shallow_smoke.json), [gaussian1d_shallow_stability](../../configs/forge/tasks/gaussian1d_shallow_stability.json), [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json), [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json).

Declared Forge numerical gates and sampling:

[gaussian1d_acquisition](../../configs/forge/tasks/gaussian1d_acquisition.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[gaussian1d_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[gaussian1d_shallow_smoke](../../configs/forge/tasks/gaussian1d_shallow_smoke.json), [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[gaussian1d_shallow_stability](../../configs/forge/tasks/gaussian1d_shallow_stability.json), [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| gaussian1d_smoke | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [K3P](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / 737592c128ef / b5a03f23b4f5 | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [KA2](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / 737592c128ef / 488b09cfca27 | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [R1/R2](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / 737592c128ef / 34afb01fc624 | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [GAN v3 release 0.7](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / 737592c128ef / 78833310ac5c | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_stability | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-gaussian1d-acquisition](../toy_audit/api_contract/gaussian1d/goal.gif) | Can the public ParticleGAN trainer acquire the scalar law N(2, 0.5^2) from random initialization within 1,000 updates, with correct location, width and CDF shape at five terminal checks? Scope: Acquire N(2, .5^2) in 1,000 updates from random initialization; exact Gaussian CDF, location and width gates must pass at five terminal checks. Tier 1 remains provisional; this standalone API cohort grants no whole-view qualification. | MoGParticlePrior (sigma_rel=0) | COMPLETE / FAIL; 1000/1000 updates; cdf_ks <= 0.05, last 5 post-update metric observations do not all pass | k3p / cpu / 666642486d3c | [definition](../toy_audit/api_contract/gaussian1d/results.json); [readout](../toy_audit/api_contract/gaussian1d/results.json); [recipe and provenance](../toy_audit/api_contract/gaussian1d/results.json) |

### Experiment: grid100

Recover all 100 Gaussian components, balanced mass, centers and within-mode covariance/radial spread; distinguish clean from noisy served laws.

Forge declarations: [grid100](../../configs/forge/tasks/grid100.json), [grid100_14k](../../configs/forge/tasks/grid100_14k.json), [grid100_affine_paired_laws_v1](../../configs/forge/tasks/grid100_affine_paired_laws_v1.json), [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json), [grid100_affine_square_named_v1_14k](../../configs/forge/tasks/grid100_affine_square_named_v1_14k.json), [grid100_release07_cloud_named_v1](../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Declared Forge numerical gates and sampling:

[grid100](../../configs/forge/tasks/grid100.json), [grid100_14k](../../configs/forge/tasks/grid100_14k.json), [grid100_affine_paired_laws_v1](../../configs/forge/tasks/grid100_affine_paired_laws_v1.json), [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json), [grid100_affine_square_named_v1_14k](../../configs/forge/tasks/grid100_affine_square_named_v1_14k.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[grid100_release07_cloud_named_v1](../../configs/forge/tasks/grid100_release07_cloud_named_v1.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| grid100 | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-grid100](../toy_audit/api_contract/media/api-grid100.gif) | Recover all 100 equal-weight Gaussian modes, their mass and local width, including independent density-fidelity bounds. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / PASS; 7000/7000 updates | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-bars4

Healthy location transfer: four horizontal/vertical bar positions test spatial coverage.

Forge declarations: [img_bars4](../../configs/forge/tasks/img_bars4.json), [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json).

Declared Forge numerical gates and sampling:

[img_bars4](../../configs/forge/tasks/img_bars4.json), [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| img_bars4 | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_bars4-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_bars4-residual_upsample16.gif) | Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. Scope: Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 600/600 updates; distribution_tv, finite_template_tv, last 5 post-update metric observations do not all pass | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_bars4-source-transpose12](../toy_audit/api_contract/media/image-develop-img_bars4-source-transpose12.gif) | Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. Scope: Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 600/600 updates; distribution_tv, finite_template_tv, modes, last 5 post-update metric observations do not all pass | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-blobs4

Healthy location transfer: four small corner patches test localized quality and coverage.

Forge declarations: [img_blobs4](../../configs/forge/tasks/img_blobs4.json), [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json).

Declared Forge numerical gates and sampling:

[img_blobs4](../../configs/forge/tasks/img_blobs4.json), [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| img_blobs4 | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_blobs4-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_blobs4-residual_upsample16.gif) | Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. Scope: Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_blobs4-source-transpose12](../toy_audit/api_contract/media/image-develop-img_blobs4-source-transpose12.gif) | Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. Scope: Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 600/600 updates; distribution_tv, finite_template_tv, hq, modes, last 5 post-update metric observations do not all pass | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-intensity2

Healthy photometric transfer: two patch intensities require intensity fidelity as well as support coverage.

Forge declarations: [img_intensity2](../../configs/forge/tasks/img_intensity2.json), [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json).

Declared Forge numerical gates and sampling:

[img_intensity2](../../configs/forge/tasks/img_intensity2.json), [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| img_intensity2 | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_intensity2-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_intensity2-residual_upsample16.gif) | Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. Scope: Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_intensity2-source-transpose12](../toy_audit/api_contract/media/image-develop-img_intensity2-source-transpose12.gif) | Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. Scope: Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-stripes2

Healthy orientation transfer: two distinct stripe orientations with an adequately sized convolutional GAN.

Forge declarations: [img_stripes2](../../configs/forge/tasks/img_stripes2.json), [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json).

Declared Forge numerical gates and sampling:

[img_stripes2](../../configs/forge/tasks/img_stripes2.json), [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| img_stripes2 | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_stripes2-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_stripes2-residual_upsample16.gif) | Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. Scope: Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_stripes2-source-transpose12](../toy_audit/api_contract/media/image-develop-img_stripes2-source-transpose12.gif) | Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. Scope: Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: mid-scale-identity

Checks identity preservation and target edit magnitude at intermediate control strength.

Forge declarations: [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json).

Declared Forge numerical gates and sampling:

[mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json)

| Metric | Required bound |
| --- | --- |
| concept_cos_plus | >= 0.85 |
| concept_cos_minus | >= 0.85 |
| concept_mag_plus | >= 0.75 |
| concept_mag_plus | <= 1.25 |
| concept_mag_minus | >= 0.75 |
| concept_mag_minus | <= 1.25 |
| identity_at_0 | >= 0.85 |
| identity_at_mid | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| mid_scale_identity | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-midscale-identity](../toy_audit/api_contract/media/api-midscale-identity.gif) | Retain identity at half strength in addition to correct neutral and signed poles Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 800/800 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: mode-hold

Checks all eight ring modes and HQ through the sampled terminal hold; not within-mode density fidelity.

Forge declarations: [mode_hold](../../configs/forge/tasks/mode_hold.json), [ring_extension](../../configs/forge/tasks/ring_extension.json), [ring_hold](../../configs/forge/tasks/ring_hold.json), [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json).

Declared Forge numerical gates and sampling:

[mode_hold](../../configs/forge/tasks/mode_hold.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[ring_extension](../../configs/forge/tasks/ring_extension.json)

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[ring_hold](../../configs/forge/tasks/ring_hold.json)

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json)

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| mode_hold | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

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

Declared Forge numerical gates and sampling:

[residual_student](../../configs/forge/tasks/residual_student.json)

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |
| success_rate | >= 1 |
| wrong_pad_rate | <= 0 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | conditional_prior_centers_with_scheduled_output_noise |
| Scoring weights | live |
| Evaluation output noise | public_recipe_schedule |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| residual_student | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-trajectory-residual](../toy_audit/api_contract/media/api-trajectory-residual.gif) | Recover the fast trajectory with a residual head and reject a correct marginal with wrong identities Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 400/400 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: ring16-acquisition

Acquire all 16 equally weighted two-dimensional Gaussian clusters from scratch: radius 3, sigma 0.1; require meaningful occupancy in every cluster, roughly balanced mass and noncollapsed local spread within 1600 updates. No extended hold phase.

Forge declarations: [ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json), [ring16_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Declared Forge numerical gates and sampling:

[ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 96 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[ring16_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| ring16_acquisition | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [K3P](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / 737592c128ef / b5a03f23b4f5 | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [KA2](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / 737592c128ef / 488b09cfca27 | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [R1/R2](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / 737592c128ef / 34afb01fc624 | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [GAN v3 release 0.7](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / 737592c128ef / 78833310ac5c | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-ring16-acquisition](../toy_audit/api_contract/ring16/goal.gif) | Acquire all 16 equally weighted two-dimensional Gaussian clusters from scratch: radius 3, sigma 0.1; require meaningful occupancy in every cluster, roughly balanced mass and noncollapsed local spread within 400 updates. No extended hold phase. Scope: From random initialization, acquire sixteen equal radius-three sigma-.1 Gaussian clusters within 400 updates. Five terminal acquisition checks add no hold phase. Tier 1 is provisional; this standalone K3P/API initializer and RNG cohort supplies no Forge promotion credit. | MoGParticlePrior (sigma_rel=0) | ERROR / FAIL; 400/400 updates; component_covariance_error <= 0.85, hq >= 0.85, last 5 post-update metric observations do not all pass, goal media/state error: ModuleNotFoundError: No module named 'matplotlib' | k3p / cuda:0 / 746146cbdc5a | [definition](../toy_audit/api_contract/ring16/publication.json); [readout](../toy_audit/api_contract/ring16/publication.json); [recipe and provenance](../toy_audit/api_contract/ring16/publication.json) |

### Experiment: rotated100

Recover all 100 Gaussian components, balanced mass, centers and within-mode covariance/radial spread; distinguish clean from noisy served laws.

Forge declarations: [rotated100](../../configs/forge/tasks/rotated100.json), [rotated100_14k](../../configs/forge/tasks/rotated100_14k.json), [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json), [rotated100_affine_square_named_v1_14k](../../configs/forge/tasks/rotated100_affine_square_named_v1_14k.json).

Declared Forge numerical gates and sampling:

[rotated100](../../configs/forge/tasks/rotated100.json), [rotated100_14k](../../configs/forge/tasks/rotated100_14k.json), [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json), [rotated100_affine_square_named_v1_14k](../../configs/forge/tasks/rotated100_affine_square_named_v1_14k.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| rotated100 | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-rotated100](../toy_audit/api_contract/media/api-rotated100.gif) | Recover all 100 equal-weight Gaussian modes, their mass and local width, including independent density-fidelity bounds. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / PASS; 7000/7000 updates | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: staggered100

Recover all 100 Gaussian components, balanced mass, centers and within-mode covariance/radial spread; distinguish clean from noisy served laws.

Forge declarations: [staggered100](../../configs/forge/tasks/staggered100.json), [staggered100_14k](../../configs/forge/tasks/staggered100_14k.json), [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json), [staggered100_affine_square_named_v1_14k](../../configs/forge/tasks/staggered100_affine_square_named_v1_14k.json).

Declared Forge numerical gates and sampling:

[staggered100](../../configs/forge/tasks/staggered100.json), [staggered100_14k](../../configs/forge/tasks/staggered100_14k.json), [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json), [staggered100_affine_square_named_v1_14k](../../configs/forge/tasks/staggered100_affine_square_named_v1_14k.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| staggered100 | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-staggered100](../toy_audit/api_contract/media/api-staggered100.gif) | Recover all 100 equal-weight Gaussian modes, their mass and local width, including independent density-fidelity bounds. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / PASS; 7000/7000 updates | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: trajectory

Checks the extracted trajectory edit while preserving identity in finite paired rows.

Forge declarations: [trajectory](../../configs/forge/tasks/trajectory.json).

Declared Forge numerical gates and sampling:

[trajectory](../../configs/forge/tasks/trajectory.json)

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | conditional_prior_centers_with_scheduled_output_noise |
| Scoring weights | live |
| Evaluation output noise | public_recipe_schedule |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| trajectory | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-trajectory-edit](../toy_audit/api_contract/media/api-trajectory-edit.gif) | Change angular speed while preserving each trajectory's radius and starting phase Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 400/400 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: two-pole

Does an unchanged global recipe move from zero by 800 updates with schedule horizon 800? Movement and bounded slope are the declared question; two-mode fidelity is diagnostic only.

Explanation, interpretation and reproduction: [experiment readout](k3p-two-pole-horizon-v1/README.md).

Forge declarations: [two_pole](../../configs/forge/tasks/two_pole.json), [two_pole_800_schedule800_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json), [two_pole_800_schedule80_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json), [two_pole_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Declared Forge numerical gates and sampling:

[two_pole](../../configs/forge/tasks/two_pole.json), [two_pole_800_schedule800_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json), [two_pole_800_schedule80_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json)

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = prior, discriminator; mechanism exercised = True; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_particles_and_critic_gradient |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

[two_pole_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_learned_particles_and_critic_gradient |
| Scoring weights | state_selected |
| Evaluation output noise | not_applied_to_measurement |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| two_pole | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [K3P](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / b5a03f23b4f5 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [KA2](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / 488b09cfca27 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [R1/R2](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / 34afb01fc624 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [GAN v3 release 0.7](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / 78833310ac5c | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-two-pole-grid12](../toy_audit/api_contract/media/api-two-pole-grid12.gif) | Check six target offsets per pole in one full row-ID realization, beyond mere travel; full served output-law fidelity remains unmeasured when latent perturbation is active. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 80/80 updates; max_grid_quantile_error_halfwidth <= 0.1, support_fraction >= 0.95, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: unipolar

Checks an intended edit with preservation of unrelated content.

Forge declarations: [unipolar](../../configs/forge/tasks/unipolar.json).

Declared Forge numerical gates and sampling:

[unipolar](../../configs/forge/tasks/unipolar.json)

| Metric | Required bound |
| --- | --- |
| cover | >= 0.85 |
| off_caption | <= 0.05 |
| neu_hold | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| unipolar | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-unipolar-hold](../toy_audit/api_contract/media/api-unipolar-hold.gif) | Make the positive 4D edit while holding the free scale-zero origin Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 400/400 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: unused-token-hold

Checks that active controls move and unused controls remain unchanged.

Forge declarations: [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json), [unused_token_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Declared Forge numerical gates and sampling:

[unused_token_hold](../../configs/forge/tasks/unused_token_hold.json)

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = generator, discriminator; mechanism exercised = True; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

[unused_token_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_learned_parameter_measurement |
| Scoring weights | state_selected |
| Evaluation output noise | not_applied_to_measurement |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| unused_token_hold | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [K3P](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / b5a03f23b4f5 | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [KA2](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / 488b09cfca27 | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [R1/R2](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / 34afb01fc624 | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [GAN v3 release 0.7](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 737592c128ef / 78833310ac5c | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-unused-token-hold](../toy_audit/api_contract/media/api-unused-token-hold.gif) | Move the concept slot on its target axis while keeping the unused slot fixed Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / FAIL; 200/200 updates; concept_move, last 5 post-update metric observations do not all pass | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-anisotropic

Checks covariance shape: a narrow axis cannot be rescued by a wide one.

Forge declarations: [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json), [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json).

Declared Forge numerical gates and sampling:

[vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json), [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_anisotropic | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-anisotropic](../toy_audit/api_contract/media/api-vector-anisotropic.gif) | Checks covariance shape: a narrow axis cannot be rescued by a wide one. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, resolved_max_component_spill <= 0.05, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-overlap

Scores the observable distribution when latent components are not identifiable.

Forge declarations: [vector_overlap](../../configs/forge/tasks/vector_overlap.json), [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json).

Declared Forge numerical gates and sampling:

[vector_overlap](../../configs/forge/tasks/vector_overlap.json), [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_overlap | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-overlap](../toy_audit/api_contract/media/api-vector-overlap.gif) | Scores the observable distribution when latent components are not identifiable. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-spiral

Checks continuous curved mass rather than a finite list of target mode centers.

Forge declarations: [vector_spiral](../../configs/forge/tasks/vector_spiral.json), [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json).

Declared Forge numerical gates and sampling:

[vector_spiral](../../configs/forge/tasks/vector_spiral.json), [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_spiral | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-spiral](../toy_audit/api_contract/media/api-vector-spiral.gif) | Checks continuous curved mass rather than a finite list of target mode centers. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1600/1600 updates; last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-two-broad

Basic learnable multimodal distribution and within-mode spread.

Forge declarations: [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json), [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json).

Declared Forge numerical gates and sampling:

[vector_two_broad](../../configs/forge/tasks/vector_two_broad.json), [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_two_broad | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-two-broad](../toy_audit/api_contract/media/api-vector-two-broad.gif) | Basic learnable multimodal distribution and within-mode spread. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-unequal-mass

Checks target occupancy including the rare 2% component, not uniformity.

Forge declarations: [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json), [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json).

Declared Forge numerical gates and sampling:

[vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json), [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_unequal_mass | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-unequal-mass](../toy_audit/api_contract/media/api-vector-unequal-mass.gif) | Checks target occupancy including the rare 2% component, not uniformity. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-unequal-width

Checks component-specific scales without imposing one shared Gaussian width.

Forge declarations: [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json), [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json).

Declared Forge numerical gates and sampling:

[vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json), [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_unequal_width | [BCAP dualnorm (experimental starting point)](../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / unbound / 37d98f2bc216 | [source-bound receipt index](technique-inventory.json) |

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

Declaration input digest: `04de12c263ea5c85d5736ab27fa97e450acd2472061f4c2c968661240a0b2e58`. The JSON form includes the individual task and view file hashes.

Published artifact input digest: `b23b0f41d21d3ee681008d65d3c6169296a76a881a81b1629d5d8456aafa360c`. Artifact hashes and exact recipe/source/runtime bindings are included in the JSON form.
