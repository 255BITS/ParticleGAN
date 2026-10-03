# Eight source-bound questions and retained observables

Source `26ff278c3796d775969391adc0bde52e3af11149`; metadata/Recipe reads only. This checklist is retrospective source documentation, not a newly preregistered scientific run. No fixture/model, restore, sample, scorer or CUDA call occurs in this reader.

Each family uses one shared `lr=.006375 / prior_lr_mult=1 / d_lr_mult=1` configuration over all eight definitions. All16 outcomes stay in the denominator. The original full-budget/final-five gate and additional first-five-plus-five-later hold gate are separate; unreached cases remain UNKNOWN.

Primary serving is fast-only with no DV12 or latent standardization. Fixed output noise is a training regularizer: images and ordinary vectors evaluate without additive output noise; native100 primary evaluation includes its scheduled sigma. Native clean panels are diagnostics. The generic provider caption mentioning retained latent perturbation is not evidence that this named family has a controller. `amsgrad=False` selects the ordinary Adam denominator while preserving the public KA2/K3P penalty, guard and conditional row-damping owners.

## image-develop-img_intensity2-source-transpose12

Can the original convolutional GAN recover both equal-mass patch intensities with accurate pixels and sustained finite-template fidelity?

Two 8x8 grayscale images: central rows/columns 2:6 at .35 or .85; all other pixels zero; target probabilities .5/.5. Training pixels add independent sigma=.01 noise and are clipped to [0,1]; evaluation uses the exact clean ordered bank.

Reason to retain: Photometric recovery: matching patch geometry or nearest-mode coverage cannot hide brightness errors, rejected images, or unequal template frequencies.

Original resources: 600 updates, batch32, evaluation n=1024; 24 post-update numeric checks, final5 required. Exact frozen thresholds:

```json
{
  "distribution_tv_max": 0.1,
  "finite_template_tv_max": 0.1,
  "hq_min": 0.9,
  "min_mode_fraction": 0.25,
  "minimum_stable_checks": 5,
  "modes": 2,
  "observations": 24,
  "quality_rmse": 0.06
}
```

GIF purpose: Fixed ordered .35/.85 goals and real served images at actual update boundaries; keep the default FAIL/PASS and HQ/finite-template TV visible. A two-patch-looking endpoint alone is insufficient.

Retained diagnostic fields: Saved image arrays: central-patch intensity and background leakage; Exact distinct clean outputs and draw multiplicities; nearest/quality counts per target; Recorded hq, rejected_mass, distribution_tv and finite_template_tv; Final prior row geometry, G/D and EMA parameter differences; optimizer row histories and critic call/anchor records.

Limits: Mean RMSE is diagnostic; .06 is a per-image quality cutoff, not an aggregate RMSE gate. Deterministic 32-row population counting requires all32 distinct outputs to be present in the retained draw; row-to-output IDs and intermediate weights are not saved.

## api-vector-two-broad

Can the same shared family configuration learn two broad Gaussian components with correct mass and within-mode spread, rather than collapse onto two centers?

Two stationary means (-1,0)/(1,0), mass .5/.5, isotropic sigma=.25 (covariance .0625I).

Reason to retain: Broad two-mode smoke condition tests mass and continuous width before the expensive narrow100 modes. Center-only coverage is explicitly rejected by full covariance/eigenvalue and analytic CDF gates.

Original resources: 1200 updates, batch128, evaluation n=4096; 24 post-update numeric checks, final5 required. Exact frozen thresholds:

```json
[
  [
    "sw1_normalized",
    "<=",
    0.18
  ],
  [
    "mass_tv",
    "<=",
    0.15
  ],
  [
    "hq",
    ">=",
    0.85
  ],
  [
    "component_covariance_error",
    "<=",
    0.85
  ],
  [
    "component_min_eigen_ratio",
    ">=",
    0.15
  ],
  [
    "sample_count",
    ">=",
    4096
  ],
  [
    "projection_ks",
    "<=",
    0.06
  ]
]
```

GIF purpose: Global held-out target/generated scatter, nearest-mode mass bars, and the fixed mode0 zoom at every media boundary; preserve the analytic projection-KS bound even if both centers look occupied.

Retained diagnostic fields: Recorded mass_tv, hq, component_covariance_error, component_min_eigen_ratio, sw1_normalized and projection_ks; Retained scatter arrays permit per-mode means/covariance, spill/outlier and fixed-projection CDF attribution without new draws; Final learned prior/G weights and public optimizer state; data/latent/noise streams.

Limits: The public receipt keeps scalar metrics only; per-component arrays from the underlying vector scorer are not retained as scalar fields. A sampled CDF failure does not identify a critic or optimizer mechanism.

## api-grid100

Can the original linear-generator/public-critic host learn all100 equally weighted narrow Gaussian modes, their local widths and independent density-fidelity bounds?

10x10 axis-aligned centers from coordinates -4.5 through4.5; equal .01 weights; isotropic sigma=.03.

Reason to retain: High mode count and narrow precision jointly test population coverage, mass, local density and long-horizon stability. Exact center occupancy is not enough.

Original resources: 7000 updates, batch2048, evaluation n=20000; 24 post-update numeric checks, final5 required. Exact frozen thresholds:

```json
{
  "accuracy": {
    "abs_cov_trace_bias": 0.1,
    "center_rms_sigma": 0.2,
    "mass_tv": 0.06,
    "radial_ks": 0.04
  },
  "native": {
    "all_finite": true,
    "max_cov_eig_ratio": 1.7,
    "max_mass_tv": 0.1,
    "max_mode_mass": 0.02,
    "max_radial_median_ratio": 1.4,
    "min_cov_eig_ratio": 0.4,
    "min_hq_mode_mass": 0.005,
    "min_modes": 100,
    "min_precision": 0.97,
    "min_radial_median_ratio": 0.65,
    "min_samples": 20000
  }
}
```

GIF purpose: Global100-mode cloud and mass display plus fixed mode0 sigma-scale zoom; noisy-primary and output-noise-off diagnostic panels must be explicitly distinguished.

Retained diagnostic fields: Primary native precision/mass/covariance/radial metrics and independent accuracy_ metrics; Saved primary noisy cloud and clean_ diagnostic cloud; output_sigma and clean_gate_passed; Long-horizon scheduled G/D/prior rates and optimizer counters; Actual fast serving flag; no feature-cell or DV12 controller exists under these named families.

Limits: Native primary is sampled with the current scheduled fixed output noise; clean diagnostics never replace its gates. Zero-clock sigma0 capacity is not a certificate for later .029 additive-noise reachability.

## api-rotated100

Do the original100-mode coverage, width and accuracy requirements still hold after a fixed25-degree rotation?

The same 10x10 equal100 Gaussian mixture rotated by25 degrees; isotropic sigma=.03 remains unchanged.

Reason to retain: A useful orientation control: it preserves mode count/width/masses and varies alignment with the generator/critic features. This is related to grid100, not an independent natural-data generalization law.

Original resources: 7000 updates, batch2048, evaluation n=20000; 24 post-update numeric checks, final5 required. Exact frozen thresholds:

```json
{
  "accuracy": {
    "abs_cov_trace_bias": 0.1,
    "center_rms_sigma": 0.2,
    "mass_tv": 0.06,
    "radial_ks": 0.04
  },
  "native": {
    "all_finite": true,
    "max_cov_eig_ratio": 1.7,
    "max_mass_tv": 0.1,
    "max_mode_mass": 0.02,
    "max_radial_median_ratio": 1.4,
    "min_cov_eig_ratio": 0.4,
    "min_hq_mode_mass": 0.005,
    "min_modes": 100,
    "min_precision": 0.97,
    "min_radial_median_ratio": 0.65,
    "min_samples": 20000
  }
}
```

GIF purpose: Show the actual rotated global target and generated cloud with the identical local-width/mass panels and unchanged numerical annotations.

Retained diagnostic fields: Same native/accuracy and noisy-vs-clean fields as grid100; Per-mode residual direction/covariance from retained arrays can expose axis-sensitive errors; Final parameter/rate/optimizer records.

Limits: The rotation is stationary, not a moving-target adaptation test. Failure/success differences need matched evidence before being attributed to feature alignment.

## api-staggered100

Can the same100-mode host preserve precision, mass and local density on a row-offset lattice with compressed horizontal spacing?

Original grid centers: coordinate0 scaled by .85; alternating rows shift coordinate1 by -.25/+.25; equal .01 masses and isotropic sigma=.03.

Reason to retain: Tests structured but nonrectangular geometry and spacing. It is neither another seed of grid100 nor a moving law.

Original resources: 7000 updates, batch2048, evaluation n=20000; 24 post-update numeric checks, final5 required. Exact frozen thresholds:

```json
{
  "accuracy": {
    "abs_cov_trace_bias": 0.1,
    "center_rms_sigma": 0.2,
    "mass_tv": 0.06,
    "radial_ks": 0.04
  },
  "native": {
    "all_finite": true,
    "max_cov_eig_ratio": 1.7,
    "max_mass_tv": 0.1,
    "max_mode_mass": 0.02,
    "max_radial_median_ratio": 1.4,
    "min_cov_eig_ratio": 0.4,
    "min_hq_mode_mass": 0.005,
    "min_modes": 100,
    "min_precision": 0.97,
    "min_radial_median_ratio": 0.65,
    "min_samples": 20000
  }
}
```

GIF purpose: Actual staggered target/served cloud globally, plus fixed mode0 local width and mass panels; show all100 masses rather than relying on dot visibility.

Retained diagnostic fields: Same original native/accuracy and clean-diagnostic fields; Per-row/per-mode deficits, residual direction and spill attribution using the saved clouds; Actual scheduled rates and population optimizer state.

Limits: Unequal spacing does not change target component sigma or uniform mass. No separate feature-cell mechanism is active in this noncontinuous cohort.

## api-vector-unequal-mass

Can the shared configuration reproduce strongly unequal mode probabilities while preserving the shape of components resolved by the256-row population?

Four corners at (+/-1.5,+/-1.5), source order (-,-),(-,+),(+,-),(+,+), masses .55/.30/.13/.02; all sigma=.18.

Reason to retain: Rare-mode and dominant-mode balancing differs from equal mixtures. The .02 component is explicitly under the32-row expected-population shape floor, but its mass and distributional CDF remain tested.

Original resources: 1200 updates, batch128, evaluation n=4096; 24 post-update numeric checks, final5 required. Exact frozen thresholds:

```json
[
  [
    "sw1_normalized",
    "<=",
    0.18
  ],
  [
    "mass_tv",
    "<=",
    0.15
  ],
  [
    "hq",
    ">=",
    0.85
  ],
  [
    "resolved_core_covariance_error",
    "<=",
    0.5
  ],
  [
    "resolved_core_min_eigen_ratio",
    ">=",
    0.15
  ],
  [
    "resolved_max_component_spill",
    "<=",
    0.05
  ],
  [
    "min_mass_ratio",
    ">=",
    0.25
  ],
  [
    "sample_count",
    ">=",
    4096
  ],
  [
    "projection_ks",
    "<=",
    0.06
  ]
]
```

GIF purpose: Global scatter and unequal target-mass bars, with the fixed first-mode width zoom. Rare-mode visibility is not full rare covariance qualification.

Retained diagnostic fields: min_mass_ratio, mass_tv, hq and analytic projection_ks; resolved_core_covariance_error, resolved_core_min_eigen_ratio and resolved_max_component_spill; Retained arrays allow component counts and covariance/spill attribution with the frozen resolution mask; Prior row ownership/counters and final G/D state.

Limits: Expected row counts are140.8/76.8/33.28/5.12; the last component is excluded from resolved shape aggregates, not dropped from mass/HQ/CDF tests. Original historical full-shape failures remain unchanged; this named current gate has a different explicit resolution contract.

## api-vector-anisotropic

Can the shared configuration recover three differently oriented Gaussian ellipses, including their narrow eigen-directions and resolved local spill?

Equal three modes at (-2,-1),(0,1.5),(2,-1), with source covariances [[.09,.018],[.018,.0081]], [[.0081,-.018],[-.018,.09]], [[.04,.03],[.03,.04]].

Reason to retain: Correct global variance or round blobs can hide local anisotropy; signed covariance and the minimum whitened eigenvalue test distinct shape recovery.

Original resources: 1200 updates, batch128, evaluation n=4096; 24 post-update numeric checks, final5 required. Exact frozen thresholds:

```json
[
  [
    "sw1_normalized",
    "<=",
    0.18
  ],
  [
    "mass_tv",
    "<=",
    0.15
  ],
  [
    "hq",
    ">=",
    0.85
  ],
  [
    "resolved_core_covariance_error",
    "<=",
    0.5
  ],
  [
    "resolved_core_min_eigen_ratio",
    ">=",
    0.15
  ],
  [
    "resolved_max_component_spill",
    "<=",
    0.05
  ],
  [
    "sample_count",
    ">=",
    4096
  ],
  [
    "projection_ks",
    "<=",
    0.06
  ]
]
```

GIF purpose: Three target ellipses and generated cloud globally; target mass bars and fixed mode0 zoom reveal narrow-axis shrinkage and spill.

Retained diagnostic fields: Resolved core covariance error/minimum eigenvalue/spill plus mass_tv,hq,sw1_normalized,projection_ks; Retained samples allow per-component rotated eigensystems and remote-outlier attribution; Final generator/prior parameter shape and optimizer row counters.

Limits: All three components resolve at about85.33 expected rows; broad-axis error cannot be traded against a missing narrow axis. The gate does not identify which optimizer owner caused a local-shape failure.

## image-develop-img_bars4-source-transpose12

Can the same original convolutional host recover all four equal-mass bar locations/orientations with accurate pixels and sustained frequency fidelity?

Four8x8 binary templates: vertical bars in columns1:3 and5:7, followed by horizontal bars in rows1:3 and5:7; .25 mass each. Training pixels add sigma=.01 noise and clip to[0,1].

Reason to retain: Spatial/orientation recovery in a convolutional host; it distinguishes correct locations from mean intensity alone and tests four-way mass after earlier prerequisites.

Original resources: 600 updates, batch32, evaluation n=1024; 24 post-update numeric checks, final5 required. Exact frozen thresholds:

```json
{
  "distribution_tv_max": 0.1,
  "finite_template_tv_max": 0.1,
  "hq_min": 0.9,
  "min_mode_fraction": 0.125,
  "minimum_stable_checks": 5,
  "modes": 4,
  "observations": 24,
  "quality_rmse": 0.1
}
```

GIF purpose: Ordered four reference bars beside actual served images at real updates, retaining mode count, HQ and both mass-fidelity annotations.

Retained diagnostic fields: Saved image arrays: nearest-template location/orientation counts and rejected-image attribution; Recorded modes,hq,distribution_tv,finite_template_tv,rejected_mass; G/prior parameter diversity and critic penalty/optimizer state.

Limits: .10 is the per-image RMSE cutoff; both aggregate TVs separately require<=.10. An unreached case remains UNKNOWN even if its zero-update capacity witness is SUPPORTED.

## Evidence limits

The public runner retains all metric/media-union sample arrays, scalar metrics and a final complete owner checkpoint. It discards step-return losses and does not save intermediate parameters, per-row evaluated output IDs, gradients, or penalty-phase events. Final optimizer counters can establish completed calls or an anchor having started, but cannot reconstruct every earlier transition or explain why a particular row crossed a quality bound. Final G/prior EMA weights can be compared as parameters; their unobserved output law cannot be inferred without a separately declared model evaluation.

Exact case metadata, full per-family resolved Recipes, public sampling laws, source file SHA256s and observation/media boundaries are in `source-questions-and-observables.json`. No scientific grades are assigned by this checklist.
