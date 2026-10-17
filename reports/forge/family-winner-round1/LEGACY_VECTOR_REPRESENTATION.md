# Legacy scalar vector capacity

All six required scalar-vector laws have constructive states that pass their unchanged Forge gates in the actual frozen public host. This resolves the target-capacity prerequisite for this cohort; it gives no trained pass or optimizer learnability claim. The [compact certificate](legacy-vector-representation.json) binds source digest `9f89fa7cc7af552ef7e41405fab415abbd00acf3d3c6e4af8e4435fb2423142e`, exact frozen request/card/source hashes, complete recipes, runtime and all external state/sample/metric hashes. The completed evaluation took 6.441063 seconds on CPU1, with zero fitting or model updates. Three observer setup errors before any sampling or model updates are retained separately.

The public prior has 256 rows in four dimensions with a continuous Gaussian kernel of sigma 0.025; the generator has two 64-unit LeakyReLU layers. Output noise is disabled under the declared scalar sampling law. The five mixtures use selector coordinate 0 and independent Gaussian coordinates 1/2, with component-local affine maps set by each target covariance. These maps use at most 30 existing first-layer units. Equal masses use equal row counts; unequal mass uses fixed largest-remainder counts `[141,77,33,5]`, whose mass quantization TV is 0.0015625. Even the rare five-row group produces a continuous distribution, so the full-component shape gates remain active. Anisotropic row counts `[86,85,85]` have mass quantization TV 0.00260417.

The piecewise Gaussian maps depart from the ideal component maps only in far latent tails. They are numeric tolerance witnesses, rather than claims of an identical Gaussian mixture law. Probe maximum floating-point errors are below 0.000026. The noisy spiral uses fixed 256 midpoint curve locations and an affine scale of 3.2, giving the target Gaussian width 0.08. Its analytic absolute Wasserstein coupling bound for curve quadrature is 0.036965; this approximates the continuous uniform-curve law.

Every one of five 4,096-sample public capacity draws and each independent 100,000-sample holdout passes the original full-component/distribution gate. Every zero-cloud negative control fails. Values below are the worst across the five capacity draws, not a selected best sample or trained terminal suffix.

| Question | What its target verifies | Representative worst bounds |
|---|---|---|
| `vector_two_broad` | Two separated modes retain their Gaussian spread | SW1 0.0268911; covariance error 0.0309533; minimum eigenratio 0.958233 |
| `vector_unequal_mass` | The four declared masses include the rare 2% group with local width | SW1 0.0280504; covariance error 0.0793360; minimum eigenratio 0.832883; minimum mass ratio 0.939941 |
| `vector_unequal_width` | Each of four components keeps its own sigma | SW1 0.0300030; covariance error 0.0527098; minimum eigenratio 0.925756 |
| `vector_anisotropic` | Narrow axes and oriented covariances survive | SW1 0.0309218; covariance error 0.0438763; minimum eigenratio 0.933449 |
| `vector_overlap` | The observable density's mean/covariance/projections match despite unidentifiable labels | SW1 0.0244928; mean error 0.0298730; covariance error 0.0447779 |
| `vector_spiral` | Curved continuous mass and Gaussian transverse width survive | SW1 0.0179270; mean error 0.0136223; covariance error 0.0487394 |

The script constructs the real public models, applies analytic parameter states and samples through `GANTrainer.sample`. All model/prior bytes stay unchanged during observation; only named evaluation streams advance. Optimizer states remain empty and completed updates remain zero. Held-out measurements are never training signals. The fixed protocol seed and all setup attempts are disclosed.

Atlas/E22's automatic feature selection, DV12 latent perturbation and served averaging define a different public-policy sampling cohort. These scalar MoG witnesses cannot be used as proof records for that cohort. They also cannot qualify unexecuted vector tasks after a candidate stopped on a preceding required failure.
