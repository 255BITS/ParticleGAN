# Diagnosing the recorded toy failures

This zero-training audit joins **all 109 catalog entries** and explains **73 failing or deficient arms**: 17 failed frozen references, 43 failed proposal arms, six clean/strict native results, four ungated MisGAN laws, two execution blockers and PR224's constructed unsafe release. The [machine-readable map](failure-diagnosis.json) binds every case/arm to exact source and artifact hashes. Historical outcomes, thresholds, budgets and clean/noisy cohorts remain unchanged.

## What the evidence establishes

- **Three late image acquisitions fail the hold:** PR62's original arm has a final passing suffix of 3, PR71 has 2, and PR63's positive control has 4; the declared gate requires 5. Their final samples pass, but the scientific verdict remains FAIL. PR154's control ends at HQ .875, below .9, so that failure is still endpoint fidelity.
- **Rare-mass shape is under-resolved:** the historical unequal-mass gate fails only the 2% component's minimum covariance eigenvalue (.0812 versus .15). A 256-particle table allocates about 5.12 atoms to it; 4,096 evaluator draws do not create new support atoms. Mass, HQ and resolved-core measurements pass. This is a frozen evaluator/resolution mismatch, preserved as the original FAIL.
- **Ring covariance errors often come from spills:** full covariance error can be many times the in-core error. The decomposition attributes the exact population covariance to core variation, far-spill variation and the offset between those groups. A healthy core does not excuse spill or occupancy failures; no historical gate is substituted.
- **Mean-only D and uniform G are structurally limited:** equal-mean blobs are indistinguishable to the former; the latter's best possible stripe RMSE is .433, above .1. These are useful negative controls, not demonstrated solvable positives. Width2/latent1 is an observed capacity/optimization stress, not a proven representation impossibility.
- **Atlas optimizes a different population law from the clean gate:** output sigma about .029 supplies about 93% of a sigma-.03 target's component variance. Analytically, a clean Gaussian with sigma **.007681** convolved with sigma-.029 jitter exactly recovers the target, yet its clean variance ratio **.06556** fails the native **.40** floor. Clean served spread contracts while paired noisy samples satisfy spread gates. **Clean sampling still includes feature-cell/DV12 latent perturbation**; the saved raw-table census deliberately omits it and is not the clean serving law. The terminal critic probe examines expansion and centering directions without a training update; it does not establish the actual optimizer path.
- **Moving recovery is weaker than density recovery:** the original gate accepts 95 modes and 90% of baseline HQ. Its endpoint has 97 modes and about .926 precision, failing the absolute density requirements. The last shift leaves only two terminal gate observations, so the stationary five-check hold is not established either.
- **MisGAN marginals do not establish posterior recovery:** p50's output is almost deterministic despite a broad saved noise table; p80 has roughly plausible diversity magnitude but wrong posterior modes; block has excellent projected generation and poor single-sensor conditioning. The 2D projection cannot observe six orthogonal dimensions. All remain measured ungated results, not post hoc binary FAILs.
- **Circle/sprite never train on this snapshot:** their retained exceptions name the missing host import/API method. All 13 archived vector scripts' initial public-API errors are retained separately; their adapted experiments are not current KA2/Atlas qualifications.

**Confidence is scoped.** Structural impossibility, loss of critic information, sampling-law spread and exact gate shortfalls are established. Output trajectories do not reveal an Adam/KA2/critic causal mechanism. Each unresolved record names the missing weights, optimizer states or caller cursor and a bounded next diagnostic. Passing composite controls establish solvability under their exact architecture/init recipe; they do not isolate each changed factor.

The [separate revised vector controls](NON_IMAGE_QUALITY.md) strengthen narrow distribution gates and retain exact initialization confounds. Any failures exposed only by those revised gates belong to that new diagnostic cohort; this report explains the frozen original results.

## Every failing frozen reference and proposal arm

`original` and `control` denote the proposal's frozen arm ordering. First/best/terminal trajectories and per-mode/per-template decompositions are in the JSON; best snapshots never replace terminal evidence.

| Catalog ID / arm | Frozen failing gate or hold | Diagnosis |
|---|---|---|
| `develop-vector_unequal_mass` / frozen_reference | component_min_eigen_ratio=0.081165 >= 0.15 | finite particle evaluator mismatch; full/core covariance error 0.442/0.376 |
| `develop-mode_hold` / frozen_reference | modes=4 >= 8; hq=0.49243 >= 0.9 | support misplacement |
| `develop-residual_student` / frozen_reference | success_rate=0.91667 >= 1; wrong_pad_rate=0.083333 <= 0 | late paired identity regression |
| `develop-vector_overlap` / frozen_reference | mean_error=0.16579 <= 0.15 | late observable mean drift |
| `pr45-adapted` / proposal_arm | sw1_normalized=0.30692 <= 0.18; mass_tv=0.26025 <= 0.15 | mode mass imbalance; full/core covariance error 0.083/0.083 |
| `pr131` / proposal_arm | modes=1 >= 2; hq=0.5625 >= 0.9 | template fidelity; 18/32 quality atoms |
| `pr73` / proposal_arm | modes=0 >= 2; hq=0.1875 >= 0.9 | template fidelity; 6/32 quality atoms |
| `pr68` / proposal_arm | hq=0.8125 >= 0.9 | template fidelity; 26/32 quality atoms |
| `pr166` / proposal_arm | modes=1 >= 2; hq=0.6875 >= 0.9 | template fidelity; 22/32 quality atoms |
| `develop-img_bars8` / frozen_reference | modes=5 >= 8; hq=0.6875 >= 0.9 | template fidelity and occupancy; 22/32 quality atoms |
| `pr170` / proposal_arm | modes=1 >= 2; hq=0.3125 >= 0.9 | template fidelity; 10/32 quality atoms |
| `pr154` / proposal_arm | hq=0.71875 >= 0.9 | template fidelity; 23/32 quality atoms |
| `pr154` / positive_control | hq=0.875 >= 0.9 | template fidelity; 28/32 quality atoms |
| `pr62` / proposal_arm | Final gates pass; suffix 3/5 | insufficient terminal stability; 30/32 quality atoms |
| `pr71` / proposal_arm | Final gates pass; suffix 2/5 | insufficient terminal stability; 31/32 quality atoms |
| `pr74` / proposal_arm | hq=0.84375 >= 0.9 | template fidelity; 27/32 quality atoms |
| `pr66` / proposal_arm | modes=0 >= 2; hq=0.25 >= 0.9 | template fidelity; 8/32 quality atoms |
| `pr79` / proposal_arm | modes=0 >= 2; hq=0 >= 0.9 | template fidelity; 0/32 quality atoms |
| `pr151` / proposal_arm | modes=1 >= 2; hq=0.6875 >= 0.9 | template fidelity; 22/32 quality atoms |
| `pr58` / proposal_arm | modes=0 >= 2; hq=0 >= 0.9 | template fidelity and occupancy; 0/32 quality atoms |
| `pr78` / proposal_arm | modes=0 >= 2; hq=0 >= 0.9 | template fidelity; 0/32 quality atoms |
| `pr150` / proposal_arm | modes=0 >= 2; hq=0.03125 >= 0.9 | template fidelity; 1/32 quality atoms |
| `pr106` / proposal_arm | modes=0 >= 2; hq=0 >= 0.9 | template fidelity; 0/32 quality atoms |
| `pr77` / proposal_arm | hq=0.8125 >= 0.9 | template fidelity; 26/32 quality atoms |
| `pr67` / proposal_arm | modes=1 >= 2; hq=0.375 >= 0.9 | template fidelity; 12/32 quality atoms |
| `pr69` / proposal_arm | modes=0 >= 2; hq=0.0625 >= 0.9 | template fidelity; 2/32 quality atoms |
| `pr59` / proposal_arm | modes=0 >= 2; hq=0 >= 0.9 | template fidelity and occupancy; 0/32 quality atoms |
| `pr159` / proposal_arm | modes=0 >= 2; hq=0 >= 0.9 | template fidelity; 0/32 quality atoms |
| `pr75` / proposal_arm | modes=0 >= 2; hq=0 >= 0.9 | template fidelity; 0/32 quality atoms |
| `pr72` / proposal_arm | hq=0.71875 >= 0.9 | template fidelity; 23/32 quality atoms |
| `develop-img_tiny_generator` / frozen_reference | modes=1 >= 4; hq=0.375 >= 0.9 | template fidelity and occupancy; 12/32 quality atoms |
| `pr80` / proposal_arm | modes=1 >= 2; hq=0.53125 >= 0.9 | template fidelity; 17/32 quality atoms |
| `pr64` / proposal_arm | modes=0 >= 2; hq=0.0625 >= 0.9 | template fidelity; 2/32 quality atoms |
| `develop-reserved_alternating_critic_updates` / frozen_reference | hq=0.84766 >= 0.85; component_covariance_error=21.342 <= 0.85 | far spill dominates covariance; full/core covariance error 21.3/0.373 |
| `develop-stress_fast_critic` / frozen_reference | component_covariance_error=3.9312 <= 0.85 | far spill dominates covariance; full/core covariance error 3.93/0.375 |
| `develop-stress_large_critic` / frozen_reference | mass_tv=0.17529 <= 0.15; component_covariance_error=10.852 <= 0.85 | far spill dominates covariance; full/core covariance error 10.9/0.346 |
| `develop-stress_long_horizon` / frozen_reference | component_covariance_error=5.6872 <= 0.85 | far spill dominates covariance; full/core covariance error 5.69/0.307 |
| `develop-stress_r1_r2` / frozen_reference | mass_tv=0.177 <= 0.15; hq=0.81958 >= 0.85; component_covariance_error=17.92 <= 0.85 | far spill dominates covariance; full/core covariance error 17.9/0.226 |
| `develop-stress_slow_critic` / frozen_reference | sw1_normalized=0.20497 <= 0.18; mass_tv=0.22754 <= 0.15; hq=0.55933 >= 0.85; component_covariance_error=51.596 <= 0.85 | far spill dominates covariance; full/core covariance error 51.6/0.582 |
| `develop-stress_small_batch` / frozen_reference | hq=0.78149 >= 0.85; component_covariance_error=21.761 <= 0.85 | far spill dominates covariance; full/core covariance error 21.8/0.37 |
| `develop-stress_weak_critic` / frozen_reference | sw1_normalized=0.18118 <= 0.18; mass_tv=0.23657 <= 0.15; hq=0.13721 >= 0.85; component_covariance_error=34.753 <= 0.85; component_min_eigen_ratio=0 >= 0.15 | spread and spill failure; full/core covariance error 34.8/1.54 |
| `pr152-adapted` / proposal_arm | sw1_normalized=0.60709 <= 0.18; mass_tv=0.5 <= 0.15; component_min_eigen_ratio=0 >= 0.15 | mode mass loss; full/core covariance error 0.559/0.559 |
| `pr57-adapted` / proposal_arm | sw1_normalized=0.20053 <= 0.18; mass_tv=0.27653 <= 0.15 | mode mass imbalance; full/core covariance error 0.15/0.15 |
| `pr53-adapted` / proposal_arm | sw1_normalized=0.40482 <= 0.18; mass_tv=0.32992 <= 0.15; component_min_eigen_ratio=0 >= 0.15 | mode mass imbalance; full/core covariance error 0.389/0.389 |
| `pr50-adapted` / proposal_arm | sw1_normalized=0.37205 <= 0.18; mass_tv=0.33423 <= 0.15 | mode mass imbalance; full/core covariance error 0.142/0.142 |
| `pr51-adapted` / proposal_arm | sw1_normalized=0.22902 <= 0.18; mass_tv=0.2382 <= 0.15 | mode mass imbalance; full/core covariance error 0.139/0.123 |
| `pr55-adapted` / proposal_arm | sw1_normalized=0.26188 <= 0.18; mass_tv=0.21729 <= 0.15 | mode mass imbalance; full/core covariance error 0.159/0.159 |
| `develop-vector_narrow` / frozen_reference | sw1_normalized=0.48209 <= 0.18; mass_tv=0.5 <= 0.15; component_covariance_error=12.117 <= 0.85; component_min_eigen_ratio=0 >= 0.15 | far spill dominates covariance; full/core covariance error 12.1/0.772 |
| `pr52-adapted` / proposal_arm | sw1_normalized=0.581 <= 0.18; mass_tv=0.59102 <= 0.15; component_min_eigen_ratio=0.067818 >= 0.15 | mode mass imbalance; full/core covariance error 0.344/0.344 |
| `pr54-adapted` / proposal_arm | sw1_normalized=0.35504 <= 0.18; mass_tv=0.29451 <= 0.15 | mode mass imbalance; full/core covariance error 0.0847/0.0847 |
| `pr49-adapted` / proposal_arm | sw1_normalized=0.35128 <= 0.18; mass_tv=0.33333 <= 0.15; component_min_eigen_ratio=0 >= 0.15 | mode mass loss; full/core covariance error 0.379/0.379 |
| `pr56-adapted` / proposal_arm | sw1_normalized=0.32816 <= 0.18; mass_tv=0.37231 <= 0.15; component_min_eigen_ratio=0 >= 0.15 | mode mass loss; full/core covariance error 0.438/0.438 |
| `pr47-adapted` / proposal_arm | sw1_normalized=0.23265 <= 0.18; mass_tv=0.19971 <= 0.15 | mode mass imbalance; full/core covariance error 0.0719/0.034 |
| `pr48-adapted` / proposal_arm | sw1_normalized=0.62534 <= 0.18; mass_tv=0.5 <= 0.15; component_min_eigen_ratio=0 >= 0.15 | mode mass loss; full/core covariance error 0.528/0.528 |
| `pr65` / proposal_arm | modes=0 >= 2; hq=0 >= 0.9 | template fidelity; 0/32 quality atoms |
| `pr63` / proposal_arm | modes=1 >= 2; hq=0.53125 >= 0.9 | template fidelity; 17/32 quality atoms |
| `pr63` / positive_control | Final gates pass; suffix 4/5 | insufficient terminal stability; 30/32 quality atoms |
| `develop-img_mean_discriminator` / frozen_reference | modes=0 >= 4; hq=0 >= 0.9 | critic nonidentifiability; 0/32 quality atoms |
| `pr61` / proposal_arm | modes=1 >= 2; hq=0.65625 >= 0.9 | template fidelity; 21/32 quality atoms |
| `develop-img_uniform_generator` / frozen_reference | modes=0 >= 2; hq=0 >= 0.9 | representation impossible; 0/32 quality atoms |

## Frozen terminal native critic probe

The probe increases within-mode clean width while keeping each empirical mode mean fixed, using the terminal trained critic and captured jitter. Synthetic paired reals use the assigned target center plus that fixed jitter scaled to target sigma. A positive derivative means expansion locally increases the generator game. This FP64 CPU input-gradient probe does not reproduce the training pairing, controller-preconditioned parameter update or earlier trajectory.

| Saved endpoint | Width expansion derivative under captured noisy law | Modes penalizing expansion | Center correction derivative |
|---|---:|---:|---:|
| `atlas-grid100` | 2.55899e-06 | 68 | -1.07755e-06 |
| `atlas-rotated100` | 1.70156e-06 | 79 | -8.03218e-07 |
| `atlas-staggered100` | 8.12843e-07 | 61 | -6.54315e-07 |
| `atlas-rotated100_moving` | 6.04361e-05 | 81 | -5.90455e-07 |

All three stationary endpoints locally penalize expansion on this declared probe while rewarding center correction. This supports a terminal critic contraction signature. Acquisition/contraction checkpoints and actual caller pairings are still needed to explain the training mechanism.

## Native and conditional endpoint evidence

| Catalog ID / law | Observed failure or deficiency | Scope |
|---|---|---|
| `pr224` / native_release | The predeclared native release doubles the stiff-coordinate spectral step factor from 1.6 (stable) to 3.2 (unstable). Cancelling only that release keeps log(2); the safe geometry still releases safely. This is a causally isolated constructed controller fixture, not a trained-dataset failure. | proven within constructed unit fixture |
| `atlas-grid100` / clean | All 100 centers and mass gates pass, but clean within-mode covariance and radial spread contract below their frozen lower bounds. The paired served-law observations differ only by recorded Gaussian output jitter; its variance is approximately 93% of the target component variance. An exact noisy Gaussian fit at sigma_out=.029 needs clean sigma=.007681 and variance ratio .06556, which itself fails the clean .40 covariance floor. The optimized noisy law and the clean full-width gate ask different population questions. | proven population-law mismatch; optimizer cause unresolved |
| `atlas-rotated100` / clean | All 100 centers and mass gates pass, but clean within-mode covariance and radial spread contract below their frozen lower bounds. The paired served-law observations differ only by recorded Gaussian output jitter; its variance is approximately 93% of the target component variance. An exact noisy Gaussian fit at sigma_out=.029 needs clean sigma=.007681 and variance ratio .06556, which itself fails the clean .40 covariance floor. The optimized noisy law and the clean full-width gate ask different population questions. | proven population-law mismatch; optimizer cause unresolved |
| `atlas-staggered100` / clean | All 100 centers and mass gates pass, but clean within-mode covariance and radial spread contract below their frozen lower bounds. The paired served-law observations differ only by recorded Gaussian output jitter; its variance is approximately 93% of the target component variance. An exact noisy Gaussian fit at sigma_out=.029 needs clean sigma=.007681 and variance ratio .06556, which itself fails the clean .40 covariance floor. The optimized noisy law and the clean full-width gate ask different population questions. | proven population-law mismatch; optimizer cause unresolved |
| `pr22` / current_api_host | ImportError: cannot import name 'edit_cap' from 'lib.gym_particle_finetune' (/ml2/hypergan/ParticleGAN-toy-problem-audit/lib/gym_particle_finetune.py) | proven execution cause |
| `pr196-block` / generation_and_conditional_imputation | Projected generation covers all 100 modes with 99.82% HQ, while the 10% single-sensor rows have ambiguous accuracy .097 versus .777 Bayes. Excess diversity (.152 versus .062) and posterior TV show that good marginal generation does not imply correct conditioning. | observed; structural scorer limitation proven |
| `pr196-mcar_p20` / generation_and_conditional_imputation | Projected generation is good, but full-8D off-plane distance exceeds the true data scale and conditional posterior TV exceeds the finite-draw oracle. Only one ambiguous test row exists, so its perfect acc_lo is weak evidence of posterior recovery. | observed; structural scorer limitation proven |
| `pr196-mcar_p50` / generation_and_conditional_imputation | Generation reaches 99 projected modes yet only 38.18% 3-sigma fidelity. The imputer is almost deterministic despite a broad learned noise table: missing-coordinate std is about 6% of Bayes, and ambiguous-row accuracy .112 is far below .555. This is loss of useful conditional noise response, not a collapsed input-noise bank. | observed; structural scorer limitation proven |
| `pr196-mcar_p80` / generation_and_conditional_imputation | Severe data fidelity and conditional failures coexist: 32 modes, 8.51% HQ, posterior TV .510 versus .179 finite-draw Bayes. Diversity magnitude alone looks plausible (.325 versus .334) but lands in the wrong posterior modes. | observed; structural scorer limitation proven |
| `pr153` / current_api_host | AttributeError: 'Recipe' object has no attribute 'make_gradient_penalty'. Did you mean: 'make_critic_penalty'? | proven execution cause |
| `atlas-rotated100_moving` / clean | The original relative reacquisition criterion permits 95/100 modes and 90% of the pre-shift HQ. Final strict scoring has only 97 genuine modes, missing low-mass components and overly broad residual populations; accuracy moments are undefined when a component lacks enough in-radius samples. Only two gate observations occur after the last target jump, below a five-check final hold even if both were good. | proven gate mismatch; optimizer cause unresolved |
| `atlas-rotated100_moving` / noisy | The original relative reacquisition criterion permits 95/100 modes and 90% of the pre-shift HQ. Final strict scoring has only 97 genuine modes, missing low-mass components and overly broad residual populations; accuracy moments are undefined when a component lacks enough in-radius samples. Only two gate observations occur after the last target jump, below a five-check final hold even if both were good. | proven gate mismatch; optimizer cause unresolved |
| `atlas-rotated100_moving` / accuracy | The original relative reacquisition criterion permits 95/100 modes and 90% of the pre-shift HQ. Final strict scoring has only 97 genuine modes, missing low-mass components and overly broad residual populations; accuracy moments are undefined when a component lacks enough in-radius samples. Only two gate observations occur after the last target jump, below a five-check final hold even if both were good. | proven gate mismatch; optimizer cause unresolved |

## Reproduce without training

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m benchmarks.toy_audit.failure_diagnosis \
  --artifacts /ml2/hypergan/toy-audit-artifacts-20261001 \
  --output /tmp/toy-failure-diagnosis
```

The default reads terminal checkpoints for a fixed native critic direction probe and a MisGAN noise-response enumeration. `--no-state-probes` produces a separately labelled cloud/metric-only diagnostic. No training loop, checkpoint mutation, continuation, RNG-seed search, default/config repair or qualification promotion runs.

The generated JSON is compact: hashes, aggregate decompositions, first/best/terminal measurements and exact next actions. Raw arrays, checkpoint weights, per-update streams and execution logs remain outside Git.
