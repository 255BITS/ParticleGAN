The vector provider supplies **54 runnable public-API variants mapping 39 existing
catalog questions**. It retains every native 100, vector, stress, vector-proposal,
two-pole, source Gaussian and narrow-ring question in its scope. Named controls
sharing a target law are comparison arms, not additional independent problems.
The API inventory and GIF runner consume `list_cases()` and `build_case()` from
[api_vectors.py](../../../benchmarks/toy_audit/api_vectors.py).

These are new audit callers on develop 664ce464. They do not rewrite historical
recipes, ratings, results or receipts, and do not qualify an optimizer default.
The common fixed seed is 24002. Independent evaluator/control streams are
predeclared; their seeds are not training candidates. The selected public preset,
actual recipe, initializer, resources, sampling law and external budget belong to
each new run's receipt. This provider has not performed full-budget scientific
training or produced a training-convergence claim. The common runner supplies
actual-state GIFs when it executes the registered callers.

| Variants | What the problem verifies | Gate and limits |
| --- | --- | --- |
| `api-grid100`, `api-rotated100`, `api-staggered100` | Recover all 100 modes under ordinary, rotated and staggered geometry, with correct mass and within-mode width. | Original 20,000-sample coverage gate **and** frozen density-fidelity gate: mass TV ≤ .06, center RMS ≤ .20 sigma, absolute conditional covariance-trace bias ≤ .10, radial KS ≤ .04. |
| `api-rotated100-moving` | Fit that full law after two target turns; a poor initial baseline cannot make relative retention sufficient. | Same absolute gates in the current target frame; updates 1–500 / 501–1000 / 1001–1500 add 0/30/60 degrees to the initial 25-degree rotation. |
| Eight `api-vector-*` development cases | Broad bimodality; unequal mass including the 2% mode; unequal widths; anisotropic covariance; observable overlap; narrow widths; joint scale drift; continuous noisy spiral. | Current published law-specific vector bounds, plus 4,096 draws and 32 fixed projected-CDF KS ≤ .06. Gaussian projections have an analytic oracle; spiral uses a fixed independent 8,192-draw reference. |
| `api-reserved-annulus` | Uniform radial **area** mass and continuous rotational support. | The area-uniform sampler, original global vector bounds and the independent 32-projection CDF test; an outer-radius circle is rejected. This already audited family has no fresh held-out-selection credit. |
| Eight `api-stress-*` cases | Preserve the same narrow ring under fast/slow D learning, smaller batch, larger D, longer training, R1/R2, weak D; or fit the distinct broad overlapping-ring law. | Original fixed stress gates plus analytic projected CDF. Broad overlap never requires hidden-component reconstruction; its centers-only historical false positive is rejected. |
| `api-reserved-alternating-critic-updates` | Fit the same law with half-frequency D feedback, preserving work counts and phase. | Public KA2 component caller: D updates on outer steps 1, 3, 5,…; G/prior every step. Default 2,400 outer steps means 1,200 D / 2,400 G updates. No Atlas/DV12 lifecycle equivalence is claimed. |
| Published/control arms for PR45,47–57,152 | Fit each exact unit-rescaled or polygon/pair geometry under its registered **composite** contrast. | Original mass/width bounds plus projected CDF ≤ .06. Every arm remains in the denominator; the comparison does not isolate a single critic or initializer mechanism. |
| `api-two-pole-grid12` | Check six finite target offsets per pole, rather than merely travel from the origin. | One realization enumerating all 12 prior row IDs, with additive output noise off but enabled DV12 latent perturbation retained; balanced mass, ≥ .95 support and maximum sorted quantile error ≤ .10 of the .05 offset half-width. This is a row-cohort quantile check, not exact enumeration or verification of the entire served output law. The target remains the source's actual 12-row data law. |
| `api-gaussian2d` | Fit N((1,1),.04I), and preserve a software continuation checkpoint. | Mean error ≤ .10 sigma, covariance eigenratios [.85,1.15], radial KS ≤ .075, 16 projected CDF KS ≤ .06. Same-mean/same-covariance circle fails. Checkpoint equality is software evidence, separate from generator convergence. |
| `api-ring8-acquire`, `api-ring8-hold`, `api-ring8-shift` | Acquire the narrow sigma .07 ring, persist without optimizer reset, or recover after a +1 x translation. | 4,096 public served-law draws with additive output noise off; enabled DV12 latent perturbation remains. All 8 modes, HQ ≥ .90, mass TV ≤ .075, eigenratios [.5,1.5], per-mode radial KS ≤ .10. Acquisition must pass at 1,200 before hold; hold must pass at 2,400 before shift. No failed acquisition is extended to seek a pass. |
| `api-ring8-resolution12` | Keep the original 12-row resource as a low-resource control, with its actual public sampling law. | Same distribution gate on the perturbed served law. A **separate unperturbed twelve-equal-atom witness** has minimum discrete mass TV 1/6 and insufficient full 2D local covariance. That proof does not apply to the stochastic DV12/feature-cell served law. A continuous data oracle PASS is evaluator calibration, never a trained 12-row generator PASS. |

The narrow-ring reform uses **256×4 actual learned prior rows**, explicitly
changing the source's actual 12×4 resource. The historical configuration's 20,000
field was not its actual resource. The ring sigma .07 law remains distinct from
the stress ring's sigma .12 law. Atlas still trains its learned output-noise law;
ring, ordinary vector, Gaussian and finite-pole gates sample the **public served
latent law with additive output noise disabled**, using `output_noise=False`.
`GANTrainer.sample()` still calls `policy.generate()`, and `ServedModel` retains
the same enabled DV12/feature-cell latent perturbation before G. Disabling output
noise does not turn that stochastic law into a finite set of deterministic
generator atoms, even with exact enumeration of prior row IDs. The `clean_*`
diagnostic metric names mean only output noise disabled. Native 100 scores the
law with output noise enabled and reports the disabled-output-noise gate
separately. Neither cohort borrows the other's result. The caller-owned KA2
alternating-D loop has no DV12 controller or input/output training noise.

The ring shift occurs at update 2,401; intended recovery is by 2,800, followed by
retention through 3,600. A distribution PASS at an isolated endpoint is not the
full adaptation claim. That claim also requires recorded phase/deadline checks
and an unchanged, matched frozen-checkpoint control. The caller checks the
recovery deadline before permitting updates beyond 2,800. Warm/cold restarts share
the complete checkpoint, data generator and optimizer state; software
continuation equality cannot stand in for scientific acquisition or hold.

The R1/R2 stress is an explicit public `ka2` preset with `reg_arm="a_r1r2"`
(effective fixed/K3P penalty). Atlas+this fixed penalty currently reaches a
`CriticStepRecord` without DV12's `sur_base` field. The provider rejects that
unsupported combination before training and exposes the compatible fixed
preset through `GANTrainer`. The half-frequency D caller also declares KA2,
using public `Recipe.make_optimizers`, `GANLoss`, the recipe penalty/regularizer
and rate helper. It preserves the unequal cadence without bypassing the
public formulation or pretending that ordinary `GANTrainer.step()` does so.
Its input/output training noise is explicitly zero, matching the original
alternating-loop law; its public KA2 optimizer mechanics are the declared
adaptation.

Proposal controls preserve the actual published initializer changes. PR45
jointly changes kernel lengths and prior standard deviation. The other 12
comparisons change critic architecture **and** prior initialization. Their
previous captions did not identify these effects separately; the new metadata
names both factors and makes no unsupported “necessary” or “solvable” claim.
Initializers see no evaluator centers or component assignments.

For unequal mass, allocating the nominal 256 table rows by the target masses
gives 5.12 expected rows to the rare 2% component. Public DV12 served outputs
are not constrained to that finite-atom law. The current published local
covariance/spill rule is inherited explicitly: it excludes components below
the 32-row floor while mass, HQ and minimum mass ratio cover every component.
That inherited gate is not a capacity theorem for the perturbed public law.
The projected-CDF test does not prove complete density equality or
rare-component covariance recovery. The historical full-shape rare-component
FAIL remains unchanged.

The native host retains 20,000×2 prior rows, batch 2,048, identity affine G and
the 128×3 Fourier critic, with 7,000 default updates (1,500 for moving). The
quickstart retains 20,000×2 rows, batch 2,048, a 64×2 G and public batch-distance D,
with 1,000 updates. Its initializer is the new caller's explicitly seeded
normal prior, rather than the old example's R2 initialization. Two-pole keeps
the stored host critic, twelve zero prior rows and 80 updates, while adding a
trainable identity affine generator required by `GANTrainer`. These are
explicit API adaptations, not source-byte replays.

Software and evaluator validation: **87 tests passed in 20.14 seconds** on one CPU
thread. All 54 fixed independent full target draws pass. Missing rare mass,
collapsed width, anisotropic trace-matched wrong axes, centers-only overlap,
wrong radial mass, missing native modes, undersized and nonfinite draws fail.
Representative actual public trainer updates execute, observations preserve
state and RNG, and an observed versus unobserved checkpoint continuation
matches exact tensor bytes. Caller-owned KA2 cadence and checkpoint state are
also checked. A separately constructed **unperturbed** 256-atom ring witness
passes its fixed gate; this shows attainable finite-population approximation,
not a trained public-policy PASS. Optimizer uninitialized NaN sentinels are compared by bytes;
generated nonfinite samples still fail closed. Full training/quality evidence
remains the common runner's separately bound responsibility.

Reproduction after integration, with raw artifacts outside Git:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/test_toy_api_vectors.py

# Short software/media evidence; cannot qualify the full default budget.
python -m benchmarks.toy_audit.api_run --recipe auto --device cpu \
  --case api-vector-two-broad --steps 32 --frames 9 \
  --output /ml2/hypergan/toy-api-vector-short

# Full declared budgets (omit --steps); GPU placement is caller-owned.
python -m benchmarks.toy_audit.api_run --recipe auto --device cuda:0 \
  --case api-grid100 --case api-reserved-alternating-critic-updates \
  --output /ml2/hypergan/toy-api-vector-full
```

No full-budget run is part of this provider's software validation.
