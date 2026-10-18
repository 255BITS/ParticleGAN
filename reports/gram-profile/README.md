# Guarded dense Gram profile

`gram_fp64_guarded_v1` is an opt-in numerical profile. The existing solver
remains the default. K/V rowspace reduction is excluded. The [public API](../../docs/api.md)
specifies restrictions, checkpoint identity and terminal failure behavior.
The guard checks exactly the factors forming the returned update, with no
second decomposition or implicit fallback.

## Historical evidence scope

The preceding isolated Sana research measured complete protected G optimizer
steps at 17.2661721 s with default SVD and 8.61826356 s guarded (2.00344x).
Actual FP32 rounded displacement agreed with CPU64 to global relative error
6.95816e-8 and worst-leaf 2.04925e-7. These are source-bound historical results,
not throughput claims for this new implementation or whole training.
Twenty-one leaves differed from old FP32 SVD by more than 1e-4; that fidelity
hold remains. Native protection branches were inactive. Fixed-input restore
tests do not establish an evolving trajectory or output quality.

Package tests use numerical fixtures and public optimizer steps. These are
kernel/API checks, not GAN acquisition experiments. No training, models or
weights are published.

## Reproduction

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_gram_guarded_profile.py tests/test_gram_profile_integration.py
# Caller enforces wall/memory/output caps; CUDA needs separate admission.
timeout 60 python -m benchmarks.gram_profile --device cpu --size 32 --output /tmp/gram-profile.json
```

The benchmark times complete public `NormalizedOptimizer.step`, including the
selected guard, and reports preparation, CPU64 reference and reporting separately.
The repeated synthetic bank is not actual Sana gradients. Each repeat starts
fresh; execution PASS does not waive numerical gates. Inspect rounded-update
errors and preserve HOLDs.

## Package qualification

The expanded CPU optimizer/recipe suites passed 437 tests with 244 CUDA skips.
One additional archived-history migration regression passed after fetching its
exact historical commits. The standalone benchmark contracts passed 19 tests;
independent core and integration reviews found no remaining material issue.
Bounded final GPU qualification is pending. Raw stdout and JUnit files remain
outside Git; compact final measurements will be added here.
