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

Implementation commit `ba725989f279121be55a8bd2f083bd957a93d32d`, based on
`develop` at `7b8908c0a55ad58250626c44b3b4324a66debbc9`:

- Expanded CPU optimizer/recipe suites: **437 passed, 244 CUDA skips**. One
  history-dependent regression initially lacked its archived Git commits and
  passed separately after fetching those exact commits. Standalone benchmark
  contracts: **19 passed**. Initial fixture failures are preserved locally.
- Bounded CUDA suite: **158 passed**, including 15 new selected-profile tests
  for CPU64 agreement, active/inactive protection, pending restore, fixed next
  updates and terminal cutoff holds.
- Public `NormalizedOptimizer.step` benchmark: 112 FP32 1152x1152 matrices plus
  112 biases, repeated identical synthetic full-rank inputs, seed0 initializer,
  one warmup and two measured fresh-state repeats. Default median **11.1761252 s**;
  guarded median **9.68237786 s**, **1.15427x**. Both include their complete update;
  the guarded number includes same-factor guard work. This measures neither
  DirectionBlend nor model forward/backward and is not whole-training throughput.
- Guarded rounded-update relative error against the independent CPU64 SVD map:
  **1.63098e-9** maximum across measured repeats, max absolute **4.54747e-13**.
  Default relative error on this fixture was **3.74374e-4**. A completed benchmark
  does not mean equality with the default or a waiver of historical holds.
- Supervisor: **completed / exit0**, 142.626 s (300 s worker cap), 3.701 GiB
  sampled aggregate RSS, maximum 63 C and 224.52 W; owned group absent and CUDA
  released at completion. Resident process identity and both device leases were
  preserved. No training or pretrained model was loaded.

[Compact results, hashes and source pins](qualification.json) bind the local raw
JSON/JUnit evidence. Raw stdout, telemetry and tensor-free benchmark detail stay
outside Git. Independent core/math/serialization, integration, benchmark/docs
and supervisor source reviews closed with no remaining material findings.
Repository CI is tracked on the draft PR separately.
