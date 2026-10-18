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

## Actual Sana benchmark on the PR branch

Tested code head `90ae663551dd0fa35749d56e1859f276a4d09587` through the imported
public PG API, with an isolated source-bound Sana host profile. The original
installed host and old training cohorts were unchanged. Both disposable arms
restored the same CP500 numerical state; only source/profile identity changed.
The selected profile covers ordinary dense G and D parameters. Model, rows,
conditions, FP32 arithmetic, batch2, rates and hardware were matched.

| Measurement | Default | Guarded Gram | Speedup |
| --- | ---: | ---: | ---: |
| Frozen real all224-gradient public DirectionBlend step | 17.1004 s | 8.48881 s | 2.01447x |
| Native full training update | 19.2530 s | 10.0377 s | 1.91806x |
| Native G optimizer step | 17.7800 s | 8.54653 s | 2.08037x |
| Native D optimizer step | 0.0055134 s | 0.0127239 s | slower |

One warmup was excluded per arm; each median has two measured samples. Frozen
replay restores CP500 afresh each time. Native arms each perform three ordinary
updates: warmup501 and measured502/503. Timers synchronize at optimizer boundaries,
so these are instrumented full-update measurements. Both include the selected
same-factor guard. Warmup-only cProfile recorded112 guard calls and2.03475 s
cumulative Python time including GPU waits; it is not an exclusive CUDA cost or
an additive production overhead estimate.

Both arms started with training-row q `[0.31516149, 0.74955386]`. After three
updates, default reached `[0.31401577, 0.74785414]` and Gram
`[0.31401590, 0.74785539]`; losses stayed finite and penalties positive. This is
short learning sanity, not convergence, heldout transfer or image quality.
The old-default numerical fidelity holds remain; no trajectory parity claim.

Supervisor completed/exit0 in268.267 s under the300 s cap, with8.976 GiB sampled
RSS,15.166 GiB peak Torch-reserved GPU memory, maximum70 C and226.89 W.
Original checkpoint/source/base checks passed. Owned group was absent, CUDA
released and both leases available at postflight. No checkpoint or weights
were published and the old long training runs were not restarted.

The real Sana result is consistent with the earlier research2x measurement.
The1.15x synthetic result above uses a different, repeated full-rank gradient
bank and a plain NormalizedOptimizer; it does not predict Sana throughput.
[Compact measured results and artifact hashes](sana-speed.json).

## Repository compatibility follow-up

The24 new applied-recipe compatibility failures found by the first CI run were
fixed at90ae663 by omitting the unselected default field from historical receipts
after validating the full Recipe/ownership registry. The unchanged24 historical
comparisons and3 added receipt cases passed locally. CI then passed those cases
and exposed4 stale Forge scorer/source bindings. Current task cards append an
explicit source-only revision, retaining the complete previous card/revision,
thresholds, laws, budgets and archived variants. Historical fixtures are tested
with their exact archived Git objects; missing objects were hydrated rather than
removing checks. Seven failures reproduced on unchanged develop remain separately
tracked on the PR and are outside this numerical-profile change.
