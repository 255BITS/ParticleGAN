# Fixed generator-step sensitivity

This is one prospective configuration per family, Atlas and E22, with shared
overrides `lr=0.00265625`, `prior_lr_mult=3.0`, `d_lr_mult=4.5`. The nominal
generator and output-noise base step is half the previous critic-rate contrast.
Nominal prior and critic rates remain 0.00796875 and 0.011953125. Actual policy
rates are endogenous, so the test cannot isolate a guaranteed causal mechanism.

The source changes only this report-owned runner, a separate capacity binder,
and software tests. Training remains the existing public `api_run` path, using
unchanged public Recipe/optimizer/policy APIs and the maintained coordinator.
Generic snapshot, safety, cost and prefix-admission functions are called from the
unchanged previous report helper; its globals and frozen evidence are untouched.
That delegated helper is explicitly bound in the new source manifest and frozen
snapshot, and imported in the child bootstrap solely so the public receipt source
collector records its bytes. The import executes no verification or science.

Fresh capacity must record all 16 outcomes under the new exact recipe and source.
Each family needs its own eight SUPPORTED public-sampler witnesses. A legitimate
UNRESOLVED or BLOCKED capacity outcome blocks that family before training, while
its fully supported sibling may proceed. These zero-update witnesses are
necessary representation evidence and provide no learned-quality credit.

The ordered eight cases remain:

| Case | Tier | Full updates | Evaluation outputs/check | Acquisition cap |
| --- | --- | --- | --- | --- |
| intensity2, source transpose width 12 | Smoke | 600 | 1,024 | 180 s |
| two-broad vector | Smoke | 1,200 | 4,096 | 180 s |
| grid100 | Quality | 7,000 | 20,000 | 2,100 s |
| rotated100 | Quality | 7,000 | 20,000 | 2,100 s |
| staggered100 | Quality | 7,000 | 20,000 | 2,100 s |
| unequal-mass vector | Quality | 1,200 | 4,096 | 180 s |
| anisotropic vector | Quality | 1,200 | 4,096 | 180 s |
| bars4, source transpose width 12 | Quality | 600 | 1,024 | 180 s |

Batch sizes, initialization, seed 24002, held-out seed 34002, original numerical
gates, serving laws and all 24 primary post-update scoring observations stay
unchanged. Both smoke cases must pass before quality; the first non-PASS stops
the candidate. Unreached cases remain UNKNOWN in the required eight-case
denominator. Nine actual training GIF states retain the original public grade.
The adjacent readout also displays the added first-window persistence grade:
first five consecutive primary PASS checks, then at least five later checks,
with every later primary check passing. No reacquisition, horizon extension or
automatic failed retry is permitted. Original terminal grading remains separate.

Native primary scoring uses the declared noisy served law; clean outputs are
separate diagnostics. The 24 × 20,000-output scope does not inherit Atlas19's
independent 100,000-output certification. Particle-cloud exceptions retain their
own source/law identities and do not qualify current Forge MoG defaults.

The shared finite contrast ceiling remains **15,360 seconds**, including the
previous source-bound 59.116680497769266 seconds: original engineering error
4.757908704923466, Atlas scientific FAIL 27.0030500178691 and E22 scientific FAIL
27.3557217749767. All prior grades, artifacts and costs remain unchanged.
Remaining quotas are Atlas **7,648.239041277208**, E22 **7,652.6442782250235**, pair
**15,300.883319502231** seconds. Prior costs are bound through the previous
combined result, certification, published compact result, family studies and
durable supervisor receipts, and are charged once. They fill no new test cell.

Every launch needs its complete acquisition allowance plus 60 seconds export
grace. Insufficient remaining allowance yields INCOMPLETE; it cannot start a
shorter task. Paid supervised time and conservative interruption reserves stay
separate. CPU capacity replay, queue wait and parent certification are separate
diagnostics. No default adoption, fair speed ranking or cross-source pooling is
authorized.

Only root freezes sources, captures fresh capacity, reserves and launches. After
that freeze, the explicit commands are:

```sh
# CPU-only preparation; requires a new external output and committed binder.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python reports/forge/generator-step-20261003/bind_capacity.py \
  --output /EXTERNAL/NEW-CAPACITY

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python reports/forge/generator-step-20261003/run_generator_step.py \
  spec /EXTERNAL/NEW-CAPACITY/capacity.json --output /EXTERNAL/NEW-SPEC.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python reports/forge/generator-step-20261003/run_generator_step.py \
  plan /EXTERNAL/NEW-SPEC.json

# Explicit serial launch on the shared physical GPU1 coordinator only.
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python reports/forge/generator-step-20261003/run_generator_step.py \
  run /EXTERNAL/NEW-SPEC.json --family atlas --output /EXTERNAL/NEW-ATLAS \
  --queue-root /ml2/hypergan/ParticleGAN-single-recipe/runs/forge
```

E22 uses the same command with `--family e22` and a separate new output, after
root releases the serial lane. GPU1 must have at least 12,288 MiB free and be at
or below 82°C; one CPU thread and a 0.2 CUDA memory fraction are enforced.

Final read-only certification uses `combine ATLAS/study.json --archive
E22/study.json`; it explicitly replays the CPU capacity sampler and certifies
retained numeric traces, with no ordinary training. Run it at the frozen science
source before introducing another Python publication helper into that inventory.
Publish later from a separate checkout with exact source/artifact bindings.
