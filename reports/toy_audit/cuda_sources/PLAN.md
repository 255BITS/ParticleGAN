Four source entries whose CPU attempts were blocked by CUDA will receive one
separate CUDA attempt each. The repaired runtime prerequisite authorizes this
new cohort; prior blocked receipts and the original 109-case catalog remain
unchanged. Scientific source is exported exactly from
`6ec7e5788e14ea15ddc3e16ac71110458108b6a6`.

| Entry | Original configuration | Seed | Batch | Full update budget |
| --- | --- | --- | --- | --- |
| source-family-02 | denoising/diagnostics/ddgan_class_free_28k_s24002.yaml | 24002 | 256 | 28,000 |
| source-family-03 | denoising/default.toml | 24002 | 2,048 | 7,000 |
| source-family-04 | trajectory/default.yaml | 24002 | 128 | 10,000 |
| source-family-05 | trajectory/diversity/confirm_10k/mlp_continuous.yaml | 24002 | 128 | 10,000 |

Each entry has a 120-second supervised subprocess allowance, including Python
and CUDA startup, three unobserved source update pairs, three observed source
update pairs, and at most one full-budget attempt. An internal deadline stops
updates three seconds before the supervisor cap to leave time for receipts.
Source export is preparation outside the subprocess allowance, as in the
original capped conditional runner. Preparation and actual subprocess times
remain separate. Original relative output paths are preserved by executing
each phase in its own raw working directory. There are no config/recipe/source
patches, batch reductions, changed seeds or retries.

The pure observer preserves Python, NumPy, CPU and every visible CUDA global
RNG, plus the original nested named generators. It verifies modules, buffers,
optimizer state, gradients, modes and requires-grad flags. Four source boundary
hashes must match before the quality attempt; that attempt also verifies its
initialization against the baseline. Prefix checks preserve the complete
schedule/budget. A failed source/API/parity prerequisite stops that entry.

Denoising observations retain the original full marginal sampler and all
original reverse-transition probes (four observations, all classes and times,
the configured 256 rows per panel). The separate existing analytic gate uses
4096 clean draws at class0, observation0 and diffusion step2/alpha_bar .5,
seed99123, live and EMA independently. This panel does not establish all
class/time/observation posteriors. Its thresholds remain the existing
`conditional-source-distribution-v1` law; PASS also requires complete original
endpoint execution and five passing terminal observations.

Trajectory observations retain all original train/test geometries and class
contexts, 512 rows per context, chunk size256, and the original DDGAN sampling
order with fresh prior particles and Gaussian reverse noise. Source route
mass/support/collision/boundary/diversity diagnostics and finite-reference
floors are reported without inventing an aggregate acceptance threshold.
Original source scientific status is `NO_FROZEN_GATE` for both families.

Held-out scores never enter training, native guards, structural moves or budget
selection. Non-full outcomes remain INCOMPLETE or exact API BLOCKED errors,
never a convergence PASS. Initial and scheduled states use completed updates
0,1,10,25,50,100, then multiples of250 and the exact final budget. GIFs contain
only actual captured states. Raw stdout, JSONL streams, optimizer checkpoints
and source snapshots remain in the external CUDA artifact cohort.

Root coordinates the device and executes the following commands from this
worktree. `CUDA_VISIBLE_DEVICES=0` exposes exactly one GPU; each command finishes
before the next starts. The software CUDA controls must pass first. No GPU job
was launched by the observer implementation agent.

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/pr155-e22-venv/bin/python -m pytest -q tests/test_toy_source_cuda_observation.py

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/pr155-e22-venv/bin/python -u -m benchmarks.toy_audit.source_cuda_observation \
  --entry source-family-02 \
  --output /ml2/hypergan/toy-source-cuda-observation-20261001/source-family-02

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/pr155-e22-venv/bin/python -u -m benchmarks.toy_audit.source_cuda_observation \
  --entry source-family-03 \
  --output /ml2/hypergan/toy-source-cuda-observation-20261001/source-family-03

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/pr155-e22-venv/bin/python -u -m benchmarks.toy_audit.source_cuda_observation \
  --entry source-family-04 \
  --output /ml2/hypergan/toy-source-cuda-observation-20261001/source-family-04

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/pr155-e22-venv/bin/python -u -m benchmarks.toy_audit.source_cuda_observation \
  --entry source-family-05 \
  --output /ml2/hypergan/toy-source-cuda-observation-20261001/source-family-05

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/pr155-e22-venv/bin/python -m benchmarks.toy_audit.source_cuda_observation_report \
  --artifacts /ml2/hypergan/toy-source-cuda-observation-20261001 \
  --output reports/toy_audit/cuda_sources
```

Each raw `execution.log` can be tailed under its entry directory. The fresh
`coverage.json`/README and GIFs will be supplemental publication files. Their
records retain catalog IDs, original/added scientific statuses, exact
source/config/recipe/runtime binding, caps, failures, actual frame indices and
external archive identities. No Forge/default promotion follows this cohort.
