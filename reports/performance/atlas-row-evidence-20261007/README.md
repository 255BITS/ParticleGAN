# Atlas row-evidence scale-search performance

The scaled row gate evaluates the same effective sample size in each of its
24 bisection trials. Hoisting its two-dimensional exponent and denominator
eliminates **115 repeated tensor operations per update**. Both divisions keep
their original order, so rounding, search decisions and resulting row flags
are preserved. Other dimensions retain the existing incomplete-gamma path.

The change is independent of the earlier detached output-sigma cache. The
comparison below uses that cache on both sides, with the immutable matched
real/fake word-noise host and all Atlas mechanisms enabled. Only
`RowEvidence._scale` is replaced after fixture construction; bindings, archived
source files and the full resolved recipe remain untouched.

## Measured results

CPU: PyTorch 2.13.0+cu126, Python 3.12.13, one Torch/BLAS thread,
deterministic algorithms, TF32 disabled, protocol seed 0. Baseline and optimized
run on the same machine with alternating order: baseline, optimized,
optimized, baseline. See [the compact receipt](cpu-results.json) for every
window and source/artifact hashes.

| Measurement | Baseline | Optimized | Observed time reduction |
| --- | ---: | ---: | ---: |
| Matched Atlas training, 256 updates after 8 warmups, mean of two windows | 25.265 ms/update | 24.409 ms/update | 3.39% |
| Scale-search microbenchmark, 5 rows, float64, 512 calls/window | 744.721 µs/call | 530.966 µs/call | 28.70% |
| Same microbenchmark, 25 rows | 739.596 µs/call | 507.612 µs/call | 31.37% |
| Same microbenchmark, 128 rows | 789.003 µs/call | 588.137 µs/call | 25.46% |

The complete-training effect is **small and noisy**: baseline windows range
from 24.503 to 26.028 ms/update; optimized windows from 24.339 to 24.479.
The microbenchmark and operator counts establish the local saving more clearly
than these four full-update windows establish a stable throughput percentage.
CPU measurements do not establish GPU acceleration.

The original 64-update CPU profile of this matched host attributed 16.7% of
time to birth/death processing, 4.7% to row evidence and 3.5% to its scale
search. Autograd and network forwards remain the main CPU costs. This patch
does not address the larger birth/death path.

## Correctness and screening scope

All tensor **storage bytes** and scalar values match across four 264-update
timing trajectories and paired 1,200-update continuations, including all loss
outputs, final parameter gradients, model/optimizer/controller state and
14 named RNG streams. Every training output is finite. The 1,200-update
trajectory naturally exercises 157 row-gate holds, 206 summed flagged rows,
1,200 birth/death evaluations and the later KA2 penalty phase. It has zero
particle moves; an additional public `GANTrainer` test explicitly exercises
stationarity, 16 moves/resets, blocked drift and anchored release.

Reference tests cover float32/float64, dimensions 1/2/3/8, no valid rows,
lower/upper search bounds, ties, NaNs masked as invalid, denominator clamping,
gradient-history evolution, resets and checkpoint continuation. A dedicated
CUDA test repeats the scale-search cases when CUDA is available.

Every fixture keeps its original **20,001-update execution limit and full
training schedules**, even when the driver stops early. No optimizer,
controller, noise law, RNG draw, objective, Atlas mechanism or quality gate
is changed. These runs are software/performance evidence and confer no
gauntlet qualification.

A 128-update window after eight warmups is a useful initial profiling size;
32 updates provide too little coverage of controller windows. Use repeated
alternating windows and extend toward 256 when differences are only a few
percent. Later-phase throughput still needs its own timed window. A
1,200-update quality screen is a separate decision: this implementation's
parity at 1,200 does not calibrate a safe early rejection rule or demonstrate
that 1,200 predicts the whole gauntlet.

## Reproduction

Run [benchmark.py](benchmark.py) from this checkout. It validates the original
adapter's frozen source/binding, constructs the actual public-components host,
then transplants this checkout's `_scale` method in memory. Pass the archived
inputs explicitly; they are not replaced by an approximate new fixture.

```sh
BASE=/ml2/hypergan/pg-atlas986-fiveword25-runtime-20261006/v1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B reports/performance/atlas-row-evidence-20261007/benchmark.py \
  --owner-root "$BASE/owner-source" \
  --adapter "$BASE/runtime-profile1077/v1/cached-adapter-source/v1/adapter.py" \
  --binding "$BASE/runtime-profile1077/v1/CACHED-BOUND-BINDING.json" \
  --output /tmp/atlas-row-evidence-cpu \
  --windows 256 --warmup 8 --stability-steps 1200

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q tests/test_row_evidence_scale.py \
  tests/test_e22_row_contract.py tests/test_e22_policy.py tests/test_k3p.py \
  tests/test_training.py tests/test_e22_routed_policy.py
```

Raw logs, profiler dumps and checkpoint tensors remain outside Git in the
task directory named by the receipt. GPU runs require an explicitly coordinated
physical UUID, one visible device, a 2 GiB allocator cap and an external wall
time/overlap supervisor. GPU0 was occupied; Last Epoch was verified and closed
gracefully with user authorization, while serving workloads remained running.
The first GPU1 launch was cancelled before any training when its coordinated
window expired. GPU measurement remains pending a clear science slot.

The existing repository workflows run on master/release, so a draft targeting
develop has no automatic PR test workflow. Local validation is recorded in the
receipt; absent GitHub checks must not be reported as passing CI.
