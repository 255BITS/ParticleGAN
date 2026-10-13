# Word-host performance diagnostic

The public joint word host spends most of its time computing nine full SVDs per update. An explicit CPU full-SVD backend reduces the measured training time by **2.48×** on the RTX A6000. Carry `optimizer_svd_backend="cpu"` into the declared phase-2 repair candidates, then measure the complete original word task against its unchanged 900-second limit.

This is a software diagnostic, with protocol seed 0 and the public deterministic initializer. Each timing uses the original joint objective, model, five-row prior, batch 256, actual named streams, 20,000-update schedule and 20,001-update execution cap. It measures 200 updates after 16 warmup updates, followed by 32 cProfile and eight Kineto updates. Each arm executes only 256 updates. No acquisition, hold, Tier-1 or Tier-2 credit is claimed.

| SVD path | ms/update | Projected training time for 20,001 updates | Recommendation |
|---|---:|---:|---|
| Native CUDA, unchanged default | 43.77 | 875.5 s | Retain as incumbent and compatibility control |
| CUDA `gesvd` driver diagnostic | 41.91 | 838.3 s | Reject: little remaining wall-time headroom |
| Explicit CPU full polar/SVD | 17.64 | 352.7 s | Include as declared numerical trainer change |

These extrapolations exclude ordinary evaluation, checkpoints and GPU contention. They support testing the complete task; they do not establish that it finishes or passes. The earlier native profile measured 44.03 ms/update and the exploratory CPU path measured 17.82 ms/update, consistent with the implemented option.

In the original 32-update cProfile, full SVD takes 0.966 of 1.354 seconds, **71.3% of wall time**. Kineto attributes **89.7% of self CUDA time** to SVD. Transport contributes about 1.03 ms/update, protected-gradient binding about 1.25 ms/update, and the two backward calls about 3.25 ms/update. The new path retains those original forwards, losses and backward calls. It computes the complete polar factor on CPU using the same dtype promotion, full SVD, cutoff and smoothing, then copies the direction back before the existing parameter update. Sampled-prior row normalization and bias normalization retain their original device operations.

CPU LAPACK and CUDA cuSolver can produce different floating-point polar directions. Those differences amplify during training: the matched 256-update snapshots differ in 34 tensor leaves, and native has two projection conflicts while CPU has zero. The maximum tensor difference is recorded in the compact receipt. This mode is a structural numerical trainer delta, with a fresh recipe identity and evidence cohort. The prefix's word scores and actual-training GIF are diagnostic observations; they do not replace sustained original gates.

The default remains `native`. Its recipe field and optimizer checkpoint field stay absent, preserving old packet identity. Active CPU checkpoints retain their backend and reject cross-backend loads before mutation. CPU-native inputs take exactly the old arithmetic path. The focused suite passes **213 checks in 4.96 seconds**, covering smoothing, dense and grouped Conv2d/ConvTranspose2d directions, G/E/D roles, sampled prior ownership, old default packets, rank deficiency, finite output and exact active CPU/CUDA resume. A public CUDA word checkpoint resumes with identical losses, observations, complete state and named streams.

An independent replay compares the actual archived source `79f7ddb512f0bc1e1457257e85c97adbe6d11bd1` with backend commit `eca024857`. Eight public CUDA word updates produce identical complete initial/final state, including ambient RNG, outputs and the original 1,024-sample observation. The longer 256-update retained profiles also match all consumed state exactly under the default; their unaligned ambient process starts are explicitly retained in the receipt. CPU backend state matches the exploratory CPU tensor path after projecting only the two declared backend configuration fields.

Reproduce with the project Python environment, from the repository root:

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
python reports/forge/bcap-tier1-performance/profile_word_host.py --backend native --output /your/archive/implemented-native > /your/archive/native-profile.log 2>&1
python reports/forge/bcap-tier1-performance/profile_word_host.py --backend cpu --output /your/archive/implemented-cpu > /your/archive/cpu-profile.log 2>&1
python reports/forge/bcap-tier1-performance/profile_word_host.py --diagnostic-driver gesvd --output /your/archive/gesvd-driver > /your/archive/gesvd-profile.log 2>&1
python reports/forge/bcap-tier1-performance/replay_native.py --output /your/archive/native-replay
python -m pytest tests/test_bcap_svd_backend.py tests/test_dualnorm_optimizers.py tests/test_dualnorm_smoothing.py tests/test_dualnorm_convolution.py tests/test_dualnorm_default_parity.py -q
```

The measured snapshots, raw profiles, JUnit, parity packets and actual-training GIFs are archived under `/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/performance`. Easy-tail logs are `implemented-cpu-profile.log`, `implemented-native-profile.log` and `focused-tests.log`. [receipt.json](receipt.json) pins their hashes and final metrics; bulk output is excluded from Git. Source changes are limited to the optional public recipe field/default projection, normalized optimizer/backend checkpoint binding, and Forge's structural field ownership. Canonical task source rebinding and full-budget execution belong to the parent's phase-2 freeze.
