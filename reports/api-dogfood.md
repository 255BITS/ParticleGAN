# Dogfooding the ParticleGAN API

The examples and six experiment trainers now consume the documented public API.
Training loops, models, data, diagnostics, EMA, and checkpoints remain owned by
each application. The reference remains [docs/api.md](../docs/api.md).

## Changes

- The Gaussian and five-mode examples use recipe factories for priors, losses,
  penalties, and G/D optimizers. Their separate prior optimizer is retained.
- Denoising, trajectory, and CIFAR trainers resolve their existing config fields
  into public recipes. Learned-noise parameter groups remain explicit additions
  to ordinary Adam optimizers.
- Regularizer-arm and sparse experiments use public primitives directly,
  preserving their specialized optimizers, penalty schedules, and diagnostics.
- Toy, sparse, trajectory, and image critics share `ucd_scores`. This new
  function selects scores from already computed logits, so custom models keep
  their existing forward signatures and checkpoint keys. `UCD` uses it too.
- The optimizer factory accepts Adam execution options such as `fused=True`.
  Repeated cosine schedules use `learning_rate_scale`; historical toy decay
  arithmetic is retained, including very short training horizons.
- Probes, rendering helpers, and provenance analysis use public names.
- `DrawSource` is a research compatibility adapter over `ParticlePrior` sampling.
  Its historical `table` key, initialization, and Gaussian/zero controls remain.
  The fresh-Gaussian toy control retains its reference table for reproducible
  snapshots. New applications use `ParticlePrior` or `GaussianPrior` directly.

All six trainer `DEFAULTS` mappings are unchanged. One existing config bug was
fixed: omitted `cache_condition` now resolves to false for pixel CIFAR critics
and true for pretrained critics. Direct CLI and grid launches resolve it the
same way. An explicitly unsupported setting still raises an error.

## Validation

Baseline: commit `4ead9f0` on branch `api`. Checks ran on two RTX A6000 GPUs,
with multiple independent jobs sharing each GPU. Existing seeds were retained;
no seed sweeps were run.

| Check | Result |
| --- | --- |
| Full regression suite | 209 passed, 22 subtests passed; 4 opt-in CUDA resume cases skipped in this invocation |
| CIFAR suite with CUDA resume enabled | 88 passed, including all 4 real-data resume cases |
| Existing config loading | 797 YAML/TOML files passed schema/default merging |
| Conditional config validation | Denoising, trajectory, and CIFAR configs additionally passed trainer validation and recipe construction |
| Distinct GPU smoke cases | 25/25 passed, plus baseline and final verification reruns |
| CIFAR checkpoint probes | 3/3 passed with finite JSON output |
| Built wheel outside checkout | TOML loop, standalone UCD gradients, optimizer options passed |
| Trainer default parity | All 6 mappings match the baseline |

Two existing test warnings remain: SciPy's deprecated FID `disp` argument and a test's
conversion of a gradient-bearing tensor to a scalar.

### Config coverage

| Trainer | Existing config files loaded |
| --- | ---: |
| Regularizer arms | 416 |
| Sparse | 181 |
| Denoising, including speed configs | 109 |
| CIFAR DDGAN | 57 |
| Trajectory | 25 |
| Gaussian example runner, including speed configs | 9 |

Loading all configs is distinct from training them. GPU training used a small
set of structurally different configurations:

| Family | Cases | Coverage |
| --- | ---: | --- |
| Toy/examples | 6 | Default TOML GAN, OAdam, lazy16, Linf, sparse UCD with class partitions, five modes |
| Denoising | 6 | Class and joint UCD, learned noise, concat/Gaussian prior, one-shot GAN, class-free GAN |
| Trajectory | 6 | MLP, temporal and hybrid critics, learned noise, one-shot and concat controls |
| CIFAR | 7 | Class UCD, concat, joint UCD with finite differences/lazy penalties, channels-last, NCSN++, flat generator, attention |

The [smoke manifest](api-dogfood-smokes.json) records each original config and
its exact overrides. Runs used 4–20 updates and smaller evaluation counts;
original architecture sizes and seeds were retained. CIFAR used real cached
images and its real FID evaluator. Resume tests separately stub the expensive
FID calculation while checking actual model and optimizer replay.

### Before/after comparisons

| Workflow | Comparison against baseline |
| --- | --- |
| CIFAR | Bit-for-bit model, EMA, Adam, and dedicated RNG state equality after 4 updates |
| Denoising: class UCD, joint UCD, learned noise | Bit-for-bit saved checkpoint and generated sample equality |
| Toy frozen/fresh Gaussian controls, regularizer-arm and sparse runs | Bit-for-bit checkpoint equality |
| Toy learned-particle example | Maximum absolute parameter difference `1.19e-7`; unchanged-code CUDA rerun showed the same maximum difference |
| Trajectory baseline and learned noise | Maximum checkpoint difference `1.19e-7`; maximum generated-sample difference `7.45e-8` |

Numerical comparisons used `rtol=1e-5, atol=1e-6` where CUDA accumulation was
not bitwise stable. Existing checkpoint layouts remain compatible. CIFAR's
source fingerprint enforcement still intentionally rejects cross-source resume;
this refactor does not bypass that provenance policy.

## Results and recommendation

All tested workflows pass execution and compatibility checks. Continue using
the public API for new experiments; keep configuration translation and custom
training behavior in the experiment files. The remaining adapters preserve
archived research semantics and contain no separate particle sampling algorithm.

These short runs do not support a model-quality leaderboard. In particular,
ten-sample CIFAR FID is only an evaluator smoke check. Existing performance
recommendations are unchanged; no training-speed conclusions are drawn from
concurrent jobs.

## Logs and reproduction

Local logs and artifacts are under `runs/api/dogfood/` (gitignored):

```bash
tail -F runs/api/dogfood/full-tests.log
tail -F runs/api/dogfood/toy/parity.log
tail -F runs/api/dogfood/denoising/jobs.log
tail -F runs/api/dogfood/cifar/smoke-jobs.log

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
RUN_CUDA_IMAGE_TESTS=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_cifar_resume.py
```

`config-audit.log`, `defaults-parity.log`, `wheel.log`, and `wheel-loop.log`
record aggregate checks. Each family directory contains individual job logs,
resolved configs, and checkpoint comparison results. Smoke cases can be rerun
by merging a manifest entry's overrides into its source config and passing the
result to the corresponding trainer; use a new output directory. The five-mode
entry calls `examples.five_modes.train(**overrides)` instead of loading a config.
