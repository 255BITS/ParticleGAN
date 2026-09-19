# Validation

Subagent implemented a standalone trainer and ran11 tests, all passing in6.90 seconds:

```bash
CUDA_VISIBLE_DEVICES=0 RUN_CUDA_IMAGE_TESTS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_cifar_ae_deconv_wide_norm.py
```

Covers925,763 parameters, exact output geometry, hidden GroupNorm placement/no output norm, finite generator/latent gradients, E-only optimizer isolation, unchanged original D/E initialization RNG, unchanged historical deconv construction, valid fixedsigma/ignored-override rejection, strict architecture/sigma resume checks and CUDA bit-exact full-state split resume. Test-only deterministicD pooling substitution is confined to the test.

Root reviewed the trainer diff and ran the production pipeline smoke onGPU0:

```bash
.venv/bin/python -u experiments/cifar_ae_deconv_wide_norm_scout.py --smoke
```

Smoke passed: actual16,384-particle model,16 updates including lazy double backprop at8/16, FID128, test reconstruction, retained checkpoints and certified summary. Total44.5 seconds, training3.3 seconds. Smoke scores and throughput are not benchmarks. Frozen D features andsigma unchanged. D/E initialization SHA, actualsigma and initial prior calibration exactly match the original small-deconv run. Final losses/gradients finite. GPU1 simultaneously continued the original deconv from40k toward200k.

Production config differs from the completed small-deconv full config only in generator_arch, output path and wall-time cap; the orchestrator asserts this. No historical source files changed. New G jointly varies width and normalization, so any gain cannot be attributed to either alone.
