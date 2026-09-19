# Validation

Subagent implemented the standalone trainer and ran:

```bash
CUDA_VISIBLE_DEVICES=1 RUN_CUDA_IMAGE_TESTS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_cifar_ae_deconv.py
```

Result: **10 passed in 6.42 seconds**. Covers output geometry, absence of normalization, latent gradients, E-only reconstruction optimizer isolation, identical D/E initialization and global RNG relative to the original CNN, fixed-sigma validation/metadata, unsupported architecture overrides, and bit-exact full-state checkpoint resume. Resume test uses deterministic execution and equivalent fixed average pooling in D; these switches are test-only.

Root reviewed the trainer diff and ran the actual production-path GPU 1 smoke:

```bash
.venv/bin/python -u experiments/cifar_ae_deconv_scout.py --smoke
```

Passed and certified against current sources/config. Real CIFAR, 16,384 independent particle centers, full-size 298,595-parameter G, 16 updates, bcap at8/16, FID128 at8/16, reconstruction, EMA, and checkpoint outputs. Frozen D features and fixed sigma unchanged; G/E/prior gradients finite and nonzero. Smoke total42.7 seconds, training3.2 seconds; throughput is not representative because initialization dominates. Smoke FID is not a benchmark.

Full run launched on GPU 1 with PID273670, verified through step300/40000 with finite losses and intended configuration. Both GPUs active, GPU 0 continues its existing job. New trainer is standalone and no historical source files changed.
