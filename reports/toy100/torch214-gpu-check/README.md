# PyTorch 2.14 does not fix the two GPU blockers

**All four 2.14.0+cu126 runs reproduce their 2.13.0+cu126 controls exactly in
all non-timing result fields.** The two original-recipe blockers still fail,
with both native CUDA initialization and copied CPU initial parameters.

| Initialization | Gate | PyTorch 2.13 | PyTorch 2.14 |
|---|---|---|---|
| Native CUDA | Ring | FAIL: 7 modes, HQ .997802734375 | Identical FAIL |
| CPU parameters, CUDA training | Ring | FAIL: 7 modes, HQ .9970703125 | Identical FAIL |
| Native CUDA | Unequal mass | FAIL: minimum eigenvalue ratio .02909558 | Identical FAIL |
| CPU parameters, CUDA training | Unequal mass | FAIL: minimum eigenvalue ratio .03095048 | Identical FAIL |

Both versions use the same RTX A6000, CUDA12.6, cuDNN91002, driver, original
scheduled recipe, seed0, 1200-update budgets, FP32, deterministic algorithms,
TF32 off, and one CPU thread. Actual parameters, gradients and Adam moments
remain CUDA. The complete random-draw receipts and initial parameter hashes
also match exactly. All four frozen verdicts and 2400 Adam updates per run were
independently checked. This is a two-blocker comparison, not a 22-toy score.

The upgrade was installed only in `/tmp/particlegan-torch214-cu126`, an isolated
venv. The original 2.13 benchmark environment is unchanged. The compatible
2.14 CUDA12.6 build uses the existing CUDA libraries; torch, triton and the
matching torchvision are installed in the new environment. These toy runs do
not use torchvision or torch.compile. [Environment](environment.json) and
[install plan](https://github.com/255BITS/ParticleGAN/blob/b979d3c90bdbf8f58c62d759bc3c5fa94cdf18c8/reports/toy100/torch214-gpu-check/install-plan.log) bind the tested stack.

The successful CPU recipe is sensitive to backend-dependent initialization,
random draws and arithmetic. CPU initialization alone already recovers four
of the six original GPU failures. Even identical initial parameters and random
draws leave ring and bars failing in the separate full-stream diagnostic.
PyTorch documents that CPU and GPU floating-point computations may differ even
with identical inputs ([numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)).
This evidence does not establish a broken CUDA kernel. Upgrading alone is not
an immediate solution to the remaining ring/rare-component failures.

[Exact 2.13/2.14 comparison](comparison.json) · [Four audited results](summary.json) ·
[Original backend controls](../cpu-recipe-gpu-port/README.md)

The exact executed command for each run is in summary.json; run.py is the
executed batch driver with recorded local paths. The unchanged probe and
checksum-verifying source preparer are in ../cpu-recipe-gpu-port. To replay a
single control, create a fresh source workdir with prepare.py, use the isolated
2.14 interpreter to run that probe with `--backend cuda`, and supply the retained
CPU initialization fixture only for the CPU-initialized profile. Keep
CUDA_VISIBLE_DEVICES=1 and CUBLAS_WORKSPACE_CONFIG=:4096:8, with the documented
single-threaded numerical profile. Do not rerun these completed controls without
a concrete new software or numerical change.
