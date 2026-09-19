# Validation

Subagent implemented the standalone G/D attention trainer and tests. Nine tests passed in7.69 seconds:

```bash
CUDA_VISIBLE_DEVICES=1 RUN_CUDA_IMAGE_TESTS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_cifar_ae_sagan.py -q
```

Verified nonidentity attention from initialization, gradients to every q/k/v/output projection, G16x16 placement, unchanged baseline model tensors and global initialization RNG, E-only reconstruction isolation, D attention registration/toggling while pretrained weights and BN statistics remain frozen, nonzero default-kappa bcap CUDA double backward, and bit-exact full-state split resume. Existing trainers and shared sources unchanged. G930,883 parameters; G and D each add5,120 attention parameters at production width.

Root reviewed the diff and ran:

```bash
.venv/bin/python -u experiments/cifar_ae_sagan_scout.py --smoke
```

Certified actual16k-particle production-path16-update smoke passed, including lazy bcap, FID128, reconstruction and saved checkpoints. Attention gradient norms(G,D) were(0.10218,0.13931) atstep1 and(2.18603,0.08541) atstep16. All recorded attention gradients finite/nonzero. Actual bcap penalty positive atstep16(0.03727); step8 was below the cap and returnedzero, as intended. Frozen features/sigma unchanged. Metadata confirms fixed unit residual, no gate and no phase-in on both sides. Smoke scores are not benchmark results.

Full scout launched onGPU1 with pipelinePID292013. Production config changes only generator_arch and output path versus the wide GroupNorm no-attention baseline, which completed40k FID50k18.2285 with a valid current-source certificate. No new spectral normalization or SAGAN loss/rate changes. This is an attention adaptation within the current recipe.
