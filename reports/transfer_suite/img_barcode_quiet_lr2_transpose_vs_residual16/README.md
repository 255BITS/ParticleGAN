# img_barcode_quiet_lr2 — published transpose12 FAIL / residual16 PASS

Break episode for ParticleGAN `particle-finetune/base` tip `ac83ce32b99bd488c2dd700786c0c93f0bb1f27b`.

## Reproduce
```bash
cd /path/to/ParticleGAN
CUDA_VISIBLE_DEVICES= python -u reports/transfer_suite/img_barcode_quiet_lr2_transpose_vs_residual16/reproduce_arms.py
```

## Solvability
Winner FAIL alone is insufficient; residual16 in-family control PASS verifies the same GAN formulation can solve the task.
