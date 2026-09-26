# img_braille_cell_lr2 — transpose12 vs residual16

Break episode: published transpose12 FAIL vs residual_upsample width-16 PASS on a
novel Braille cell left/right mass principal.

Reproduce:
```
cd /home/mikkel/sliders-outscore/ParticleGAN
git checkout ac83ce32b99bd488c2dd700786c0c93f0bb1f27b
CUDA_VISIBLE_DEVICES= /home/mikkel/anaconda3/envs/conceptmod/bin/python -u \
  /home/mikkel/sliders-outscore/break_tests/img_braille_cell_lr2_transpose_vs_residual16/reproduce_arms.py
```

See `principal.md` and `summary.json`.
