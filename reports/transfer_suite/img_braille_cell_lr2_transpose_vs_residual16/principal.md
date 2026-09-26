# Principal: Braille cell left-heavy vs right-heavy mass

**Track:** application (product-adjacent 8×8 image demo)

**Task:** `img_braille_cell_lr2` — Braille-style tactile cell with mass concentrated
LEFT (cols 1–2 raised dots) vs RIGHT (cols 5–6), sharing a faint middle vertical
guide. Distinct from barcode quiet-zone LR, finder diagonal pairing, diffraction
order LR, and sparse observation tasks: the cue is discrete tactile-grid asymmetry
with a shared guide, not a continuous quiet margin or barcode bars.

**Published winner (must FAIL):** image suite baseline **transpose12**
(RpGAN + b_cap + particles, published batchfeat path).

**In-formulation control (must PASS):** **residual_upsample width 16** — same
RpGAN + b_cap + particles family, different architecture card only.

**Solvability gate:** winner FAIL alone is insufficient; control PASS proves the
same GAN formulation can still solve the task.

**Harness:** 2 modes, 8×8 grayscale, hq_min=0.9, quality_rmse=0.05, observations=24,
minimum_stable_checks=5 (barcode HIT thresholds).

**Tip lock:** `ac83ce32b99bd488c2dd700786c0c93f0bb1f27b` (`origin/particle-finetune/base`). Never push base.
