# Larger counts converge near FID17;16k leads the final endpoints narrowly

Both20k→40k continuations completed and passed source/config certification, frozen-feature/sigma checks, optimizer rate checks and original data/noise RNG pairing. Existing4096 duration results were reused. Same recipe and checkpoints, no seed repetitions or architecture changes.

| Particles | FID20k | FID25k | FID30k | FID35k | FID40k | Additional training minutes |
|---|---:|---:|---:|---:|---:|---:|
|4096|18.5207|17.8876|17.4796|16.5033|17.2350|14.66|
|8192|18.1602|17.5680|17.6849|17.7297|17.7683|14.43|
|16384|18.0136|17.3601|17.5677|17.4324|17.0982|14.77|

At40k,16384 beats8192 by0.6701FID and4096 by only0.1368. The latter difference is too small to establish a strong advantage from one trajectory. No new run beats the best previously observed16.5033 at4096/35k. The16k endpoint remains4.0982 above target13; the best observed score is3.5033 above it.

The8192 run improves to25k then worsens slightly at every remaining observation. The16384 run improves at the last two observations (30k→35k→40k), which makes it the more promising continuation candidate. This is a recent trend, not proof that duration will reach13. The early monotonic count ranking at20k did not become a large monotonic advantage with further training.

All particles are selected: cumulative direct draw counts since expansion average468.75 per8k row and234.375 per16k row, with minima380/176. These count bothD andG draws; standardization/regularization also distribute gradients to unselected rows.16k sibling latent RMS0.1832 (~0.862sigma),8k0.1575 (~0.741sigma). Coupled-noise feature differences persist. Neither exposure nor latent distance is a semantic-coverage measure.

Both final sample grids inspected: varied objects/scenes with continuing shape/detail errors; no total-collapse claim. New endpoint information/density/coverage comparisons are stored separately in ../particle_scaling_40k_information/.

Recommendation: continue16k as a bounded duration test, for example40k→80k withFID50k every10k, before another large count increase. The8k trajectory provides little reason for further compute at this setting. Keep the comparison against the existing4k curve and retain the distinction between best observed and finalFID. No next training job was launched for this completion review.
