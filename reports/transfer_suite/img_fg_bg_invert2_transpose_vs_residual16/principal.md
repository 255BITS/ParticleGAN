# Principal: img_fg_bg_invert2

Figure-ground / photographic-invert unpaired polarity on a shared 8×8 support:

- mode0: bright central 4×4 foreground on dark background
- mode1: dark central 4×4 foreground on bright background (invert)

Same spatial support, opposite figure-ground polarity. Tests whether the published image GAN path covers invert / style-polarity modes that residual upsampling can represent.

In-formulation control: `residual_upsample` width 16 under the same RpGAN + b_cap + 32-particle recipe. Not a diffusion / non-GAN baseline.
