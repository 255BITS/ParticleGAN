# Principal: img_colorize_lr2

Colorization-adjacent unpaired shade transfer on a shared 8×8 silhouette:

- mode0: left-bright / right-dim central patch (chrominance proxy)
- mode1: left-dim / right-bright (swapped photometric assignment)

Same spatial support, opposite left/right intensity assignment. Tests whether the published image GAN path can cover photometric channel-emphasis modes that residual upsampling can represent.

In-formulation control: `residual_upsample` width 16 under the same RpGAN + b_cap + 32-particle recipe. Not a diffusion / non-GAN baseline.
