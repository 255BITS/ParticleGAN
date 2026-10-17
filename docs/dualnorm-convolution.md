# Opt-in DualNorm convolution updates

`Recipe.optimizer_convolution="per_offset"` supports `nn.Conv2d` and
`nn.ConvTranspose2d` kernels through the public Recipe factories and
`GANTrainer`. The dataclass default is `"none"`, retaining the original unsupported-kernel
error and all existing dense/vector/prior updates and checkpoint packets.
The named `bcap` preset enables `"per_offset"` as part of its
[selected configuration](../reports/forge/bcap-tier2-search/DEFAULT_SELECTION.md).
Only the `dualnorm` optimizer family supports this adaptation.

```python
from particlegan import GANTrainer, get_recipe

recipe = get_recipe("bcap", optimizer_convolution="per_offset",
                    optimizer_smoothing=1e-5)
trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=0)
# The module-based component factories also bind kernel layout automatically:
opt_g = recipe.make_generator_optimizer(generator)
opt_d = recipe.make_critic_optimizer(critic)
```

Move modules to their intended device and apply the public deterministic
initializer before constructing optimizers. Rates retain their declared
schedule; the BCAP preset uses constant G/E `.012`, D `.018` and prior `.03`.
This option adds no learning-rate annealing, random draws or initialization
change. The earlier Tier 1 selection used smoothing `1e-5`; the current named
BCAP preset uses `0.001` and non-saturating loss. The completed search passes
stripes but fails bars, blobs and intensity, so convolution support alone does
not establish image quality.

## Kernel rule

For each channel group and spatial offset, form a channel matrix
`M = gradient[out_channels/group, in_channels/group]`. Apply the existing exact
reduced-SVD polar update to this matrix, including its numerical-rank mask and
optional fixed singular-value smoothing. Multiply each direction by

`sqrt(out_channels / in_channels) / (kernel_height * kernel_width)`.

This factor has **no** `max(1, ratio)` clamp. The original dense-matrix rule
retains its existing clamp. Each channel group receives its own SVD; unrelated
groups and spatial offsets never share a numerical-rank threshold.

Conv2d stores weights as `[out_channels, in_channels/group, height, width]`.
ConvTranspose2d stores `[in_channels, out_channels/group, height, width]`.
For transposed kernels, transpose each stored channel slice into output/input
coordinates, apply the update, then transpose it back into storage coordinates.
Rectangular kernels and unequal input/output channel counts are supported.

The unsmoothed Conv2d factor follows the spatial-maximum, channel-RMS norm
derivation in [Modular Duality in Deep Learning, section 4](https://arxiv.org/html/2410.21265#S4).
The grouped and transposed rules here extend that derivation: channel groups are
independent blocks, and transpose stride inserts zeros before the kernel acts.
Zero insertion, padding crops and output selection do not increase the spatial
maximum norm. Kernel-area scaling is a conservative bound even when stride or
dilation prevents offsets from overlapping. It does not normalize the full
finite-image convolution operator's Euclidean spectral norm.

Smoothing uses `s / hypot(s, lambda)` in each retained singular direction. Its
operator norm is at most the unsmoothed direction's norm, so the same bound
holds. Applying smoothing to convolution slices is our extension of the
[matrix feedback analysis](https://arxiv.org/html/2608.01911v1); these geometric
bounds provide no adversarial convergence or image-quality guarantee.

## Momentum, epsilon and checkpoints

Momentum, if enabled, is accumulated in the kernel's original stored tensor
shape before forming channel matrices. For each channel matrix, a current
gradient norm or accumulated direction norm below the group's `eps` skips that
slice. Thus a zero current slice stays fixed even with nonzero old momentum.
Momentum still decays, and the per-parameter step counter advances as in the
existing dense rule. The SVD rank mask and low-precision computation policy
remain the existing `polar_factor` rules.

The factories split eligible kernel weights into individual parameter groups,
preserving role, rate, beta and epsilon settings. Generator, encoder and critic
modules are inspected explicitly. Bias/vector groups and sampled-prior groups
retain their original update rules. Bare parameter iterables cannot identify
kernel layout: unlabelled rank-three or higher tensors remain rejected, even
when the adaptation is enabled. Conv1d, Conv3d and custom high-rank tensors are
unsupported. Shared parameters with conflicting module layouts are rejected.

Enabled checkpoints record the convolution mode, and every kernel group records
`update_version="per_offset_polar_v1"`, storage layout, channel groups, input and
output channels, and kernel shape. Restore validates these fields and histories
before changing live state; same-shaped ordinary/transposed kernels cannot be
interchanged. Default `"none"` is omitted from Recipe and optimizer packets.
Nonconvolution default models retain their old group layout and packet identity.

## Software verification

The CUDA tests check independent SVD algebra, grouped/transposed/nonsquare/1x1
kernels, narrowing-channel scaling, local rank-zero and epsilon skips, momentum,
the spatial-maximum/channel-RMS operator bound and its attainment, prior-row
isolation, incompatible-checkpoint rejection and exact public-API continuation.
Existing CUDA default-parity tests retain the old dense packet and trajectory
comparison. These checks establish software behavior; the separately declared
Forge image runs provide scientific metrics and actual-training GIFs.

```sh
python -m pytest -q tests/test_dualnorm_convolution.py \
  tests/test_dualnorm_optimizers.py tests/test_dualnorm_smoothing.py \
  tests/test_dualnorm_default_parity.py
```

Raw verification logs stay outside Git in `runs/dualnorm-convolution/`.
