# Can an ideal local ratio field leave a wrong subset of ring modes occupied?

The repaired cold ring acquires three mode clusters by update 100 and remains
on that subset through its failed 1,200-update gate. Its exact, passive
[first-100 replay](continuous-evidence/pr84-finite-cold-prefix100/manifest.json)
has four clean particles nearest each of modes 0, 1, and 7. I used its saved
generator and prior once, then evaluated a **population comparator**. The
[script](ideal_ratio_three_mode.py) and [receipt](continuous-evidence/ideal-three-mode/receipt.json)
contain the 12 support points, source and state hashes, every clean-point
vector, and noise-law bookkeeping. Two focused algebra tests pass. No D fit,
GAN update, new initialization, or search was run.

For independent real/fake pairs, the unrestricted logistic Rp critic has
score `r(x) = log p(x) − log q(x) + constant`: the ordered-pair log odds are
`r(real) − r(fake)`. This follows directly by symmetrizing the pair
classification objective; the [relativistic discriminator paper](https://arxiv.org/abs/1807.00734)
defines that paired game, while [Jolicoeur-Martineau (2020)](https://proceedings.mlr.press/v119/jolicoeur-martineau20a.html)
proves the associated divergence result. Neither paper makes this the
optimum of this repository's **finite MLP with `b_cap`**. The Gaussian
calculation below therefore tests what even perfect *unrestricted* ratio
information supplies locally, not what the actual penalized critic learns.

At update 100 the critic's input noise is `.0875` and fake output noise is
`.0119625`, so the two observed marginal widths are
`σp = sqrt(.07² + .0875²) = .112055` and
`σq = sqrt(.0119625² + .0875²) = .088314`. With the real eight centers and
the saved 12 generated centers, the population score gradient at each
**actual clean support** points toward its assigned occupied target center:
12/12. Seven of those same vectors also project toward a nearest empty-mode
chord. That projection is therefore compatible with within-mode correction;
it does not establish transport to an empty mode. The analytic mixture-score
formula agrees with autodiff to `3.6e−15` on these points.

I then made one controlled, posthoc copy of the cloud, moving each support
to its assigned occupied center while keeping its 4/4/4 multiplicities.
The target centers select and grade this copy only; they never enter a
training update. The ratio score knows the distribution is wrong: at
occupied mode 1 its value is `−1.457`, versus `+336.525` at empty mode 2.
Yet at the centered supports the largest clean-point score-gradient norm is
`9.35e−90`. The leading tangential cue at mode 1 toward empty mode 2 is
about `10^−144.35`, because the nearest *occupied* q component is 2.296
units away. At the evaluation HQ radius `.21`, its relative q mass is only
about `10^−119.93`. The empty-versus-occupied neighbor contrast reaches
approximately `log 2` only at the chord midpoint, 1.148 units from mode 1,
or 5.47 HQ radii. The ring's equally spaced real modes cancel the tangent
from `p`; the missing-specific local tangent comes from the exponentially
small overlap of occupied q components.

One local response calculation distinguishes a frozen critic from an
accurately *retracked* critic. Translate all four particles at mode 1 by
one real standard deviation (`.07`) along the chord toward empty mode 2,
still inside the HQ ball. The ordinary partial G score-ascent component is
`+3.400` toward mode 2 if `r` is held at the centered cloud, but `−5.575`
if `r` is recomputed for the translated q: the updated ratio pulls the
cluster back to its already occupied real mode. In the isolated-component
limit this is explicit. At the moving q center `y`, `∇ log q_y(y)=0`, so
the retracked partial field is `∇ log p(y)≈−(y−c)/σp²` near the occupied
real center `c`; the fixed-D field instead has local curvature
`σq⁻²−σp⁻²>0`. The Rp generator's logistic factor on a fixed fake point is
positive, so it preserves these score-ascent signs. This retracked *partial*
field is distinct from differentiating the generator objective **through**
a virtual D update, which the separate one-step unroll audit measures.

For scale only, applying the late zero-input-noise law (`σp=.07`, `σq=.029`)
to this **same frozen update-100 cloud** makes the centered clean-point
gradient `8.29e−232`, the leading missing tangent about `10^−1357.86`,
and the fixed/retracked one-sigma fields `+68.949/−14.286`. These are not
measurements of the trained update-1,200 generator. They also do not
integrate over noisy fake/real pairs or include the shared-network Jacobian,
Adam, the cap penalty, or the candidate's G bound. The positive isolated
fixed-D curvature means the wrong subset is not proved to be a stable
fixed-critic equilibrium; conversely, the retracked calculation is not a
proof of a stable finite-host equilibrium.

This isolates an acquisition issue from late stability: perfect ratio
values at empty modes can coexist with negligible *local* empty-mode
direction at centered generated modes, and accurate D retracking can
correct local displacement back into the wrong subset. [Mode Regularized
GANs](https://arxiv.org/abs/1612.02136) identifies the sparse visits to
missing modes as a separate coverage problem and adds an encoder-based
mode objective; that is motivation, not a theorem about this host.
[Unrolled GANs](https://research.google/pubs/unrolled-generative-adversarial-networks/)
reports improved coverage from anticipating D responses, but likewise
does not guarantee inter-mode transport in this 12-to-8 capped game.

The cheapest next saved-state diagnostic, if the one-step full-chain field
does not repair acquisition, is to measure its **tangential component after
removing within-mode centering** at the exact update-100 state. Use the
same real/fake minibatches and frozen Adam metric for the partial and
full-chain fields, then grade mode assignments only afterward. A positive
dot product toward an empty center by itself remains insufficient.
