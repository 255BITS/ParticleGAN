# Selected GAN default

**`get_recipe()` / `get_recipe("gan")` now selects the winning formulation.**
The `100gaussians` alias and direct `Recipe()` construction resolve the same
GAN settings. This promotion was requested after the formulation reached
19/19 live behavioral toys with supported architectures.

**Subsequent matched comparison:** the public numerical presets score **8/19
new versus 5/19 previous** on identical test hosts. Keeping the established
host training settings fixed gives **19/19 new core versus 17/19 previous core**.
The promoted public preset itself is not an all-tests-pass result.
[Per-test leaderboard, full curves and exact comparison](../default_comparison/README.md).

| Setting | New GAN default | `gan_legacy` |
| --- | --- | --- |
| Adversarial objective | Rp logistic | Rp logistic |
| Gradient penalty | b_cap, coefficient 3, κ=1.25 | b_cap, coefficient 1, κ=1 |
| Particle spread regularization | .05 | 1 |
| Particle L2 | None | None |
| Adam betas | (0, .99) | (0, .999) |
| G / D / particle learning rates | .001 / .0015 / .01 | .0006 / .0009 / .006 |
| Schedule | Hold 60%, cosine toward 5% | Same |
| Generic resources | 20,000 particles, batch 256, 7,000 updates | Same |
| Sampling through GANTrainer | Live; EMA is explicit | Same |

Architecture remains independent of the formulation. The optional public
`LinearSkipDiscriminator()` supplies the rare-mode reference: two 96-wide
hidden layers, Softplus(β=5), two Fourier bands, and a two-parameter raw-input
linear branch initialized to zero. It is used by the 2D quickstart and has
10,467 parameters. You can supply any discriminator to `make_trainer`.

```python
from particlegan import LinearSkipDiscriminator, get_recipe

recipe = get_recipe()
D = LinearSkipDiscriminator().to(device)  # Optional flat-vector reference.
trainer = recipe.make_trainer(G, D)       # G and data belong to your application.
```

## Verification of the promoted API

The public default recipe, public discriminator and `Recipe.make_trainer`
reproduce **all 24 live/EMA checkpoints and every schedule action exactly**
against the archived rare-mode winner. Live passes six final checks, confirms
at update 1,150 and ends with minimum normalized variance .269769. EMA passes
four final checks and remains FAIL under the same sustained rule.

The replay overrides only recipe resources (256 particles, batch 128 and 1,200
updates); all loss, regularizer, learning-rate and Adam settings come directly
from `get_recipe()`. It retains the original host G, target, units, prior
initialization std .5 and fixed random streams. No extra seed search or metric
change is involved. This verification does not add another leaderboard point.

```bash
python -u -m benchmarks.transfer_suite.default_promotion \
  --output /tmp/default-promotion-check > /tmp/default-promotion-check.log 2>&1
tail -f /tmp/default-promotion-check.log
python -m reports.transfer_suite.default_promotion.verify
```

[Exact parity](replay/parity.json.gz) · [Resolved recipe](replay/recipe.json.gz) ·
[Optimizer parameter groups](replay/optimizer_groups.json.gz) ·
[Full live/EMA curve](replay/result.json.gz) · [Source archive](replay/source.tar.gz).

**194 focused tests pass**: [193 API/configuration/behavioral tests](tests.log)
plus [one historical-baseline pinning check](history-test.log). The public
critic matches the research critic's initialization, outputs and active-cap
parameter gradients exactly. The installed wheel is tested outside the checkout;
its quickstart runs and resumes with exactly matching training state. The
early-stop run makes one extra evaluation draw, so only its separate evaluation
stream and returned sample draw differ. [Installed-wheel audit](wheel-audit.json).

## Compatibility and evidence scope

`gan_legacy` preserves every previous GAN field. `gan_behavioral` preserves the
earlier named candidate. MoG, DDGAN, denoising and autoencoder recipes preserve
their complete selected settings; the shipped MoG experiment explicitly pins
its older core and Adam beta2. Historical stock comparison scripts use the
legacy recipe, and saved full recipe dictionaries restore through `Recipe(**d)`.
[Before](previous-recipes.json) · [After](current-recipes.json).

The 19/19 formulation result uses suitable architectures and each host's
declared training settings. The new reference D itself passes 3/6 data toys.
The selected numerical starting defaults are not a claim of one network or one
optimizer preset transferring unchanged to all applications. The existing
[leaderboard](../formulations/README.md) and all failed attempts remain intact.
