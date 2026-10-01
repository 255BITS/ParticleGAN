# K3P: the previous default formulation

K3P was the package default in 0.8.0. The default is now [KA2](ka2.md), which
keeps K3P's loss, optimizers, schedules and noise and changes only how the
critic penalty blends and how its EMA critic tracks. K3P passed all 22 declared
toy gates plus the ring hold and its extension
([evidence](../reports/toy100/k3p-base/README.md)).

## The penalty

With input dimension `d`, coefficient `c` and cap `κ` (both 1 by default):

```text
A = mean(||∇D(real)||² / d) + mean(relu(||∇D(fake)|| / √d − κ)²)
B = mean(relu(||∇D(real)|| − κ)²) + mean(relu(||∇D(fake)|| − κ)²)
P = mean(||∇D(real) − ∇D̄(real)||² / d)        D̄ = parameter EMA of D (decay .999)
s = max(0, min(1, 2r) − 2f) / (1 − 2f)          r = last critic LR / max critic LR
penalty = c/2 · (s·A + (1 − s)·(B + P))
```

`f` is `network_lr_floor` (.01). While the critic LR is at its peak, `s = 1`
and the penalty is exactly `A`. As the LR anneals, it hands over to the
one-sided caps plus the anchor term `P`. The anchor starts at the first blended
call. With a constant LR, `s` stays 1.

| | K3P | KA2 |
| --- | --- | --- |
| blend | `s` from the critic LR, 1 → 0 as it anneals | pure `A` for 799 calls, then a fixed `s = .5` |
| anchor weight | always 1 | gate `W` from the critic's Adam moment surprise |
| EMA critic decay | fixed `reg_anchor_decay` .999 | adaptive, between 1 and `reg_anchor_min_decay` .9, with reseeds |
| checkpoint schema | 3 | 4 |

## Replaying K3P

`benchmarks.legacy.recipe.LegacyRecipe` pins the K3P critic (`name="k3p"`,
`reg_anchor_decay`, `particlegan.k3p.K3PCriticAdam` and the pinned multi-arm
penalty in `benchmarks/legacy/grad_regularizers.py`). Archived K3P
configurations resolve through it, and it works with `GANTrainer` and the
recipe's own-loop factories:

```python
from benchmarks.legacy.recipe import get_recipe

recipe = get_recipe(total_steps=steps)          # K3P critic, otherwise the package defaults
trainer = GANTrainer(recipe, G, D)
```

`tests/test_k3p.py` checks this path bit for bit against the frozen K3P
research mechanism. K3P trainer checkpoints (schema 3) load only in the release
that wrote them (0.8.0).
