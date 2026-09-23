# Fixed-recipe smooth ensemble discriminator screen

This screen holds the `shared_c6` recipe and all host data, generator,
initialization seed, resources, budget, evaluation checkpoints, metrics, and
thresholds fixed. It varies only the pointwise discriminator. The full recipe
is Rp logistic, b_cap coefficient 6 and kappa 1.25, prior spread .05, no
particle L2, Adam (0, .99), absolute G/D LR .00425 and particle LR .0085,
with cosine annealing from 60% to a 5% floor. Architecture may differ across
tasks under this one unchanged recipe.

The [initial plan](plans/shared_ensemble_screen.json) freezes 12 generic cards
before either remaining test is run. Each branch is an independently
initialized smooth MLP fed coordinates at a fixed declared scale. Branch
scores are summed and multiplied by the declared score scale. One card adds
fixed noncommensurate axis harmonics. No card uses target means, labels,
density estimates, batch statistics, or evaluation feedback. Input and score
scales change model parameterization; they do not change optimizer settings.

| Card | Scales | Branch network | Score scale | Extra features |
| --- | --- | --- | ---: | --- |
| ensemble2_equal_softplus64_l2 | 1, 1 | 2 × Softplus 64 × 2 | 1/√2 | — |
| ensemble2_multiscale_softplus64_l2 | .5, 2 | 2 × Softplus 64 × 2 | 1/√2 | — |
| ensemble3_multiscale_softplus64_l2 | .5, 1, 2 | 3 × Softplus 64 × 2 | 1/√3 | — |
| ensemble3_multiscale_silu64_l2 | .5, 1, 2 | 3 × SiLU 64 × 2 | 1/√3 | — |
| ensemble2_multiscale_silu64_l3 | .5, 2 | 2 × SiLU 64 × 3 | 1/√2 | — |
| ensemble3_broad_softplus48_l3 | .75, 1.5, 3 | 3 × Softplus 48 × 3 | 1/√3 | — |
| ensemble4_broad_silu48_l2 | .5, 1, 2, 4 | 4 × SiLU 48 × 2 | 1/2 | — |
| ensemble2_highscale_softplus96_l2 | 1, 3 | 2 × Softplus 96 × 2 | 1/√2 | — |
| ensemble2_mixed64_l2 | 1, 2 | Softplus 64 × 2 + SiLU 64 × 2 | 1/√2 | — |
| ensemble2_multiscale_halfscore64_l2 | .5, 2 | 2 × Softplus 64 × 2 | .5 | — |
| ensemble2_multiscale_fullscore64_l2 | .5, 2 | 2 × Softplus 64 × 2 | 1 | — |
| ensemble2_multiscale_spectrum64_l2 | .5, 2 | 2 × Softplus 64 × 2 | 1/√2 | axis frequencies .5π, 1.5π |

The hypothesis is that independent smooth branches at different coordinate
scales can supply local spread feedback without making the critic's whole
score depend on one representation. This is a testable architecture hypothesis,
not a guarantee that the generator can preserve the 2% component's narrow
variance under the fixed recipe.

Run from the repository root with a new output directory:

```sh
/tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_ensemble_search \
  --plan benchmarks/transfer_suite/plans/shared_ensemble_screen.json \
  --output /tmp/shared-ensemble-screen > /tmp/shared-ensemble-screen.log 2>&1
```

The runner snapshots source before training, verifies the hashes before every
episode and afterward, and writes each complete live/EMA curve, actual
optimizer receipts, original/effective specs, architecture card, and verdict
to an importer-ready `index.json` and `protocol.json`. All failures remain in
the output. A live PASS requires the 24 observations and at least five final
passing observations; EMA is reported separately. The two-case screen is
incomplete for the full 19-case profile.
