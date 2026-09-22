# Formulation defaults: behavioral leaderboard

**Each candidate chooses its training recipe. Architecture is separate.**
The recipe includes the adversarial loss, regularization, learning rates, Adam
settings, schedule, update balance and batch size. The current comparison tests
that recipe on behavioral problems. Imposed faster/slower discriminator rates,
smaller batches and discriminator widths do not count as additional failures.

Discriminator architecture may vary within the same formulation entry. G and D
choices are recorded independently. A supported architecture can satisfy a toy;
all tested architectures and their failures remain visible. Different training
recipes cannot be silently combined to produce one architecture result.

## Main comparison

| Formulation | Required live | Data toys | Image toys with supported architecture | Practical support |
| --- | ---: | ---: | ---: | ---: |
| **RpGAN logistic + b_cap3 / κ1.25 + prior regularization .05, no L2** | **9/9** | **3/6** | **4/4** | **7/10** |
| RpGAN logistic + b_cap10 / κ1.25 + prior regularization .05, no L2 | 8/9; not qualified yet | 4/6 | 3/4 | 7/10 |

**Scope revision:** the previous 7/16 becomes 7/10 because the user requested
that imposed training-condition variations be excluded from this PR's main
comparison. No run improved and no metric threshold changed. The nine existing
required behavioral regressions, including the required ring test, remain.
Longer training is a separate toy below; the other five former dynamics rows
remain [nonblocking diagnostic evidence](DIAGNOSTICS.md).

Transpose12 and residual16 are architecture observations inside the b_cap3
entry. Residual16 changes G to residual upsampling and increases G/D width to
16, so it does not isolate a discriminator-only effect. It solves all four
image toys with the same formulation and training settings at 600 updates.

[Main architecture matrix](MATRIX.md) · [Exact settings, metrics and artifacts](leaderboard.json).

## Separate toy: longer training

Train on eight Gaussian clusters for increasing budgets, keeping the recipe and
architecture fixed. Report sustained live performance and convergence time at
each budget, separately from the main ten-toy count.

| Formulation | 2,400 updates | 4,800 updates | 7,200 updates |
| --- | --- | --- | --- |
| b_cap3 | FAIL | FAIL: only four final passing checks | **PASS: six final passing checks** |
| b_cap10 | FAIL | Not run | Not run |

These are existing independent runs; cosine spans each run's declared budget.
The b_cap3 rows change only that budget. [Full metrics and convergence](LONG_TRAINING.md).

The old “nominal ring” name meant the ordinary eight-Gaussian reference run at
1,200 updates. It is now only an archived diagnostic, with its original
[results and sources](nominal_ring/README.md). It adds no new selection gate.

## Evidence and reproduction

This view retains existing host recipes, including their different base learning
rates and Adam betas. It does not yet demonstrate one universal numerical
optimizer preset. A new recipe must declare its choices and be compared across
the same main toys. Architecture variants stay within that recipe's entry.

All results use seed 0 and unchanged live thresholds. PASS requires every metric
for the final five of a complete 24-observation curve. EMA is separate.
Historical controller-study rankings and the earlier 16/16 individual solver
witnesses remain intact. No training was rerun for this scope change and no
production defaults changed.

```bash
python -m reports.transfer_suite.formulations.build
```

The builder verifies that only the budget changes between the b_cap3 long runs.
`architecture_cell` rejects mixing recipes, resources or targets while allowing
G/D architecture changes. These are evidence checks, not behavioral gates.
[Grouping/protocol/suite tests](tests.log) · [Artifact validation](validation.json).
