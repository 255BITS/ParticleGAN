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

The [valid-toy search](../valid_search/README.md) and
[focused rare-mode search](../rare_focus/README.md) bring the original b_cap3
recipe to **9/9 required + 10/10 practical = 19/19 live behavioral toys** with
suitable architectures. A wider/deeper D fixes overlap, Softplus(beta5) fixes
unequal width, and D96×2 Softplus5 with a raw-coordinate linear skip fixes the
rare 2% mode. Broad, anisotropic and spiral retain passing architectures.
Different toys use different D architectures; no single D passes all six data toys.
Loss, regularization, optimizer settings and update budgets remain fixed.
The wider D grows from 4,929 to about 35,000 parameters; Softplus retains the
original 4,929. No wall-time speedup is established.

The rare-mode winner adds two learnable parameters to the 10,465-parameter
D96×2 Softplus discriminator. It passes the final six measurements, starting
at step 950 and confirming at 1,150 of 1,200 updates. Its own full data profile
is 3/6: rare mass, broad and spiral pass; unequal width, anisotropic and overlap
fail. Those cases retain other supported architectures under the same recipe.
[Exact replays and diagnosis](../rare_focus/DIAGNOSIS.md) identified flattening
in the earlier failed run; the new winner is reproduced exactly on its original host.

| Formulation | Required live | Data toys | Image toys with supported architecture | Practical support |
| --- | ---: | ---: | ---: | ---: |
| **RpGAN logistic + b_cap3 / κ1.25 + prior regularization .05, no L2** | **9/9** | **6/6** | **4/4** | **10/10** |
| RpGAN logistic + b_cap10 / κ1.25 + prior regularization .05, no L2 | 8/9; not qualified yet | 4/6 | 3/4 | 7/10 |

**Second-host replication: 16/19.** Replaying every archived b_cap3 supporting
episode on another CPU host keeps source, torch 2.13.0, seed 0, settings and
thresholds identical. The result is 8/9 required and 8/10 practical: the required
eight-mode ring, the rare 2% mode and bars4 fail. Every episode differs from
its archive at the first measurement at floating-point scale. Both torch 2.13.0
builds agree bit-for-bit on the second host. The table above still counts the archived
live evidence; the 19/19 is exact on the original host but not host-robust.
[Replication audit](../host_replication/README.md).

**Scope revision:** the previous 7/16 becomes 7/10 because the user requested
that imposed training-condition variations be excluded from this PR's main
comparison. That recount changed no measurement or threshold; the subsequent D search
adds the three new practical passes shown above. The nine existing
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
witnesses remain intact. The scope recount reused existing results; the
subsequent valid-toy search archives all new runs separately. Production
defaults remain unchanged.

```bash
python -m reports.transfer_suite.formulations.build
```

The builder verifies that only the budget changes between the b_cap3 long runs.
`architecture_cell` rejects mixing recipes, resources or targets while allowing
G/D architecture changes. These are evidence checks, not behavioral gates.
[Grouping/protocol/suite tests](tests.log) · [Artifact validation](validation.json).
