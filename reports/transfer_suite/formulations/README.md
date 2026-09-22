# One entry per formulation; architecture is a separate axis

**Discriminator architecture may vary without creating another formulation.**
Generator architecture is also recorded separately. A supported architecture
can establish that a formulation passes a problem; failures on other tested
architectures remain visible in the architecture matrix rather than vetoing
that success. Each problem counts once.

The formulation fixes the adversarial loss/mode, penalty and its coefficients,
and particle-prior regularization. Loss or regularization changes cannot rescue
another formulation's row. For selecting stable defaults, LR, Adam settings,
schedule and update ratio also stay fixed within an architecture comparison.
Changing those settings creates a separate settings trial under that formulation.
More steps, particles or batch size are reported as resource changes.

## Formulation leaderboard

| Formulation | Required live | Data | Dynamics | Images with a supported architecture | Practical support |
| --- | ---: | ---: | ---: | ---: | ---: |
| **RpGAN logistic + b_cap3 / κ1.25 + prior regularization .05, no L2** | **9/9** | **3/6** | **0/6** | **4/4** | **7/16** |
| RpGAN logistic + b_cap10 / κ1.25 + prior regularization .05, no L2 | 8/9; not qualified yet | 4/6 | 0/6 | 3/4 | 7/16 |

Transpose12 and residual16 are **architecture observations inside the same
b_cap3 entry**. The original architecture profile passes 4/16; the supported
residual image architecture brings that same formulation's demonstrated support
to 7/16. It does not create a new formulation or show an optimizer improvement.

The residual image result changes G to residual upsampling and increases both
G and D width to 16. It is not evidence that changing only D solved the images.
Discriminator-only trials are allowed by the grouping rule; the exact G and D
architectures are recorded independently.

The b_cap10 entry currently lacks a required ring pass on its tested architecture.
A successful discriminator variant using the same formulation and training
settings could repair that cell. It cannot borrow b_cap3 or R1+R2 results.

[Full architecture matrix](MATRIX.md) · [Exact axes, verdicts and artifact references](leaderboard.json).

## Corrected formulation-changing condition

The historical `stress_r1_r2` condition hardcoded R1+R2. Those observations are
valid R1+R2 evidence, but cannot belong to a fixed-b_cap entry. This view uses
the same target, model, data streams, metrics and budget as a **nominal ring**
condition and runs each b_cap formulation on it. Both b_cap3 and b_cap10 fail
at 1,200 steps, so the numerical totals remain unchanged. The two new complete
runs and exact sources are [retained here](nominal_ring/README.md).

Historical R1+R2 runs, the 16/16 individual solver witnesses and the original
controller study remain intact. They are not silently reclassified as b_cap
successes. The old architecture-profile table remains available in its exact
machine-readable archive; the current leaderboard groups those profiles.

## Limits and reproduction

This grouping retains the existing declared host training recipes. Those
recipes use different base learning rates and Adam betas across hosts, and
include explicit learning-rate stress conditions. It is **not yet evidence of
one universal numerical optimizer preset**. The full per-case settings are
visible in `leaderboard.json`; architecture grouping cannot conceal a settings
change within a case.

Architecture variants are inspected development evidence from seed 0, with
unchanged live thresholds and five final passing checks in a complete
24-observation curve. EMA is separate. Diagnostic cases remain nonblocking.
No production defaults changed and no new seed sweep was run.

Rebuild this view from the archived runs:

```bash
python -m reports.transfer_suite.formulations.build
```

`architecture_cell` rejects mixing formulations, training settings, resources or
targets while allowing G/D architecture changes. These are evidence-grouping
checks, not behavioral leaderboard gates.

Validation: [19 grouping/protocol/suite tests pass](tests.log). The
[artifact audit](validation.json) verifies all 66 result references, both new
24-observation runs and their 56 archived source files.
