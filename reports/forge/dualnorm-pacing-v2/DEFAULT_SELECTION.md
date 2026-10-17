# BCAP-pure default selection

The owner requested making the winning optimizer settings the public BCAP-pure
default before merging PR #310. `get_recipe("bcap")` now selects the complete
zero-momentum dualnorm recipe below. Only the optimizer and its step sizes
change; the relativistic loss, fixed real/fake BCAP penalty (coefficient and
cap 1), constant schedule and other BCAP-pure settings retain their values.

| Setting | Winning default |
| --- | --- |
| Optimizer | Full `dualnorm` on G/E, D and learned prior |
| G/E step (`lr`) | 0.012 |
| D multiplier (`d_lr_mult`) | 1.5, giving a step of 0.018 |
| Prior multiplier (`prior_lr_mult`) | 2.5, giving a row step of 0.03 |
| Network momentum (`optimizer_momentum`) | 0 |
| Prior momentum | Always 0 |
| Epsilon | 1e-8 |

```python
from particlegan import get_recipe

recipe = get_recipe("bcap")
adam_control = get_recipe("bcap_adam")
```

Dualnorm normalizes updates according to parameter type. For a matrix gradient
`m = U S Vᵀ`, it replaces the singular values by one: the update is
`eta * sqrt(max(1, fan_out/fan_in)) * U Vᵀ`. This is the spectral-norm
steepest-descent direction, with a shape correction. Biases and other vectors
use `eta * m / (||m|| + eps)`. Each sampled particle row uses the same vector
normalization independently; unsampled rows do not move. Optional network
momentum uses `m = mu*m + g`, but the winner has `mu=0`, so `m=g`.
Near-zero matrix gradients are skipped. Full dualnorm has no Adam
per-coordinate moment adaptation. These step sizes are in different units
from Adam learning rates. The polar step does not enforce a Lipschitz bound
on the entire critic, and width/depth transfer has not been tested.

The 25-configuration optimizer-only pacing study completed 175 attempts at
protocol seed 0. The selected whole recipe passed **4/6 required Tier 1 gates**,
versus **3/6** for the matched dualnorm starter with G/E step 0.01, D/G 1.5 and
prior step 0.03. This is a pacing improvement within dualnorm; it is not a new
matched-source Adam comparison. Two-pole gains a sustained pass and both hold
tasks and word acquisition keep theirs.

| Required task | Winner result |
| --- | --- |
| Two-pole | PASS; 17 terminal passing evaluations, at least 5 required |
| Unused-token hold | PASS |
| AE hold | PASS |
| Five-word joint acquisition | PASS; all five modes, HQ 1.0, mass TV 0.01895 |
| Scalar Gaussian acquisition | FAIL; CDF KS 0.11428 exceeds 0.05 |
| 16-mode ring acquisition | FAIL; full assigned-component covariance error 9.61552 exceeds 0.85 |

The Gaussian's mean and width pass. The ring has all 16 modes, HQ 0.94385,
passing mass TV and minimum eigenvalue ratio, but the assigned tails still
fail its full-component covariance gate. Core-only or aggregate covariance
cannot replace that gate. Higher rates and positive momentum regress other
tasks. The original [readout](README.md), [results](results.json),
[analysis](analysis.json) and [measurement selection](measurement-selection.json)
retain their publication-time facts and frozen executed source
`a0f7e70e50427e0d3221d1d7f4cb4aac6e18b1be`.

This is an owner-directed API default, not calibrated scientific promotion.
The original study's `default_adoption=false`, failed gates, source bindings,
archived software receipts and immutable raw archive remain unchanged.
The publication-time report and software receipt describe the earlier Adam
public default; this separate decision supersedes that API status. The single
[current leaderboard](../technique-inventory.md) keeps its source-bound 4/6
measurement. No qualification results are rewritten and no experiments are
rerun to change the default.

`bcap_adam` preserves the previous BCAP-pure native Adam settings: G/D LR
0.00425, prior LR 0.0085, betas (0, 0.999). Halloween continues to use its
explicit historical Adam law and settings. Forge's versioned `forge-api-v1`
declarations preserve their historical `recipe_preset="bcap"` base through
the explicit Adam preset and original recipe label; existing configuration
cards and IDs retain their meaning. The winning cards already pin the full
dualnorm optimizer settings. Restore historical checkpoints using their saved
resolved `Recipe` fields, including the label, rather than today's defaults.

After merging and compaction, audit saved Gaussian CDF errors and ring tails
before deciding whether more updates or a new bounded search is justified.
No additional training is launched by this default-selection change.

The [separate software receipt](default-selection.json) records 528 passing
targeted tests across 18 modules, including default factories, unchanged cap
gradients, exact checkpoint continuation and rejection, and all 296 frozen
BCAP configuration identities. All 346 recorded runtime/test sources stayed
unchanged during that run. Forge validation, memory freshness and whitespace
checks pass. The 1,004 existing declarations, study artifacts and leaderboard
files checked against the pre-decision publication commit are unchanged; the
231,319,616-byte raw archive retains its original SHA-256. These software
checks do not change the 4/6 scientific result.
