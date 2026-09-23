# Shared-default search results

**Best supported score: 17/19 with one unchanged recipe and explicit discriminator
choices. No all-pass shared default yet.** The same recipe scores 15/19 with the
reference discriminator profile. Public presets remain 8/19 proposed and 5/19
current master; this search has not changed package defaults.

| Shared recipe | Required | Data | Images | Live total | Reference D profile |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shared_c6`, including declared D trials | 9/9 | 4/6 | 4/4 | **17/19** | 15/19 |
| `lr00425_prior2`, cap coefficient 3 | 8/9 | 3/6 | 4/4 | 15/19 | 15/19 |
| `ratio_g34_d68_p85` | 8/9 | 3/6 | 4/4 | 15/19 | 15/19 |
| `relative_cap_05`, adaptive step limiter | 7/9 | 2/6 | 3/4 | 12/19 | 12/19 |

The leader uses Rp logistic, b_cap coefficient **6**, κ **1.25**, particle spread
**.05**, no particle L2, Adam **(0,.99)**, and absolute **G/D LR .00425, particle
LR .0085 on every test**. Hold rates for the first 60% of the budget, then cosine
toward 5%. No optimizer setting or objective is adjusted by example.

## What improved

The cap-6 recipe already passed all required and image cases. Two discriminator
changes now supply the additional data passes:

| Case | Supported discriminator | Parameters | Final passing observations |
| --- | --- | ---: | ---: |
| Anisotropic | Raw MLP plus a small Fourier branch, width 64, two layers, Softplus | 5,796 | 8 |
| Overlap | Raw-input MLP, width 96, three layers, Softplus | 19,009 | 10 |

A raw-input SiLU discriminator, width 128 and three layers, also passes
anisotropic with six final passing observations. All three witnesses reproduce
exactly in independent integration replays, including all 24 live/EMA
checkpoints and optimizer receipts. [Replay checks](runs/shared-architecture-replays/README.md).

Architecture choices are separate from recipe identity. **This is not one
universal discriminator.** The complete raw-Softplus profile passes 3/6 data
tests; raw-SiLU passes 1/6. The leaderboard retains the original reference-profile
score and every architecture failure. [All 28 architecture trials](runs/shared-discriminator-search/README.md).

## Two remaining blockers

- **Rare mixture mass:** the closest tested cap-6 discriminator still collapses
  a component's narrow direction: minimum normalized variance **.01891**, below
  **.15**. Occupancy and aggregate sample quality alone do not expose this.
- **Unequal widths:** raw-SiLU meets every final bound, but only the last **one**
  observation passes. It needs at least **five** consecutive final observations.
  The original reference architecture also fails. Longer training would be a
  separate toy, not a replacement pass at this budget.

The cap-3 leader still ends mode-hold at **7/8 modes**. Its overlap metrics pass
only the last three observations. These failures and measured bounds are now
visible directly in the main leaderboard.

## What the other agents found

- Six global G:D:particle rate ratios produced 66 episodes. The best complete
  row scores 15/19 and trades overlap for an anisotropic failure. The other
  completed row scores 13/19; four screens remain INCOMPLETE.
  [Rate study and all failures](runs/shared-ratio-search/README.md).
- Three generic, role-blind Adam proposal caps produced 31 candidate episodes.
  The selected cap scores 12/19, fixing none of the four original failures and
  adding trajectory, anisotropic and stripe regressions. The two other caps
  remain INCOMPLETE. Identity controls reproduce exactly; the mechanism and
  its actual attenuation receipts are explicit. [Adapter findings](runs/shared-adapter-search/FINDINGS.md).

This round adds **125 candidate episodes**: 66 rate, 31 adapter and 28
architecture trials. The primary importer validates **388 episodes across 29
recipe entries**, including all earlier runs and 38 public-baseline episodes.
Fourteen entries cover all 19 tests; fifteen are partial. Separate parity replays
and stopped metadata checks are retained without adding leaderboard points.
All failures, complete curves, source archives and actual optimizer settings are
preserved. There are no seed sweeps or relaxed thresholds. EMA is separate.

**35 focused tests pass.** Independent review prompted checks that reject
renamed/no-op duplicate architectures, undeclared optimizer transformations and
adaptation receipts that contradict their equation. The revised checks pass;
the primary importer reports no validation errors.

## Reproduce and continue

The [leader recipe cards](leading_candidates.json) reproduce both original
19-case reference profiles. Run the added architecture trials with:

```bash
python -u -m benchmarks.transfer_suite.shared_discriminator_search \
  --plan reports/transfer_suite/unadjusted/leading_discriminator_trials.json \
  --output /tmp/shared-d-witnesses > /tmp/shared-d-witnesses.log 2>&1
tail -f /tmp/shared-d-witnesses.log
```

That command runs both declared discriminators on both target cases, retaining
their cross-failures. The strongest next target is preserving rare-component
variance and making the unequal-width success persist, with this exact shared
recipe. A new update rule must remain identical across tasks and earn its own
full-suite row. These inspected development tests do not establish transfer to
real networks.

[Primary leaderboard](README.md) · [Contributor instructions](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md).
