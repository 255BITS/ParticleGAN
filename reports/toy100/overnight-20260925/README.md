# Completed overnight evidence — September 25, 2026

All 49 attempts in the three overnight pools finished. This review checked the
saved results behind the strongest candidates rather than treating attempt
ledger rows, acquisition screens, or different regularizer configurations as
interchangeable full passes.

The [leaderboard](../continuous-practical-leaderboard.md) gives the current
comparison. The subsequently authorized [gap-fill batch](../gap-fill-20260925/README.md)
tests missing problems using the saved formulations.

## Evidence

[`evidence.json`](evidence.json) records the 49 attempts and 82 checked results,
their original absolute paths, SHA-256 hashes, and saved result snapshots under
`results/`. Snapshots are losslessly gzip-compressed JSON. `artifact_sha256`
identifies the decompressed original; `snapshot_sha256` identifies the gzip file.
Two additional derived checks evaluate the 300-update extension after the
standard ring hold. This is selected evidence for the leading formulations,
not an exhaustive export of all 2,435 overnight ledger rows.

Original attempt reports are preserved as
[`k3p-final.md`](k3p-final.md), [`k3-final.md`](k3-final.md),
[`k3g-final.md`](k3g-final.md), and [`rg5-final.md`](rg5-final.md).
Where an attempt's interpretation differs from its raw result, the leaderboard
uses the raw result and explains the distinction. In particular, the original
RG5+A2 .1/.1 target-shift result is `UNCONFIRMED`: its live recovery checks pass,
but the attempt did not execute a matched frozen control.

## What was missing at review time

| Exact formulation | Missing qualification |
|---|---|
| K3P | 13 focused toys, three further vector toys, rotated100, staggered100 |
| K3G | Three further vector toys, grid100, rotated100, extended hold, target shift |
| RG5+b-cap, no A2, base floors | All 19 transfer toys, rotated100, extended hold, target shift |
| K3/P1 | Three further vector toys |
| RG5+A2, a_r1r2, .1/.1 floors | Matched frozen control for the saved recovery run |

These are separate from known failures: K3P's base-floor recovery, K3's late
hold/recovery, and K3's staggered100 seed 1235 already have failing measurements.
Changing learning-rate floors or combining K3P with RG5 creates a new candidate.

These snapshots preserve measurement evidence. The original paths refer to
local attempt workspaces; the snapshots alone are not a portable training
environment. The gap-fill directory separately preserves executable candidate
and harness sources, commands, fixture hashes, and runtime provenance.
