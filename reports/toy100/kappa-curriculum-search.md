# Shared critic-cap curriculum search

These scratch experiments applied one update-count rule to every selected
host: start with κ=1.25, then switch or linearly decrease to κ=1.0. The
remaining settings matched the earlier 18/19 recipe, with legacy global
output RNG and network floor .01. Seeds, budgets, data, architectures, and
gates stayed fixed. The source archives retain actual cap values at every
gradient-penalty call, alongside optimizer and noise receipts.

| Predeclared rule | Strict older-host screen | Failure |
| --- | ---: | --- |
| Switch after update 400 | 6/9 | Mode hold, blobs4, unequal mass |
| Linear 400→600 | 6/9 | Mode hold, stripes2, blobs4 |
| Switch after update 100 | 7/9 | Mode hold, bars4 |
| Linear 100→200 | 8/9 | Mode hold: only four terminal passing checks |
| Linear 100→160 | 0/1 | Mode hold: seven of eight modes |
| Linear 100→240 | 0/1 | Mode hold: three of eight modes |
| Linear 100→300 | 0/1 | Mode hold: seven of eight modes |

The best row ends with all eight modes and HQ 1.0, but still has only seven
modes at step 1000. Its four passing checks at 1050–1200 do not satisfy the
five-check requirement. The final three refinements were screened on mode
hold first; all failed, so their other eight hosts were skipped according
to the predeclared rule. No full-19 replay or production schedule support
followed these failures.

All rows are **ineligible for the common-22 gate**: their actual κ varies
while the production recipe declares a constant. Their numerical verdicts
were independently regraded using the archived scratch implementations;
they are not stitched together with other configurations.

The local evidence directories below retain manifests, source archives,
compressed episodes, complete cap traces, and relocated regrades. Copies
matched every original file by SHA-256 (24, 24, and 9 files respectively):

- `artifacts/toy100-accuracy/kappa-curriculum-v1-39254ee/`
- `artifacts/toy100-accuracy/kappa-curriculum-v2-fe5c719/`
- `artifacts/toy100-accuracy/kappa-curriculum-v3-23a6e2a/`
