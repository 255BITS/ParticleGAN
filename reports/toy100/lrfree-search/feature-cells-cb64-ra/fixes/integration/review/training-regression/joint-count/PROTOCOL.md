# Joint mass and support count refinement

Preserve frozen RA2, RA3, v4 recovery and the separate 2K support proposal. This private refinement addresses the documented 2K regression in supported-cell mass transport. All local checks are CPU-only, with CUDA hidden before Torch import, deterministic operations, one CPU thread, existing saved inputs and streams, and no optimizer updates or new seeds.

## Common count family

Retain the original v4 K-cell partition and the frozen proposal's even-fit inside/outside 2K partition. The support boundary, score, Q=.05, tie rule and geometry are unchanged. Compute the original K-cell exact conditional Hypergeometric p-values and the refined 2K p-values using their respective real/fake counts and actual sample sizes. Both families use the common cutoff Q/(3K), retaining empty categories. Recompute decisions at that cutoff; never reuse archived Q/K or Q/(2K) discoveries.

The overlapping K and 2K tests have 3K hypotheses. Under the same fixed-partition iid null, each p-value is conditionally super-uniform given its own pooled count, and a union bound controls any discovery by Q. No independence between the two overlapping test families is required. Their counts each sum to the actual sample size separately; concatenating them into a purported single categorical sample would be invalid. Repeated adaptive learned-head/FIFO decisions still have no cumulative iid guarantee.

## Fixed action order and shared reservations

1. Plan original v4 mass transport first using the re-thresholded K comparison, unchanged eligible-parent/surplus/rare-target logic and the existing floor(Q*N) ordinary budget. Broad flags retain v4's count-certified flagged-only donor branch; otherwise supported surplus rows remain available for mass transport.
2. Plan support transport using only the remaining ordinary budget. Exclude all mass children and parents from support parent and death candidates. Start its supported ledger at original unflagged counts minus unflagged mass children plus mass parents. Subtract mass outside-category deaths from refined outside death capacities and mass inside-category births from refined inside birth capacities, clamp at zero, and expose raw/spent/residual capacities. Opposite-direction mass actions do not replenish these certificate budgets. Certified flagged-outside donors and certified inside p>Q parents then obey residual certificates, remaining cell/group vacancies and unique-parent capacity.
3. Supply the combined ordinary child/parent arrays to the unchanged isolation guard and assignment policy once. Isolation excludes all ordinary children and parents, subtracts only unflagged ordinary deaths from its supported ledger, and adds all ordinary parents.

Both ordinary types together are at most floor(Q*N). Isolation retains its original small-flag guard and remaining flagged-row bound. Supported mass deaths can coexist with isolation repairs, so combined ordinary plus isolation actions may exceed the ordinary 5% budget as in v4. Group caps apply to the combined supported ledger; per-cell caps apply to ordinary actions. Legacy isolation may redistribute births among cells within a group. No prior child may be deleted twice, no prior parent may be reused or deleted, and no unsupported death may be double-subtracted from supported mass.

## Predeclared qualification

- A no-flag, pointwise-supported two-cell mass imbalance must produce certified mass actions. Compare its mass phase against the exact frozen v4 method using the same corrected K evidence and stream.
- Saved nominal and rare-hole mass/repair fixtures retain supported survivors, rare-group mass and valid isolation. Their missing even-reference features will be reconstructed only from the established fixture generator with its fixed existing inputs, not by changing labels or fitting to observed quality.
- Saved toy1000/2000 broad-hole inputs compare v4, frozen2K and joint3K response on unchanged CPU geometry, flags, fake features and stream. Report any loss from the required stricter common correction without selecting another cutoff.
- Absent eligible parents, empty/tied/degenerate bins and small guard51/52 cases remain valid. A small-guard ordinary+isolation plan must use the exact post-ordinary supported ledger and distinct rows; a deliberately restricted ordinary budget must bind both phases jointly.
- Enumerate a small exact overlapping K+2K categorical null with retained empty bins and compare rational p-values/FWER. Verify even boundary, original score/geometry, actual multiplicity and checkpoint policy isolation.

Freeze one qualified source/patch/READY plus a root-only CUDA fixed-input contract runner accepting --package-root and --output. Root selects a candidate before canonical quality; this artifact provides no learned-quality verdict.
