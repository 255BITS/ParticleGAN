# Fixed even-fit support categories: CPU qualification

The proposal makes the diagnosed within-cell support deficit visible to categorical count evidence. It splits every existing fitted cell into inside and outside regions using the unchanged support score and an even-reference empirical .95 score boundary. It replaces the original K tests with one exact 2K count family at Q/(2K). Seven CPU contracts and the independent count review pass. This is a separate experiment based on frozen v4; it is not included in conservative RA3.

## Saved response

Both comparisons reuse the saved toy CPU geometry, query flags/p-values, emitted fake features and planning RNG. No projection, latent sample, model, optimizer, threshold or seed is selected from these results.

| Saved step | v4 ordinary moves | Proposed moves | Outside excess discoveries | Inside deficit discoveries | Unique parents | Supported deaths |
|---|---:|---:|---:|---:|---:|---:|
| Toy 1000 | 4 | 25 | 9 | 18 | 25 | 0 |
| Toy 2000 | 4 | 32 | 5 | 14 | 32 | 0 |

The ordinary budget is still 51 for 1024 rows. The original broad flagged sets of 887/710 rows still fail the unchanged isolation guard, yielding zero isolation actions. Every new ordinary death is flagged, outside, and in an outside category with certified excess. Every ordinary parent is inside, unflagged, p>Q under the original odd support null, and in an inside category with certified deficit. There are 86/166 eligible inside table rows; action quotas additionally respect certified directions, actual cell/group vacancies, and bounded unique-parent supply.

Odd real inside fractions are .8711/.8223, versus .0781/.0889 for emitted fake; refined category TV is .8018/.7373. The even-fit boundary has no .95 coverage claim for odd rows. Its generalization differs from the original odd-calibrated p>Q eligibility law, which remains unchanged. Counts compare the actual odd and fake fractions rather than assuming either fraction equals .95.

## Statistical and implementation checks

The boundary is frozen before odd calibration is consumed. Changing odd reference rows changes their null and counts but leaves mean/scale/basis/centers/cell scales/boundary/even counts and fitting RNG identical. Changing fake rows also leaves the boundary identical. Original geometry, support flags, support p-values and fitting RNG match v4 exactly. A second attempt to fit the count boundary is rejected. Ties are inside; all 2K bins remain in multiplicity even when empty. Degenerate metrics disable discoveries. Minimum-six-row, all-identical, duplicated/tied and empty fitted-cell cases produce finite p-values and valid counts; query refresh leaves the boundary fixed.

The exact conditional p-values match rational Hypergeometric enumeration on all allocations of pooled counts [4,4,8,0] and [4,6,10,0]. Their four-test family rejection probabilities are 1/99 and 35/25194 at Q/4, below Q=.05. Individual p-values are super-uniform at the tested exact rational levels. Independent review uses a distinct saved-input prefix and pooled case [6,6,4,0], checks all 29 allocations, and obtains family probability 28/2145 with p-values matching within 2.4e-15. These finite cases supplement the conditional-null argument in PROTOCOL.md; they are not learned-head or repeated-training false-positive guarantees.

The small-flag shared-ledger case yields seven ordinary actions plus 38 isolation actions, 45 total from 51 flagged rows. Isolation sees supported counts plus planned ordinary births exactly; flagged ordinary deaths are not double-subtracted. Children/parents are distinct across both plans, no parent is deleted, and combined supported group counts respect max(initial,target). The minimum-reference-mass group has no donor deletion or inflation. Ordinary actions also respect actual per-cell supported vacancies. Legacy isolation retains its group caps and own-cell parent policy; it can redistribute births among cells within a group.

A no-eligible-parent case has zero actions. The new checkpoint mass/count policy rejects a v4 state before mutation; same-policy reload succeeds and clears derived snapshot/latent caches. The feature/count work remains bounded by K center comparisons and 2K conditional categories. New fitting adds one even-reference score pass; comparison adds support-score region assignment, and test enumeration doubles the maximum category count. No full reference-neighbor search was introduced.

## Limits and next qualification

This replacement law preserves every unflagged survivor and therefore does not transport a purely supported categorical imbalance with no flagged outside donors. Existing support-score rare false positives are not resolved by this count partition. A flagged point near a rare distribution tail may still be eligible for removal if all count and vacancy conditions pass. The topology and parent kernel remain the inherited real-center and bounded-local laws.

The learned critic and FIFO are adaptive, so conditional iid count validity does not imply an unconditional or cumulative training guarantee. These CPU snapshots preserve their own reconstructed partition and streams, not archived CUDA projection RNG. More certified actions do not establish higher sample quality or acceptance. Root must run any CUDA contract and matched toy/MNIST/replay quality gates in its single GPU0 queue before integration. Keep conservative RA3 separate so its accounting/lineage effect remains identifiable.

`cpu-check-attempt1.json` and its log are retained. The first six tests passed; subsequent source edits only added comparison-shape validation and trainer diagnostic metadata, then a checkpoint test was added. The final seven-test receipt records the final audited source unchanged. There was no law/threshold/hyperparameter sweep. All local receipts report CUDA uninitialized and no optimizer updates or new seeds.

See [PROTOCOL.md](PROTOCOL.md), [cpu-check.json](cpu-check.json), [SUPPORT-COUNT.patch](SUPPORT-COUNT.patch), [READY.json](READY.json), and the independent [count-review.json](../../../../performance/training-regression/count-review/count-review.json).
