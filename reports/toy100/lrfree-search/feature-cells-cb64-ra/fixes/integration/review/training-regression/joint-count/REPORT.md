# Joint K+2K proposal, frozen CPU qualification

This proposal retains v4 supported-cell mass transport and adds the frozen even-fit inside/outside recovery partition. Both original K and refined 2K conditional count families use the common Q/(3K) correction, Q=.05, including empty bins. It preserves the boundary, original support score/null, geometry, optimizer recipe and quality gates. Eleven CPU action cases, four statistical/checkpoint contracts and independent review pass. It is a separate source and does not modify RA2, RA3, v4 or frozen2K evidence.

## Fixed-input results

| Input | Mass moves | Support moves | Isolation moves | Ordinary total |
|---|---:|---:|---:|---:|
| Saved toy1000 | 3 | 24 | 0 | 27 |
| Saved toy2000 | 4 | 23 | 0 | 27 |
| Supported imbalance, no flags | 51 | 0 | 0 | 51 |
| Supported imbalance plus46 small holes | 51 | 0 | 46 | 51 |
| Original nominal | 0 | 0 | 46 | 0 |
| Original rare-hole | 0 | 5 | 41 | 5 |
| Actual supported table balanced | 0 | 0 | 0 | 0 |
| No eligible parent | 0 | 0 | 0 | 0 |

The no-flag imbalance addresses the frozen2K regression directly: all ordinary actions come from the original v4 mass phase. The phase matches the frozen v4 method's exact action arrays and RNG when both receive the same freshly corrected K evidence. Its function body is unchanged apart from the private method name. The stricter common correction produces two original mass discoveries at each saved toy step; archived Q/K masks are not reused.

Both ordinary phases share the original51-slot budget. Mass plans first. Support reserves mass children/parents, starts from the post-mass supported ledger, and subtracts gross mass outside deaths/inside births from refined certificate capacities. It never restores certificate capacity for opposite-direction mass moves. Final ordinary births obey cell/group vacancies. Isolation receives the combined plan once, reserves all previous rows and subtracts only supported ordinary deaths. The supported-mass+small-hole case has97 total actions because legacy isolation retains its separate small-flag repair bound; the ordinary budget remains51. It proves the supported-death subtraction branch as well as parent/child uniqueness.

The51/52-flag guard cases yield7 support+37 isolation and7 support+0 isolation, respectively. Restricting the existing ordinary max_moves argument to3 yields exactly3 mass moves and no support moves. These are contract inputs, not changes to a recipe or gate. The rare-hole case preserves both legitimate rare survivors and the rare group's final count of2. Nominal/rare reference features are reconstructed exactly from the established fixture; its clean table supplies the mechanical count sample because the original artifact omitted emitted features. Those cases are not iid emission replays. Saved toy cases use the original emitted features, geometry, flags and planning RNG.

## Null and independent review

K and2K families overlap, so they are evaluated separately with actual sample sizes and share one3K union-bound correction. Each conditional test has its own pooled-total null; no test-family independence is assumed. The exact overlapping-family reference covers all25/29/32 allocations of three pooled cases, including empty bins and unequal sample sizes. Their family rejection probabilities are1/6435,28/2145 and35/25194 at Q/6; p-values match rational enumeration within2.9e-15. Independent review additionally checks unequal5/11 samples with family probability1/182. These checks and the fixed-partition argument establish the implemented conditional law, not a repeated adaptive training guarantee.

Independent review verifies geometry/support/RNG equivalence to frozen2K, no odd/fake boundary leakage, empty/degenerate/tied categories, freshly corrected masks, exact v4 mass actions, shared rows/ledgers and certificate reservations. Its targeted exhaustion case spends32 mass births into the left cell, fully consuming that refined inside quota. Despite32 unused parents and84 physical left vacancies, support sends its remaining19 actions to the right cell and zero left. This catches quota reuse that uniqueness and vacancy caps alone would miss.

The audited six-case input is preserved as `inputs-attempt1.pt` with the original preparation receipt. The final input adds only the supported-mass+small-hole contract; final11-case and statistical receipts hash that input. Production source remains the independently audited `ada7f248…cade9f`. No source, seed, cutoff, boundary, gate or score sweep occurred.

## Remaining finite-bin power limitation

After27 ordinary moves, both saved toy cases have24 ordinary slots left but no further locally certified action capacity. Toy1000/2000 have43/45 observed inside-deficit cells without local discovery,61/141 unused eligible inside rows, and60/105 physical births after cell/group caps. These are sampled deficits and capacities, not oracle distribution labels. The local birth certificates, rather than the ordinary budget or total parent availability, are exhausted.

The requested descriptive global2 comparison aggregates the same fixed even-fit partition. Inside/outside p-values are approximately5.58e-223 and6.33e-189 for the two steps at the correctly accounted candidate cutoff Q/(3K+2)=Q/194. Recomputing the existing families at194 leaves their discovery counts unchanged. Aggregate residual certificates, distinct parents and physical vacancies permit a capacity bound of24 additional moves in each saved case. This diagnosis changes no controller or action law. It structurally motivates root's separate3K+2 proposal before candidate selection; no learned quality follows from the capacity bound.

## Delivery and remaining qualification

`JOINT-COUNT.patch` applies to frozen v4. `READY.json` freezes production sources, protocol, checker/runner, original v4 method reference, final and archived inputs, receipts and independent audit. The root-only CUDA runner uses frozen CPU geometry/boundary and current-device real/fake category assignment, records every source before/after, and compares the original mass method on the same device. CPU mode of that exact runner passes all11 cases. CUDA uses the existing case seeds on a CUDA generator, never CPU RNG bytes.

```bash
/tmp/pr38-default-env/bin/python -u -B /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/integration/review/training-regression/joint-count/contract_check.py --package-root /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/integration/review/training-regression/joint-count/pkg-joint-count --output /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/integration/review/training-regression/joint-count/gpu-contract --device cuda:0
```

Root owns CUDA and canonical matched toy/MNIST/replay qualification. Existing support rare false positives, adaptive-head/FIFO dependence and coarse topology limitations remain. No optimizer update, new seed or CUDA context was created locally.
