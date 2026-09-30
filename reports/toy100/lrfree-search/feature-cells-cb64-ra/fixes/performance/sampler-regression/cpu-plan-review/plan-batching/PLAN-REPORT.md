# Planner findings and proposal

The concrete warm hotspot is repeated GPU scalar conversion inside four
per-group largest-remainder allocations and the quota/group matching cell
loops. The proposal batches only these allocations and copies bounded
metadata once per phase. It preserves all three final count certificates,
gross reservations, supported ledgers, row order, and RNG calls.

CPU proof passed 6,144 exhaustive integer cases, 14 complete planner variants,
and 100 actual grouped quota calls against the same final 3K+2 law. All output
tensors, detail/accounting dictionaries, topology/work, and RNG states are
exact. The final count class is AST identical after reverting only the three
declared planning methods. The original statistical helper and MST remain
unchanged. Independent two-phase/helper prototype audit passed; final global
phase audit is requested separately.

Warm `ordinary_transport` plus `select_parents` CPU operator counts:

| Fixed fixture | Original scalar reads | Proposed scalar reads | Original nonzero | Proposed nonzero |
|---|---:|---:|---:|---:|
| Saved toy1000, all three phases | 2,148 | 79 | 666 | 591 |
| Nominal, isolation active | 844 | 332 | 375 | 367 |
| Supported mass imbalance | 66 | 31 | 26 | 24 |
| Supported mass plus small holes | 79 | 44 | 44 | 42 |

The remaining nominal reads largely come from unchanged isolation proposal
loops. These operator counts identify barriers; they do not explain the
earlier grid run's elapsed-time regression or predict quality. The prepared
root-only GPU runner reports exact contracts, operator counts, and elapsed
timing observations separately under the same final law.

Frozen RA3, AXIS-ID, all count owner sources/fixtures, the canonical harness,
and earlier failed evidence were preserved. `PLAN.patch` is reviewable against
the final global-count source. Root should compose the four verified AST
splices, retain the final count wrapper and AXIS-ID lineage/training, then run
GPU paired contracts before the authorized native validation.
