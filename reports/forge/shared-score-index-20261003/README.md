# Shared ParticleGAN score index

The [current family leaderboard](../technique-inventory.md) contains the generated family/view totals and links to each family's experiments, metrics and pass criteria. Refresh it with `python reports/forge/regenerate_technique_inventory.py`.

The [performance-by-test matrix](../test-performance.md) shows complete recipe cells, zero-pass tests, execution errors and separately typed feasibility witnesses.


## Scoped non-Atlas discoveries

[Verified complete recipes, exact protocols and all outcomes](../test-performance-evidence/nonatlas-cell-moment-full-runs.json): **45 unique full runs** in three separately scoped seed-0 rounds. Aggregate outcomes are **35/36 PASS**; the **9 unique controls** are **5/9 PASS**. These are execution-ledger counts, not a pooled score, independent control repetitions or an ordinary whole-suite winner.

| Registered round / task IDs | Executed source | Aggregate PASS / total | Control PASS / total | Inclusive child seconds |
| --- | --- | ---: | ---: | ---: |
| cell-moment-round<br>`vector_unequal_mass`, `vector_unequal_width`, `vector_anisotropic` | `f90962a5b351` · seed 0 | 12/12 | 2/3 | 746.970010416 |
| cell-moment-transfer-round<br>`vector_spiral`, `vector_two_broad`, `vector_overlap` | `f90962a5b351` · seed 0 | 12/12 | 1/3 | 730.446546641 |
| cell-moment-acquisition-round<br>`vector_unequal_width_published`, `gaussian1d_smoke`, `ring16_acquisition` | `a21587094267` · seed 0 | 11/12 | 2/3 | 745.665159237 |

All four aggregate objectives pass the anisotropic, unequal-width, unequal-mass, broad, overlap and spiral scoped cases in the first two rounds. Acquisition preserves **Bures ring FAIL**. The Gaussian acquisition smoke uses any scheduled full pass plus an independent same-state draw; ring executes 1,600 updates with 96 observations and its original 400-update schedule horizon. Published-discriminator width is a distinct task ID. Metric regressions remain visible; no objective is declared universally superior.

[Paired finite-realization mechanism prefix](../test-performance-evidence/nonatlas-finite-realization-prefix64.json): two 64-update arms, both **INCOMPLETE** against the original 7,000-update staggered100 quality gates. Prefix charge **20.104729663 seconds**; disjoint charge including the three full rounds **2243.186445957 seconds**. No prefix PASS is imported into full quality.

Preparation/audit errors and exclusions are retained in the pinned receipts. Original source, generated arrays and checkpoints remain local-only; individual consumed CUDA packets were not separately retained. These discoveries change display coverage only. Selected family rows, ordinary qualification and defaults remain unchanged.

## Qualification and accounting

Each model/configuration keeps its own source, law and declared denominator. Existing C6 diagnostics retain their 26-slot protocol; standalone API rows use their own task gates. MoG + particles (per task) is a mixture of separate suite hosts, not a hybrid-model claim. Named diagnostic passes do not fill ordinary cells or combine into a family score. INVALID supplies no accepted numerical grade; original N5 word execution remains BLOCKED. No family default or comparable speed winner is established.

The named campaign's predecessor-inclusive charge is **1466.669318475062 / 10500 seconds**, reserve zero: GPU0 **234.82608077581972 / 7500**, GPU1 **1231.8432376992423 / 3000**. The separate half_base attempt charged **556.4301753160544 / 900 seconds**, with zero reserve and overrun. Prior charges are included once; the original C6 word INVALID is unchanged.

[Historical report archive](ARCHIVED_REPORTS.md) · [Original snapshot index](index.json). The archived source, scores, costs and links retain their original bytes and separate scopes.
