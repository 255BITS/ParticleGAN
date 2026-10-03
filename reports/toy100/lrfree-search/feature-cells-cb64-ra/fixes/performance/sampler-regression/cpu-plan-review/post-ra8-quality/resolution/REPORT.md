# Finite fitted-reference resolution proposal

**CPU source and mechanical contracts pass; quality is untested.** The private
package requests128 cells with the even-only average resolution cap
`min(requested, max(1, fitted_rows // max(1, effective_rank)))`.
Toy512/rank8 remains64 cells; grid10000/rank8 allows128. This regularizes
average fitted rows per rank and makes no per-cell occupancy guarantee.

Only feature_cells.py changes. Trainer and28 other modules are byte identical
to RA8. The prospective config changes only birth_death_cells64->128.
Noise, rates, model/prior averaging, serving geometry/expiry, population,
birth/copy/parent accounting, 5% ordinary budget, fixtures and evaluator gates
retain their original definitions. Every count and capacity uses actual K;
the common family remains K+2K+2 at Q/(3K+2), with empty bins retained.

## Exact matched saved-input reactions

| Saved toy input | Actual cells | Ordinary moves | Copies | New latents | Isolation | Exact plan/law/state/RNG |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
|1250|64|51|47|4|0|yes|
|2000|64|51|51|0|0|yes|

Both baseline request64/backend7 and proposal request128/backend8 execute the
actual maybe_apply path on the same saved current weights, FIFO, prior row
moments/history, graph and cloned CPU RNG. Fitted geometry, count evidence,
birth/copy plans/new latents, both tables, optimizer row state/history,
supported ledgers, graph, moved rows, row state, counters and semantic events
are bit identical. Each phase respects the same joint51-row ordinary budget;
children and copy parents/source seeds remain unique and disjoint. Model
weights/buffers, existing gradients and module modes remain unchanged.

Full numerical/backend state comparison excludes only declared backend7->8,
requested settings.cells64->128, the new resolution-policy setting, and the
original observational last.eval_seconds. Every other field, including all
work counters, paired serving stamp and private reaction stream, is exact.
This is a fixed ready-reaction contract using the frozen no-init saved-weight
fixture; it does not replay historical CUDA training. The fixture's requested
recipe metadata is explicitly prospective, not an old checkpoint migration.

## Edges and persisted state

Nine fixed cap cases and four fitted controls pass: minimum6, odd9, rank1 and
rank0/identical references. Degenerate metric tests stay disabled and all
support p-values equal1. Actual category/multiplicity/cutoff formulas agree
for tiny K1/K3; no level or gate is changed. A typed N801/rank1/request401
control uses even401/odd400 and actual401, preserving the ceil-even bound.

Thirteen malformed metadata controls reject atomically: backend7, wrong
resolution policy, requested versus actual cells, last-cell mismatch,
categories, multiplicity, cutoff, integer fields encoded as floats, q encoded
as string, and extra/missing partition keys. Fresh backend8 state roundtrips
exactly and discards ephemeral chart/head/axis caches. The existing trainer5
validates backend state before loading model/optimizer state; independent
trainer API/serving checks remain root-owned after composition.

Backend8 settings explicitly version the resolution law. Noninitial saved
stamps must match the even-count/effective-rank actual cap and last chart;
partition ordinal/types, category count and actual common multiplicity/cutoff
are checked strictly. Old backend7 has no production compatibility path.

## Freeze and limits

The initial source version was preserved under source-attempt1 with an
explicit original-path to archive map when independent review found Python
dict equality accepted integer metadata as floats. This was a static review
issue, before numerical tests. The corrected source/input seal06649bc0… was
created at08:48:38.260UTC, and helper seal97a7f482… preceded execution.
The sole numerical attempt passed and exited at08:56:27.267UTC. All70 source
and input guards, loaded checkpoint tensors and global CPU RNG remain exact.
No numerical helper failed; no CUDA, optimizer/training update, new seed,
quality emission, holdout fit, source/config promotion or quality claim was
made. Existing reaction-internal draws follow the unchanged noise law.

Independent source/state review passed on FC39558fb3… before execution.
Full source inverse AST proof covers only the declared helper/fit/schema/
settings/metadata changes; all other code remains RA8. Evidence is the design,
source/helper seals, cpu-attempt1/result.json and log, independent static
receipt, package patch and final READY/FROZEN. Root owns composition, CUDA
contracts, original replay and strict toy/grid quality acceptance.
