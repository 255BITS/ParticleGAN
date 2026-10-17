# Exact planner optimization

## Scope

The proposal is an exact performance correction to the final 3K+2 count
planner. Compose only the four entries in `READY.json` → `ast_splices`:
one new `_group_integer_allocate` helper and the mass, local support, and
global support planning methods. Their certificate calculations, capacities,
reservations, shared ledgers, action order, and RNG calls are retained.

Each method batches the four disjoint group allocations into bounded G×K
matrices (G≤K≤64). Stable largest remainders retain the original cell ties.
A single six-vector CPU transfer supplies the existing cell loops with their
integer quotas, pool lengths, and group IDs. Python group lists preserve the
original ascending group/cell order. No semantic state or schema changes.

`pkg-PLAN-FINAL` is the review source for these splices. Root composes its
snapshot methods with AXIS-ID lineage/training and the count owner's final
class/settings. The original `_mass_topology` remains byte/AST identical.

## Fixed CPU proof

Use one CPU thread, hidden CUDA, and `/tmp/pr38-default-env/bin/python`.
`final-cpu-02.json` records 6,144 exhaustive small integer allocation cases,
14 fixed saved/supported/guard/no-parent/budget variants, and 100 captured
group allocation calls across all three phases. Every action, parent, quota,
certificate detail, comparison, work counter, topology and RNG state matches
the same final count law. Fixtures and source maps are checked before/after.

The 14 variants include the count owner's null/opposite global signal cases;
saved toy1000/2000 each exercise 24 actual global moves. Guard51/52 exercise
10/11 global moves. No features, quality trajectory, or alternate seeds are
generated. CPU profiler operator counts support a hotspot diagnosis, without
assigning elapsed GPU evaluation time to one cause.

## Root-only GPU paired contract/profile

Run `profile_plan_pair_gpu.py --package-root COMBINED_PACKAGE --output FRESH_JSON`
with the explicit reference/input/contract paths in `READY.json` → `gpu_command`.
The reference must contain the same final 3K+2 count law with the performance
splices absent. The runner rejects a changed class/certificate family before
opening the GPU context, then proves exact actions/RNG/accounting and every
captured batched quota on physical GPU0. Warm/cold operator profiles and three
fixed-call timing observations are reported in separate fields. Preparation,
state capture, and parity checks are excluded from measured planner calls.

The first passing final-law receipt remains in `final-cpu-01.json`. The first
freeze attempt correctly rejected a changed contract-tool hash, recorded in
`freeze-attempt1.json`; the numerical source and inputs were unchanged. The
same short proof was repeated against the updated contract tool before freeze.

## Optional MST evidence

`pkg-PLAN` is a separate older 3K prototype with the copied bounded-distance
MST selection. `mst-cpu-02.json` proves 14 fixed/tied/duplicate/constant cases
and reduces cold CPU scalar reads 382→4. `mst-cpu-01.log` retains a failed test
assertion that incorrectly expected a distance computation for K=2, where
both original and proposed topology skip the MST. The production method was
unchanged between those direct MST checks.

MST is excluded from `pkg-PLAN-FINAL`, `PLAN.patch`, and `ast_splices`. Including
it would require a separate GPU proof of edge ordering/lengths/cut/groups via
`--include-mst` against an otherwise identical count law before a new freeze.
