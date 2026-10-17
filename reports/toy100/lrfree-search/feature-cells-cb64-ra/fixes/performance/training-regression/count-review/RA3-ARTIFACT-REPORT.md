# Frozen RA3 saved-artifact review

Status: PASS for source, initialization, state, graph and replay evidence. The toy quality gate remains FAIL.

The original immutable `integration/review/audit_learned.py` was run with `--validation validation-ra3 --variant CB64-RA3` and the new output directory `integration/review/ra3-artifact-audit-state-review`. Both completed training records are VALID; toy and MNIST replays are PASS/VALID. All ten checkpoint steps per problem, matched initial model/prior bytes, full 2000-update budgets, real-data cursors, artifact hashes and GPU-typed loss/state fingerprints agree with their frozen receipts. Replay starts at1000 and checks all ten updates through1010, including exact restored state. Its only semantic exclusion remains the observational `birth_death.last.eval_seconds` field. The checker created no CUDA context.

The supplemental audit compares actual saved metadata with `integration/iteration-3/READY.json`, which the original checker's custom-validation mode records without enforcing. The complete package digest is `995cebbe51d3532ef94e75ac6f6a44ce8751cbdaf1508efece4650b8bab32586`; backend schema4, kernel `bounded_local_dv12_lineage` and mass policy `reference_topology_vacancies_unique_parents_v4` match the readiness record and all actual checkpoints/endpoints. Coordinate caches and feature snapshots are absent from saved backend state.

All20 training graphs and four replay graphs pass independent sparse shape, integer-index, no-self-link, uniqueness, symmetry and degree checks. Initial graphs are empty. Saved diagnostic edge counts agree with the tensors. Final graph degree is bounded by8, with candidate bound72:

| Problem | Final edges | Max degree | Ordinary copies | Isolation copies |
|---|---:|---:|---:|---:|
| Toy |406|5|486|0|
| MNIST |462|4|819|51|

Replay graphs are bit identical across branches:190 edges for toy and464 for MNIST at1010. Invalidating reciprocal links can leave masked -1 padding gaps; those are valid graph slots. The first supplemental run incorrectly required packed padding and stopped; its log is retained. The corrected audit checks the actual graph contract and passes.

## Completed quality evidence

| Problem | Final metrics | Training seconds |
|---|---|---:|
| Toy |Precision0.409912;17/25 modes; mass TV0.592935; clean-centre precision0.523438 with19 modes |950.49|
| MNIST |Class mass TV0.0413625;10 confident classes; confident fraction0.748291; mean classifier confidence0.916272 |112.00|

Toy fails all three original quality criteria: precision>=.9,25 modes and mass TV<=.1. MNIST has no newly introduced numerical gate. Valid initialization/replay and a bounded saved graph do not establish quality.

## Native harness limitation

Frozen RA3 `_generate(..., *, rows=None)` lacks the `indices` parameter named by the immutable native harness. Auto detection resolves plain, causing native evaluation to omit sampled row IDs and bypass known-copy candidates. Root completed the frozen learned/replay phase and stopped before native screens. Internal training and `trainer.sample` pass row IDs correctly. These results therefore qualify the learned/replay evidence only.

The separate AXIS-ID package supports the harness's fifth positional indices argument. Its independent API/cache review is in `AXIS-API-REPORT.md`; no RA3 package, lane, saved result or gate was changed.

Evidence: original checker output under `integration/review/ra3-artifact-audit-state-review/`; supplemental `audit_ra3_graph.py`, `ra3-graph-review.json`, retained attempt logs and `RA3-ARTIFACT-FROZEN.json`. All local review execution used CPU loads, zero optimizer updates and no new seeds.
