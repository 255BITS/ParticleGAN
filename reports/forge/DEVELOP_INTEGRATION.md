# Forge on E22 and Atlas

PRs #155, #223 and #221 integrate into `develop`; the release branch and tags
remain unchanged. This is software integration, with scientific receipts kept
under the source, prior, initialization and sampling cohorts that produced them.

## Public formulation and execution

- KA2 remains the public default. `get_recipe("k3p")` selects the earlier critic
  explicitly; Forge's K3P declaration no longer inherits a changing default.
  Fixed R1/R2 and BCap remain available through explicit `reg_arm` selection
  with their original L2 kernels and K3P optimizer. The completed formulation
  comparison retains its source and failed scientific verdicts.
- `get_recipe("e22")` and `get_recipe("atlas")` resolve complete presets before
  host overrides. Atlas adds automatic feature-cell selection with 128 cells and
  the settled reopen guard. A `Recipe.name` alone selects no mechanism.
- Schedule-free presets keep `total_steps=None`. Public `GANTrainer.max_steps`
  and `extend_execution` bound work independently of the formulation.
- The scalar trainer preserves the ordered `UpdatePolicy` lifecycle, schema-4
  policy/checkpoint state, serial backward mode and state-selected serving while
  adding Forge's MoG, enumeration and named-stream contracts. Old schemas are
  resumed with their original implementation, without changing their version.

## Applicability and evidence

Forge's ordinary learned-MoG prior remains distinct from the explicit equal-mass
particle-cloud cohort needed for E22 row controls. Behavioral component hosts
do not implement the E22 lifecycle. Current Forge task definitions certify
their declared clean/live laws and existing mechanism observations. E22/Atlas
policy tasks therefore return `BLOCKED` before reservation and at construction;
new policy-aware tasks need their own frozen sampler, weight selection and
control observations. Proposed `e22` and `atlas` cards earn no qualification.

Policy API conformance checks record the public policy's independently owned
birth/death stream (policy seed + 6), full state and external bound. These checks
prove construction and continuation, not quality. The clock audit exposes
KA2's call-800 blend transition; schedule-free LR alone is not a clock-free
qualification.

`check_develop_parity.py` compares the upstream Atlas tree with the integrated
tree in the same CPU/runtime cohort. Its fixed probes cover E22 kNN and Atlas
feature cells, both serial-backward settings, 840 updates per case, 105
population evaluations, four clean/noisy fast-or-EMA sample observations and
ten resumed updates. Every checkpoint field agrees after excluding only the
reaction wall-time diagnostic. This finite probe does not certify untested
hosts or replace scientific gates. Historical RA14/RA15/RA16/RA17 receipts keep
their original identities; merging or archiving unchanged research does not
create new gate runs.

## Artifact storage

The repository's `AGENTS.md` requires bulk logs and per-update streams to remain
local or in artifact storage. `reports/log-archive.json` preserves commit/blob
identities for removed files. Compact numerical receipts, source/protocol
bindings, Markdown boards and experiment memory remain tracked; large generated
JSON boards can be rebuilt with `python -m experiments.forge compile`.
