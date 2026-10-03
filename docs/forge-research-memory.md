# Source-bound research recall

Forge reads compact configuration-search publications and the policy-family
inventory directly into recall. A newly committed concluded trial is discoverable
before compilation. The compiled memory, manifest and single
`reports/forge/publication-records.json` projection provide a reviewable snapshot.
Existing authoritative goal boards remain the rankings.

```sh
python -m experiments.forge recall --goal discriminator_stability --query 093c6f2bd417
python -m experiments.forge recall --goal discriminator_stability --query min_mass_ratio
python -m experiments.forge recall --goal policy-family-defaults --query prior-balance
python -m experiments.forge compile --check
```

Complete candidate, configuration, revision, study and record IDs outrank broad
token matches. Metric names and recorded failed bounds are searchable alongside
mechanisms and goals. A metric on a failed task is an observed final metric; its
presence alone does not establish that this particular bound caused the original
full-curve failure. Recall includes the recorded task reason and source board so
an agent can inspect the original explanation.

Every projection has `evidence_scope: published_summary`,
`qualification_input: false` and `qualification_reuse: false`. Publication
records never enter the board qualification or lifecycle reducer. They preserve
the recorded candidate/configuration identity, source/protocol/runtime/recipe
bindings and original receipt hashes. Policy cases retain their original
terminal grade separately from the study acquisition/hold grade; unreached cases
remain UNKNOWN. Source commits are not substituted for candidate revisions.
Study and trial cost summaries overlap and must not be added together.

The normalizer recognizes versioned Forge configuration-search reports, the
policy-family goal inventory and the bounded campaign completion receipt.
In-progress studies participate in freshness but are not called concluded.
Unrecognized schemas fail explicitly rather than silently fabricating results.
No raw log, checkpoint, per-update stream or remote `/ml2` path is opened. Compact
summary recall remains usable in a fresh checkout; independent regrading still
requires the original evidence and the separate artifact resolver.

`compile --check` performs a read-only check and exits nonzero for stale memory.
It checks scientific declarations, existing normalized records/receipt inputs,
compact authoritative publications and their bindings, the two memory reducers,
record/view counts, and the compiled memory/projection. It excludes generated
outputs from input hashes. History-catalog bookkeeping and unrelated source,
tests and documentation do not invalidate scientific recall; current inventory
drift is reported separately. `plan` and `recall` report the same CURRENT/STALE
status and a concrete refresh command, so stale compiled coverage stays visible
without hiding available live recall.

After changing one of those scientific inputs, run `forge compile --summaries-only` and commit
the compact generated memory, manifest and publication projection. Review the
resulting denominators and source links. This refresh preserves published
qualification tables and automation snapshots, retaining their original manifest
identity. It never recomputes an archived verdict from missing envelopes. New view
indexes are explicit pointers until original evidence supports their numerical
table. Ordinary `forge compile` retains its existing receipt-reducer behavior and
should be used when the original execution envelopes are available.
Default-discovered software tests verify
freshness and all concluded published trial IDs without training or artifact
hydration. Keep execution stdout local and easy to tail, for example:

```sh
tail -F runs/forge/audit-recall/pytest.log
```
