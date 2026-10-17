# RA10 original full grid artifact audit

This is a CPU, stdlib-only saved-artifact checker. Preparation reads source,
declarations and hashes. Runtime reads saved JSON and file bytes only. It never
imports Torch, loads a PT object, forwards a model, generates samples, trains,
calls a scorer, changes a gate or writes into a validation lane.

The fixed input is CB64-RA10 in validation-cb64-ra10, with the original Grid100
seed 1234, 7000 updates, N=20000, z=2, batch2048, all34 observations, terminal
steps6000/6250/6500/6750/7000, five20k clouds and100k holdout. Holdout target,
noise and latent seeds remain2835/2836/2837. Requested128, actual resolution
policy, mean common family3K+3, population gate, output noise and serving law
remain the frozen candidate's own settings.

The independent lane review and the lineage-owned original canonical monitor
are read-only inputs. A completed grid is audited only after the original
wrapper exits successfully, the root job_complete event exists, and the owned
monitor's stable canonical grid receipt and summary agree. The helper checks
original source/initializer/prior/stream/options/runtime receipts and exact
collector declaration: only evaluation_generate plain-to-indexed was declared.
The canonical host/scorers and all other collector checks are unchanged.

Exact pure ASTs from the original coverage and fidelity predicates are applied
to the recorded JSON. The original coverage score_run post-validation suffix
is used verbatim to reconstruct sustained coverage. All34 live observations,
both live/EMA event schedules, the final five predicates, the fidelity limits
and final verdict are cross-checked. The saved100k holdout lacks several raw
coverage fields; its frozen_pass bit is retained from the hash-bound original
scorer, which already rescored that exact saved cloud. This checker independently
reapplies the holdout fidelity predicate and original final Boolean conjunction;
it does not claim a new numerical rescore or quality experiment.

Strict JSON parsing rejects duplicate keys and nonfinite constants. The actual
final canonical scalar mean diagnostics and phase row/counter identities are
checked with the frozen production pure validators. This is a saved JSON
consistency check; full trainer schema/replay was qualified separately.

All completed run files, including original clouds/checkpoint, are bound by
byte hashes and sizes. Evolving queue and monitor summaries are copied as
snapshots, never pinned as live input files. The watcher writes only its own new
area, records PID/startticks/command, exits itself at completed grid, and never
signals any other watcher or numerical queue. Every private failure is kept.

Artifact status PASS means valid original evidence. quality_verdict is exactly
the original strict scorer's PASS or FAIL. A valid negative quality result is
archived honestly. Incomplete/unrun tasks retain pending, unverified status.

Preparation seals helpers and actual source maps before runtime. Runtime starts
only after parent authorization. A separate post-exit seal binds the closed
log, final receipt, report, all snapshots and final artifact manifest.
