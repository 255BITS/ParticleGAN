# Research reports and local artifacts

The [optimizer comparison report](forge/OPTIMIZER_TYPES.md) explains the public
Adam, SGDA, normalized and DualNorm update rules with pseudocode tables,
formulation-wrapper context and the recorded BCAP optimizer-screen results.

The behavioral baseline, learned-LR, locked-shared, paired-2D, smart-descent,
and transfer-suite studies keep their source, Markdown findings, and reproduction
inputs in Git. Generated results, metric curves, run logs, episode archives,
source snapshots, plots, and checkpoints are ignored and kept locally.

Existing outputs remain at their original paths. References to these outputs in
study writeups describe local files; fresh checkouts do not include them. Use the
commands in each study's README to generate outputs before running its report
builder or historical replay. Rebuilding a historical leaderboard requires its
complete local collection of runs.

The shared-default benchmark's frozen 19-test definitions and architecture
choices are in
[`default_comparison.json`](../benchmarks/transfer_suite/plans/default_comparison.json),
including reference artifact paths and SHA-256 hashes. These were extracted
without changing any jobs from commit `d77e9e8`. The small exact-replay fixture
in `tests/fixtures/shared_default_two_pole.json` retains only the metrics and
actions checked by the regression test, with its source commit and hash.

## Archived raw logs

Bulk stdout, per-update traces, JUnit logs, and large metric/state dumps are
kept outside the tracked tree. [The archive index](log-archive.json) records
each removed file's exact commit, Git blob, and byte size. Existing report
links to these logs point to the preserved archive commit. Compact results,
qualification receipts, protocols, and reproduction sources remain tracked.
Forge's large generated JSON leaderboards are archived in the same index;
`python -m experiments.forge compile` rebuilds them from the retained compact
receipts. Markdown leaderboards and experiment memory remain available in Git.

To recover a log locally, use `git show ARCHIVE_COMMIT:PATH > /tmp/run.log`,
with the commit and path from the index. The archived runs retain their original
source and qualification identities. This storage cleanup changes no training
code, recipe, evaluator, or acceptance result.
