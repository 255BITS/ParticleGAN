# Research reports and local artifacts

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
