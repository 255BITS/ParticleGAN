# Research log retention

PR #139 removes 265 console logs and 13 per-update metric streams (8,855,653
bytes). Every scored field in those metric streams matches the retained
`metrics.json` alongside it; the streams additionally contained transient
training-loss output. Future console logs and `metrics.jsonl` files under this
report tree are ignored by Git and remain available in local run directories.

Qualification results, failed results, source bundles and hashes, initialization
fixtures, commands, event records used by regrading, and numerical summaries
remain committed. K3P's compressed raw results and all eleven pinned source
files are unchanged. `tests/test_k3p_selection.py` verifies the selection against
those files, the 22 declared passes, and the hold/extension/recovery evidence.

Historical console output remains available in the
[pre-cleanup tree](https://github.com/255BITS/ParticleGAN/tree/b979d3c90bdbf8f58c62d759bc3c5fa94cdf18c8/reports/toy100).
Commands in older reports that tail logs refer to the original local runs;
replaying a run creates new logs in its output directory.
