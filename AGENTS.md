dont do seed experiments(same thing except different seed)
be token efficient
make it easy to tail the logs
summarize and give explanations, leaderboard, recommendations on experiments after completion
establish metrics/leaderboards and use them over viewing images

Keep bulk research logs and per-update metric/event streams out of Git. Store
raw stdout, JSONL traces, JUnit logs, checkpoints, and tensor/state dumps locally
or in an artifact archive. Commit compact reports, final metrics, provenance
receipts, and reproduction sources instead. When removing tracked logs, retain
their exact archive commit/blob identities and repair report links; do not
rewrite qualification results or rerun unchanged experiments just for a merge.
Never force-add ignored bulk logs. Keep new execution logs easy to tail.
