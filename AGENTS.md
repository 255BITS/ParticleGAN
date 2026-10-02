**Experiment work: read [EXPERIMENTATION.md](EXPERIMENTATION.md) first.**
Read the [compiled experiment memory](reports/forge/EXPERIMENT_MEMORY.md) before
proposing another idea. Use Forge's declared gates and budgets; the initial
profile is provisional until its calibration criteria pass.

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

Bind trained gates to their actual recipe, prior, initialization, budget and
sampling law. Check these against the qualified evidence before attributing a
failure to a formulation change. Clean and noisy served results are separate
cohorts; matching training settings alone does not make their gates comparable.

Toy tests must follow the [public-API test contract](reports/toy_audit/api_contract/README.md):
execute through the ParticleGAN API, declare a numerical pass/fail metric, and
provide an actual-training GIF that illustrates the test's goal. Reform an
incompatible question into an explicitly scoped, runnable variant, retaining its
useful controls and the original evidence identity.
