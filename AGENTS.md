**Experiment work: read [EXPERIMENTATION.md](EXPERIMENTATION.md) first.**
Read the [compiled experiment memory](reports/forge/EXPERIMENT_MEMORY.md) before
proposing another idea. Use Forge's declared gates and budgets; the initial
profile is provisional until its calibration criteria pass.

Use the terms and ownership rules in [RESEARCH_GLOSSARY.md](RESEARCH_GLOSSARY.md).

dont do seed experiments(same thing except different seed)
be token efficient
make it easy to tail the logs
summarize and give explanations, leaderboard, recommendations on experiments after completion
establish metrics/leaderboards and use them over viewing images
only keep one generated leaderboard per goal, the current one

For new comparisons, use protocol seed 0 and the repository's public
deterministic initializer. Hold each task's architecture, target/data law,
seen batch sequence, initial prior distribution and capacity, sampling, update
budget and evaluation cadence fixed across trainer candidates. Use one global
trainer configuration across
tasks; declare the trainer delta explicitly, including changes to the recipe's
prior update policy or regularizer. Isolate constructor, data,
training-noise and evaluation RNGs, and checkpoint every consumed stream.
Fixed identity/zero fixtures and initialization diagnostics must be explicit
separate cohorts, never silently substituted for the shared baseline. Preserve
archived evidence under its original source, seed and initialization contract.

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
