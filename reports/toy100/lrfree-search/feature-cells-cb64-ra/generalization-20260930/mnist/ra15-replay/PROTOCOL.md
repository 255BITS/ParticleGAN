# Original learned RA15 CUDA replay

The original RA13 Toy/MNIST checkpoint1000 remains the input, with its actual source/config/run-receipt hash and data cursor256000. Original learned training remains labelled fresh RA13; this lane performs no fresh2000update training.

`replay.py`, primary sampler, schema validator and native-control functions are unchanged from closed RA14-r2. The only common-module change is the candidate variant label. The bridge routes the immutable RA15 package and byte-identical config using the independently pinned no-fire proof. Model initialization, seed314159, fixed real streams, batch128, table1024x128, serialized backward, strict native/CPU-map restoration, all ten per-update fingerprints, endpoint bytes and original noisy sample law are retained.

CPU preparation loads two checkpoints on CPU, confirms complete source recipe/schema, typedfires0, saved global/private CPUuint8 RNG buffers, and the exact `last_block is blocks[-1]` alias. It constructs no models, performs no sampling or updates, and initializes no CUDA. Live restoration remains a required numerical check.

Root launches after preflight and shared-lock scheduling:

```bash
/tmp/pr38-default-env/bin/python -u -B /ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/mnist/ra15-replay/launch_replay.py
```

The root coordinator holds the existing serial GPU lock across the original process, checks the parked PID identities without signals, and verifies all frozen inputs before/after. The unchanged original runtime uses physicalGPU0 UUID and fraction0.2. Exactly Toy/MNIST × native/CPU-map × ten updates are required, total40. Each latest native continuation must also match the pinned original native control. Only `birth_death.last.eval_seconds` is excluded from semantic equality.

Tail `logs/RA15-partial-recovery-replay.log`. After actual completion, root runs `close_replay.py`. Missing/error/quality outcomes are never inferred from source compatibility. Every failed prior RA13/RA14 replay or preparation remains retained. This latest40update replay and the actual latest full suite are separate required qualification gates.
