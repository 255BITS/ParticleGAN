# Serial GROUP-COUNT wrapper review

Read-only source and hash review passed for the normal execution path. All83phase inputs match the frozen READY; every owner numerical input is guarded. Both full-map checks run before the unchanged owner CUDA command starts. The shared serial lock is retained across the profiler, with parked-supervisor/outer-slot identity checks, recognized-child rejection, fixed GPU0UUID, deterministic math, TF32 disabled and0.2process memory fraction. No wrapper, numerical test or CUDA context was started here.

## Recovery limits

If the wrapper exits after launching its child, the profiler can survive while the serial lock releases. The existing jobs.json scanner does not recognize profile_anchor.py. Before another phase after interruption, verify the retained LAUNCH PID/startticks has exited. Preflight refusal creates no phase output; an exception after launch can leave partial evidence without PHASE-RESULT. Preserve logs and partial outputs and mark such a phase incomplete.

The review does not assert current parked-process state, numerical CUDA parity, speed or quality. Those checks remain at execution. Frozen source and owner READY bytes were left unchanged.
