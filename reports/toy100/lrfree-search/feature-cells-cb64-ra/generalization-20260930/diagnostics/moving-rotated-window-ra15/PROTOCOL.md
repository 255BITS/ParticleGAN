# RA15 paired rotated checkpoint window

This is a causal diagnostic continuation, not a new full validation lane. Preparation performs no tensor loads, model calls or GPU operations. Numerical execution is explicitly launched by root after source review.

Both variants load the same retained original CUDA checkpoint at1000. The unchanged RA14 control must reproduce the original checkpoint1500 exactly, except `birth_death.last.eval_seconds`, and the original final gate0.8557/99. Candidate results are accepted only after control reproduction passes.

The adapter preserves the original host, constructor recipe, deterministic seed1 critic initialization, seed1234 owned streams, serialized backward, optimizer options, two real batches of2048 per update, target schedule, draws and gate source. Constructors exist only to reconstruct the checkpoint object; no fresh training precedes restoration. The external CUDA real generator is advanced with exactly2000 original real-batch calls before saved model/optimizer/controller and private/global RNG restoration. Both variants then execute exactly updates1001…1500 at60degrees.

The original500/1000 gate rows are inherited prerequisite evidence and explicitly labelled. Each variant executes the unchanged1500 snapshot and20k gate draw, plus the original identical seeded terminal draw. Latent/noise evaluation seeds remain1637/1636. The baseline0.9609, HQ bar0.86481, minimum95modes, raw axes cap8, alpha formula, action budget and95percent serving-coherence bar remain unchanged.

Runtime observation captures locals from the actual `freeze_moment` and `run_mean_phase` calls using a temporary Python return profiler. It reads already computed count tensors, never calls a model/chart or consumes RNG. Actual critic group IDs, missing EMA groups, frozen mask/mass, mean moves, witness bounds, sourcefires, applied LR and serving coherence are recorded in `steps.jsonl`. There is no evaluator geometry in recovery source or instrumentation.

Source freezing binds both raw package manifests, unchanged config, original sourcefreeze104guards, candidate sourcefreeze72guards/CPU closure, checkpoint1000 and retained checkpoint1500, original adapter, all host/scorer inputs, generated adapters and their support. Inputs are rechecked before and after each worker. Different packages run in separate processes under one continuously held shared GPU lock. Workers require that inherited exact lock descriptor; strict parked PID identities are checked without signals. GPU0 UUID is fixed and allocation fraction is0.2.

## Root launch

After `--prepare` and `--check-only` pass, root may run:

```bash
/tmp/pr38-default-env/bin/python -u -B /ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/diagnostics/moving-rotated-window-ra15/run_pair.py --run --output /ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/diagnostics/moving-rotated-window-ra15/attempt-1
```

Tail `attempt-1/control/run.log`, then `attempt-1/candidate/run.log`. Per-step telemetry is in each variant's `steps.jsonl`; restoration, numerical completion and source close are separate JSON receipts. A failed original quality gate remains valid quality-failure evidence. Any restore/control/source failure ends the pair and prevents candidate interpretation. Every attempt uses a new output directory.
