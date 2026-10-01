# Actual PR155 E22 versus ParticleGAN Atlas convergence

The baseline is exactly PR155 commit `cabe2084284db923d525918cbf3e18de6f20faac`
with its shipped original E22 research config. CPU resolution confirms that
the recommended `get_recipe("e22", num_particles=20000, z_dim=2,
batch_size=2048, output_noise_std=.029)` has identical training fields; only
the recipe's name label differs. Optimizer reopening and anchor release remain
enabled. No Atlas feature backend or settled guard is added to the baseline.

Atlas uses the immutable latest RA17 package and the byte-identical settled
config. Each run owns the original native rotated100 host: seed 1234, 20,000
rows, latent dimension 2, batch size 2,048, identity Linear generator, 128×3
Fourier critic, deterministic critic initialization seed 1, original independent
two-real-batch stream order, serialized backward, and 1,500 complete updates.
Targets change by 30° before updates 501 and 1,001.

## Observations and metrics

The original four draws and 20,000-point gates remain at updates 0, 500, 1,000
and 1,500. Extra 4,096-point draws use the original fixed visualization seeds
77 and 78 at every 10 completed updates. Global RNGs are forked, all private
trainer/evaluation/birth streams and external data cursor are checked exactly,
and extra feature-geometry cache/work writes occur in disposable dictionary
copies. Reading state for the guard preserves the lazy DV12 record object and
does not swap served parameters. Every extra observation checks the complete
current public state, module flags, parameter versions, gradients and cursor.

There are 151 observed clouds, plus two explicitly marked `target_shift`
frames at update 500/1,000. An event retains the exact immediately preceding
cloud while target centers jump to the next original orientation: zero
intervening updates, no interpolation or invented particle motion. Float32
clouds retain the actual sampled precision.

`dense-frames.npz` has `frames[T,4096,2]`, nondecreasing `steps[T]`, additional
`angles[T]` in radians, initial `centers[100,2]`, string `event_kind[T]`,
`capture_hq[T]`, `capture_modes[T]`, explicit sample counts and JSON metadata.
The visualization metric uses those 4,096 samples, distance ≤0.09 and ≥10 HQ
points per mode. It is a separate diagnostic. The unchanged original gates
use their separate 20,000 samples and seeds 1637/1636. Each turn needs at least
95 modes and HQ at least 90% of that variant's update 500 baseline.

## Required numerical observer checks

Atlas compares two independent original checkpoint continuations, updates
1,001–1,010, with and without extra observations. The external CUDA data
generator is advanced exactly twice for each of the 1,000 previous updates,
then the original RA15 checkpoint restores all owned state and global RNG.
The modern source is the already reviewed exact RA17 bridge.

There is no current-base E22 checkpoint at update 1,000. Its two comparison
workers therefore execute the first 20 updates from the original initialization,
covering two original reaction boundaries. This is explicitly an early sampling
state contract, not a late E22 quality result. During each full capture, the
same complete state-preservation guard is checked for every extra draw,
including both drift periods. Only `birth_death.last.eval_seconds`, measured
wall time, is excluded from comparison between independent training workers.
Losses, tensor bytes (including NaN payloads), model/optimizer/controller state,
private/global RNG, module flags, gradients, parameter versions and external
data cursor must otherwise match. Full captures require a passing receipt from
both variants, bound to this exact source freeze.

## Root execution

Preparation and `--check-only` do not initialize CUDA. Agents do not launch
numerical work. Root uses these commands with the original shared mutex,
physical GPU0, 20% per-process cap, and strict checks of the original parked
process identities. Every attempt uses a new output path.

```sh
/tmp/pr38-default-env/bin/python -u -B run_capture.py --parity \
  --output /ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/visualization-pr223-convergence/parity-attempt-1
/tmp/pr38-default-env/bin/python -u -B run_capture.py --full \
  --parity-receipt /ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/visualization-pr223-convergence/parity-attempt-1/PARITY-COMPLETION.json \
  --output /ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/visualization-pr223-convergence/full-attempt-1
```

The serial launcher starts separate package-isolated processes for E22 and
Atlas. Source guards cover their packages, all 28 public core modules, configs,
native host/scorer sources, original runner, continuation inputs and adapters.
An original quality FAIL is retained as a quality FAIL even when the process
completed and source integrity was valid.
