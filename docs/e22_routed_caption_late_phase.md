# Paired caption convergence through native late phases

This new public-API test changes one factor from the frozen scalar-moment
fixture: each of the three fresh arms trains for2048 updates instead of512.
It asks whether useful untied-particle convergence survives the native penalty
phases that the shorter tests never executed. It is a generated caption-transfer
task, and a PASS does not establish an actual-caption or full-Supra fix.

From the repository root in the ParticleGAN Python environment:

```sh
CUDA_VISIBLE_DEVICES=0 python -m examples.e22_routed_caption_late_phase \
  --run --out runs/caption-late-phase-v1
```

Use a fresh output directory. The shipped protocol fixes physicalGPU0,
three2048-update arms, and900 seconds from startup through final writes. Exit0
means completed numerical PASS,1 means completed FAIL,2 means incomplete/error.
The command records the actual imported package identity and requires it to
stay unchanged during execution; it does not allowlist an old API revision.
No pretrained checkpoint, local Supra data, or manual authorization edit is
required. The bundled scalar profile is the only calibration asset.

The teacher and student execute the same frozen nonlinear host with positive
and source captions respectively. All13 caption constants, six sources,
flow-correlated latent contexts, 48FIT/12GUARD/48TEST rows, fixed FIT coordinate
standard-deviation units,29 frozen scalar-moment samples, named initializer
streams and native controls remain unchanged. Original Down/mainUp values are
shared; particles use sampledC, H/b0, Up0, a shared public128x4 bank, and native
D+G DV12. The untied arm retains its extra registered output head initialized
before optimizers/EMA. These are inherited factors, not additional changes.

Only terminal2048 raw physical float64 TEST48 RMSE sets the numerical gate:

- Untied improves both ordinary and shared-head controls by at least0.1%.
- Each of six sources suffers at most1e-6 RMSE harm against either control.
- Zeroing particle codes increases aggregate RMSE by at least0.1% and strictly
  increases every source's RMSE.
- Bank/query gradients are live on at least90% of updates2..2048; all sixC and
  particleUp norms are finite and positive.

Output metrics are offline; they never enter the native game, structural
guards, stopping or checkpoint selection. TEST snapshots512/768/800/1024/1536/2048
are all retained descriptive cohorts. Actual observed media states are
0/256/512/768/800/1024/1536/2048. No earlier endpoint must win.

Each training row copies public penalty `last_stats` and critic record
`calls`/`observed_steps` without invoking a diagnostic hook or advancing a
controller. It also reads post-update sigma and critic learning rates. Actual
phase counts and first observed blend are reported; absent/nonfinite optional
telemetry is explicit unavailable/null. No historical799/800-phase requirement
enters the performance gate. A changed late result alone would not identify
which of the naturally evolving penalty, rate, noise or routing controls caused
it.

The new envelope imports the immutable public factories/update/capture/restore
helpers; no old512 constant or module global is patched. The full prerequisite
proves initial equality, independent owners, public restore and the native
initial mainUp tangent, then every observation preserves owner/RNG/diagnostic
state. Terminal clean public restore/repeat is exact. Native learned
weights/gradients/moments must remain finite; diagnostic monitor scalars are
validated by the public native checkpoint API.

Render the saved actual observations and fixed native-loss/TEST curves after a
completed run:

```sh
python -m examples.render_e22_routed_caption_late_phase \
  --run-directory runs/caption-late-phase-v1
```

This separate CPU60-second command uses only retained tensors and JSONL rows,
with no models/native updates. NumPy/Pillow>=10.1 provide the goalGIF; Matplotlib
is loaded lazily for the fixed curvesPNG. Patch RMS is unscaled, axes and
initial-only color limits stay fixed, and the caption shows terminal2048 status.
Raw traces, tensor bundles and logs remain ignored. Compact results and actual
media will be separate publication files after execution and independent review.
