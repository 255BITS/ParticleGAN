# Automatic feature cells and the settled guard

[`ra13-settled.json`](../configs/100gaussians/ra13-settled.json) combines the
feature-cell population controls with the current caller-owned policy API and
the optional [settled R1 guard](r1-rotation.md#optional-guard-for-a-settled-game).
The configuration name identifies the shared training settings. Checkpoint
restoration and recovery after an R1 fire have been repaired with these same
settings.

## Use with caller-owned models

```python
import json
from pathlib import Path
from particlegan import GANTrainer, Recipe

fields = json.loads(Path("configs/100gaussians/ra13-settled.json").read_text())
fields.update(num_particles=N, z_dim=Z, batch_size=B)
recipe = Recipe(**fields)
# Construct and initialize G, D, and prior for your application first.
trainer = GANTrainer(recipe, G, D, prior=prior, serial_backward=True)
trainer.step(real_batch)
print(trainer.state_dict()["backend_selection"])
samples = trainer.sample(10_000, output_noise=True)
```

The caller supplies the device, networks, initialization, population, latent
width, batch size, and data stream. The recorded Gaussian problems use output
noise starting at .029; set this scale to match your own data. For an external
optimizer loop, use the existing [`E22Policy` lifecycle](e22.md#caller-owned-updates)
with the same Recipe and explicit parameter-group roles.

## How the backend is selected

Selection happens when the first real batch supplies the output shape. It does
not read task names, seeds, quality scores, or a training budget.

| Capability | Selection |
| --- | --- |
| Sufficient finite-population resolution and at most eight raw output coordinates | Feature cells |
| Insufficient population resolution or more than eight raw output coordinates | Existing kNN controls |
| Caller-owned generation/features, encoder, or router | Existing representation controls |
| Routed rows | Existing routed controls |

The finite-population condition is
`ceil(N / (.05 * (floor(N / 2) + 1))) <= floor(.05 * N)`.
Small tables fail this condition because the calibration sample cannot resolve
the controller's existing 5% evidence budget. The complete raw moment frame is
limited to eight coordinates. MNIST therefore uses kNN; its result does not
establish image-scale feature-cell performance.

Feature cells use 128 cells, rank eight, chunk size 256, bounded latent
perturbations, certified real-anchor births, and conditional raw-moment checks.
When `auto` selects this backend, generator and learned output-noise base rates
are multiplied by .25; table and critic base rates retain their configured
values. This factor is an empirical backend calibration. Explicit
`birth_death_backend="feature_cells"` leaves rate selection to the caller.

After a recorded R1 fire, mean correction can use the occupied subset of its
frozen groups. Missing EMA groups receive zero direction and weight; their
mass is not redistributed, and all odd observations remain in the bounded
witness. The count and birth controls handle missing support. This prevents
one empty group from vetoing correction throughout the population. The
original action budget and 95% serving-coherence requirement still apply.

## Checkpoints and compatibility

Backend selection, reasons, applied base rates, population controls, serving
geometry, lineage, and guard state are checkpointed. A checkpoint taken before
the first batch retains a pending selection. Invalid selection or guard packets
are rejected before model or optimizer mutation.

Device conversion preserves repeated tensor references in nested optimizer and
stationarity state. This matters when moved rows rebase a shared displacement
history after loading a CPU-mapped checkpoint onto CUDA.

CPU planning operations use explicit CPU allocation when the caller sets a
CUDA default device. Restore also requires exact agreement between the saved
backend output shape and initialized feature FIFO shape before any state is
committed.

The ordinary Recipe defaults and default checkpoint fields remain unchanged.
The new backend and guard are explicit options. CPU Adam health checks also
avoid initializing CUDA for CPU-only optimizer groups.

## Validation

The qualification results and original task definitions are recorded in the
[generalization report](../reports/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/README.md).
Larger architectures still need their own memory, throughput, and quality
measurements. Current evidence covers the recorded fixed-seed tasks; it does
not establish scaling laws or robustness over seeds.
