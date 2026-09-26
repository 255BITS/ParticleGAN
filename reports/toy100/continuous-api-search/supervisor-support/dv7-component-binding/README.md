# DV7 public component rate binding — unapplied support patch

This scratch snapshot adds the missing DV7 discriminator multiplier to the public component LR helper. It does not modify the active worker, its controller, trainer arithmetic, observations, or any qualification result. The parent reports that DV7 failed `unequal_mass` (0/24 observations; minimum eigenratio .0213 below .15) and research moved to DV8. This API preparation makes no quality claim. DV8 requires a separate source review before transfer; this patch deliberately recognizes only DV7's extra critic role.

## Provenance and files

All package files in `snapshot/` came from the immutable DV7 declaration ZIP, not the changing active checkout:

`/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T225101Z-1555568/data_drift_mobility/20260926T225101Z-1555576/repo/reports/data-drift-api/runs/dv7-single/source.zip`

ZIP SHA256: `451555bef4660196277ffbaa75c9cf4f56990e30adcf99e063977d76ea1b843e`.

- `baseline.json`: original source and package hashes, verified against the declaration manifest.
- `snapshot/`: independent package snapshot and CPU contract tests.
- `dv7-component-binding.patch`: unapplied patch for `particlegan/recipes.py`, the root export in `particlegan/__init__.py`, and the new contract test file.
- `results.json`, `cpu-contract-results.xml`: 23 passing CPU contracts and source-preservation proof.
- `trainer-rate-source.py.txt`: the two exact rate-assignment blocks extracted from the frozen `GANTrainer._step` AST.
- `detector-device-note.md`: separate, unpatched default-device issue.

## API contract

`scale_learning_rates(..., controller=controller, critic=critic)` accepts the critic module or its exact optimized parameters. DV7 always requires this explicit binding, even while the current critic multiplier is 1. Critic and prior parameters must each occupy complete, separate optimizer groups. If a module includes frozen/unoptimized parameters, pass only its optimized parameters. This prevents partial role assignments; a public critic optimizer's `.critic` ownership also catches a mistaken generator binding. With ordinary Adam, the supplied critic identity is the caller's role declaration.

Matching controller type and variant, complete base-rate lists, critic ownership, group separation, and prior/critic overlap are checked before any LR changes. The helper does not serialize or advance the controller. Callers save controller state alongside optimizer state and retain each optimizer group's original unscaled rate.

The frozen trainer first calls `observe_game` on the previous critic record, then `observe_real(real)`. Its rate assignment is:

```python
network, prior_scale = controller.current_scales()
G_lr = G_base * network
prior_lr = prior_base * prior_scale
D_lr = (D_base * network) * controller.critic_scale()
```

The helper reads that already-observed controller state and uses the same multiplication order. It never calls an observation method. `learning_rate_scales` continues returning the shared `(network, prior)` pair; its documentation now directs callers to the role-aware helper for DV7. `DataDriftController` is exported at package root so this component API is self-contained. Existing finite schedules and older symmetric continuous policies retain their prior executable behavior.

## Verification

23 CPU contracts passed in 2.00 seconds. They cover open and quiet states, module and parameter bindings, horizons 0/7000/1 billion, repeated calls without compounding, G/D/prior arithmetic, restored controller equality and reopening, missing/mismatched controllers, invalid critic roles, mixed groups, overlap, malformed base rates, finite schedules, explicit finite transitions, and DV6 compatibility. Tests construct tiny CPU modules and optimizers; they do not execute a forward pass, backward pass, optimizer step, training loop, or GPU operation.

Static verification confirms every other package file is byte-identical to the frozen ZIP, including `continuous.py`, `training.py`, `k3p.py`, and `ka2.py`. The executable AST of `learning_rate_scales` and the existing finite helper loop/return are unchanged. Patch application was checked against another pristine snapshot without applying it.

From this directory, repeat only when a later change justifies it:

```bash
cd snapshot
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -m pytest tests/test_dv7_component_rates.py -q --junitxml=../cpu-contract-results.xml
```

```bash
git -C patch-check apply --check ../dv7-component-binding.patch
```
