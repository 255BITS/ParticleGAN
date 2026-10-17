# CUDA behavioral task execution

The eight behavioral hosts in the discriminator-stability view now execute
their public components, training, sampling and measurements on the requested
CUDA device. Their task cards reserve one GPU and forbid CPU fallback. This
creates a new execution cohort; archived CPU qualification is unchanged.

The retained host loops still own their targets, architectures, losses,
budgets and gates. Portable deterministic initialization and the fixed
cover-leftover feature bank retain their CPU-generated values before transfer.
Module-level tensor fixtures move without redrawing. Constructor, data,
training-noise and evaluation RNGs remain separate, and every consumed stream,
model and optimizer is saved in the local `component-state.pt` artifact.

Trajectory and residual-student feed their entire latent table to the generator.
The shared optimizer binder now declares all these actually used rows to the
public row-normalized optimizer. This fixes a capability error without replacing
the training loop or inventing sampled rows.

Twenty-four two-update CUDA fixtures cover all eight hosts with K3P, KA2 and
BCAP dual-normalization. They verify device placement, finite updates, consumed
stream checkpoints and restoration of the caller's CPU and GPU RNGs. A resource
declaration check also passes: **25 checks passed**, with no skipped CUDA cases.
These short software fixtures confer no scientific qualification.

```sh
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/python -m pytest tests/test_forge_behavior_cuda.py -q \
  --junitxml=runs/software/behavior-cuda.xml \
  > runs/software/behavior-cuda.log 2>&1
tail -F runs/software/behavior-cuda.log
```

The post-merge inventory rerun measures the new CUDA task bindings through the
ordinary Forge queue. Tier 2 still requires one unchanged complete candidate
to pass all six required Tier 1 tasks.
